"""End-to-end checks of Algorithm 1 with mock models (no GPU, Ollama or ComfyUI needed)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from vspr import prompts as P
from vspr.comfy import ComfyGenerator
from vspr.functions import phi_sco
from vspr.io_utils import read_json, stable_seed
from vspr.judge import feedback_text
from vspr.llm import LLMResult, extract_json, split_thinking
from vspr.pipeline import VSPRPipeline, VSPRSettings, export_samples, select_best
from vspr.functions import RoleConfig

REPO = Path(__file__).resolve().parent.parent

# level = number of refinements of the prompt that made the image -> (DMS, p(object present))
DEFAULT_LEVELS = {0: (0.2, 0.9), 1: (0.5, 0.9), 2: (0.9, 1.0), 3: (0.3, 0.6)}


class FakeJudge:
    """phi_ans stand-in: reads the prompt written into the fake image file."""

    model = "fake-mllm"

    def __init__(self, levels=None):
        self.levels = levels or DEFAULT_LEVELS
        self.calls = 0
        self.unloads = 0

    def probe(self, desc, image, b):
        self.calls += 1
        level = Path(image).read_text().count("refined")
        dms, pa = self.levels.get(level, (0.5, 1.0))
        return {"count_a": {"0": 1 - pa, "1": pa, "2": 0.0, "3": 0.0},
                "count_b": {"0": 0.0, "1": 1.0, "2": 0.0, "3": 0.0},
                "separate_p": 1.0, "picture_p": 0.0, "closer": "same", "closer_order": ["x", "y"],
                "stacked_p": 0.0, "b_covers_a_p": 0.2, "a_covers_b_p": 0.1,
                "overall_p": dms, "describe": f"image {b}"}

    def unload(self):
        self.unloads += 1


class FakeLLM:
    """phi_rea stand-in: appends ' refined' to the current prompt."""

    def __init__(self, think=False, suffix=" refined"):
        self.calls = Counter()
        self.messages = []
        self.think = think
        self.suffix = suffix

    def chat_json(self, model, messages, schema, options=None):
        self.calls["rea"] += 1
        self.messages.append(messages)
        user = messages[-1]["content"]
        prompt = user.split("Current prompt P: ")[1].split("\n")[0]
        thinking = "deep thoughts" if self.think else ""
        return LLMResult({"explanation": "because", "prompt": prompt + self.suffix}, thinking=thinking,
                         raw='{"explanation": "because"}')

    def unload(self, model):
        self.calls["unload"] += 1


class FakeGen:
    name = "fake"

    def __init__(self, root: Path, batch: int):
        self.root, self.batch, self.calls, self.negatives = root, batch, 0, set()

    def generate(self, prompt, seed, negative="text, watermark"):
        self.calls += 1
        self.negatives.add(negative)
        d = self.root / f"{abs(hash((prompt, seed)))}"
        d.mkdir(parents=True, exist_ok=True)
        files = []
        for b in range(self.batch):
            f = d / f"{b}.png"
            f.write_text(prompt)  # stand-in for an image; FakeJudge reads it
            files.append(str(f))
        return files

    def release(self):
        pass


def make(tmp_path, llm=None, judge=None, **kw):
    s = VSPRSettings(**{"max_iterations": 3, "batch_size": 4, **kw})
    llm = llm or FakeLLM()
    judge = judge if judge is not None else (FakeJudge() if s.use_vqa else None)
    gen = FakeGen(tmp_path / "imgs", s.batch_size)
    return VSPRPipeline(llm, gen, RoleConfig("reasoner", {}), judge, s, tmp_path / "run"), llm, gen


DESCS = ["a cat behind a dog", "a book hidden by a vase"]


def test_call_counts_match_paper(tmp_path):
    pipe, llm, gen = make(tmp_path)
    pipe.run(DESCS)
    M, B, N = 3, 4, len(DESCS)
    assert gen.calls == N * (M + 1)               # (M+1) batches of B images
    assert pipe.judge.calls == N * B * (M + 1)    # every image of every batch is inspected
    assert llm.calls["rea"] == N * M              # N_LLM = M per description (no LLM in phi_sco)
    assert gen.negatives == {"text, watermark"}


def test_first_round_uses_description_and_shared_seed(tmp_path):
    pipe, _, _ = make(tmp_path)
    pipe.run(DESCS)
    for idx, d in enumerate(DESCS):
        seeds = set()
        for i in range(1, 5):
            meta = read_json(pipe.round_dir(idx, d, i) / "images.json")
            seeds.add(meta["seed"])
            assert meta["prompt"] == (d if i == 1 else d + " refined" * (i - 1))
            assert meta["negative"] == "text, watermark"
        assert seeds == {stable_seed(42, d)}
        assert not (pipe.round_dir(idx, d, 4) / "reasoning.json").exists()  # no phi_rea after round M+1


def test_best_prompt_selected(tmp_path):
    pipe, _, _ = make(tmp_path)  # DEFAULT_LEVELS peak at round 3 (two refinements)
    pipe.run(DESCS)
    out = tmp_path / "eval"
    export_samples(pipe.run_dir, DESCS, out, upto_round=4, batch_size=4)
    manifest = read_json(out / "manifest.json")
    assert {m["round"] for m in manifest} == {3}
    assert {m["dms"] for m in manifest} == {0.9}
    assert len(list((out / "samples").glob("*.png"))) == len(DESCS) * 4
    names = sorted(p.name for p in (out / "samples").glob("*.png"))
    assert names[0] == "a book hidden by a vase_000004.png"
    # iter-1 export only sees rounds 1-2
    export_samples(pipe.run_dir, DESCS, tmp_path / "it1", upto_round=2, batch_size=4)
    assert {m["round"] for m in read_json(tmp_path / "it1" / "manifest.json")} == {2}


def test_score_formula(tmp_path):
    pipe, _, _ = make(tmp_path, max_iterations=1, dms_weight=0.25)
    pipe.run(DESCS[:1])
    s = read_json(pipe.round_dir(0, DESCS[0], 1) / "score.json")
    # round 1: DMS 0.2 on every image; OPS = p(cat) * p(dog) = 0.9 * 1.0
    assert s["dms"] == pytest.approx(0.2) and s["ops"] == pytest.approx(0.9)
    assert s["score"] == pytest.approx(0.25 * 0.2 + 0.75 * 0.9)
    assert s["dms_weight"] == 0.25 and len(s["per_image"]) == 4
    assert phi_sco(0.2, 0.9, 0.5) == pytest.approx(0.55)


def test_tie_goes_to_earlier_round(tmp_path):
    r = lambda i, s: {"round": i, "score": {"score": s}}  # noqa: E731
    assert select_best([r(1, 0.5), r(2, 0.7), r(3, 0.7)])["round"] == 2
    assert select_best([r(1, 0.7), r(2, 0.7), r(3, 0.7)])["round"] == 1
    pipe, _, _ = make(tmp_path, judge=FakeJudge(levels={k: (0.5, 1.0) for k in range(4)}))
    pipe.run(DESCS[:1])
    export_samples(pipe.run_dir, DESCS[:1], tmp_path / "e", 4, 4)
    assert {m["round"] for m in read_json(tmp_path / "e" / "manifest.json")} == {1}


def test_resume_does_not_recompute(tmp_path):
    pipe, llm, gen = make(tmp_path)
    pipe.run(DESCS)
    assert gen.calls > 0
    pipe2, llm2, gen2 = make(tmp_path)
    pipe2.run(DESCS)
    assert gen2.calls == 0 and llm2.calls["rea"] == 0 and pipe2.judge.calls == 0


def test_feedback_reaches_the_reasoner(tmp_path):
    pipe, llm, _ = make(tmp_path, max_iterations=1)
    pipe.run(DESCS[:1])
    rd = pipe.round_dir(0, DESCS[0], 1)
    fb = feedback_text(DESCS[0], read_json(rd / "judge.json")["results"])
    rea = read_json(rd / "reasoning.json")
    assert rea["feedback"] == fb and rea["mode"] == "vqa"
    system, user = llm.messages[0][0]["content"], llm.messages[0][1]["content"]
    assert system == P.v5_system("vqa", 4)
    assert user == P.v5_user(DESCS[0], DESCS[0], fb)
    assert rea["explanation"] == "because" and rea["raw_output"] == '{"explanation": "because"}'


def test_without_vqa(tmp_path):
    pipe, llm, gen = make(tmp_path, use_vqa=False)
    assert pipe.judge is None
    pipe.run(DESCS)
    for i in range(1, 5):
        rd = pipe.round_dir(0, DESCS[0], i)
        assert not (rd / "judge.json").exists() and not (rd / "score.json").exists()
    assert llm.calls["rea"] == len(DESCS) * 3
    system, user = llm.messages[0][0]["content"], llm.messages[0][1]["content"]
    assert system == P.v5_system("novqa", 4)
    assert user.endswith("(no images: predict the likely failures)")
    assert read_json(pipe.round_dir(0, DESCS[0], 1) / "reasoning.json")["feedback"] is None
    # output = last round, iter k = round k+1
    export_samples(pipe.run_dir, DESCS, tmp_path / "e", 4, 4, select="last")
    assert {m["round"] for m in read_json(tmp_path / "e" / "manifest.json")} == {4}
    export_samples(pipe.run_dir, DESCS, tmp_path / "e1", 2, 4, select="last")
    assert {m["round"] for m in read_json(tmp_path / "e1" / "manifest.json")} == {2}


def test_vqa_needs_a_judge(tmp_path):
    with pytest.raises(ValueError):
        VSPRPipeline(FakeLLM(), FakeGen(tmp_path, 4), RoleConfig("r", {}), None, VSPRSettings(), tmp_path)


def test_long_prompt_truncated_and_recorded(tmp_path):
    long = ", " + ", ".join(f"detail number {k}" for k in range(30))
    pipe, _, _ = make(tmp_path, llm=FakeLLM(suffix=long), max_iterations=1)
    pipe.run(DESCS[:1])
    rea = read_json(pipe.round_dir(0, DESCS[0], 1) / "reasoning.json")
    assert rea["truncated"] and rea["words"] <= 60 and len(rea["prompt_raw"].split()) > 60
    assert rea["prompt_raw"].startswith(rea["next_prompt"])
    assert read_json(pipe.round_dir(0, DESCS[0], 2) / "images.json")["prompt"] == rea["next_prompt"]


def test_reasoning_trace_saved(tmp_path):
    pipe, _, _ = make(tmp_path, llm=FakeLLM(think=True), max_iterations=1)
    pipe.run(DESCS[:1])
    rea = read_json(pipe.round_dir(0, DESCS[0], 1) / "reasoning.json")
    assert rea["raw_thinking"] == "deep thoughts" and rea["explanation"] == "because"


def test_json_helpers():
    assert extract_json('blah ```json\n{"a": {"b": "}"}}\n``` end') == {"a": {"b": "}"}}
    assert extract_json('x {"prompt": "p"} y') == {"prompt": "p"}
    t, a = split_thinking('<think>reason</think>{"a": 1}')
    assert t == "reason" and a == '{"a": 1}'


@pytest.mark.parametrize("backend", ["sdxl", "cascade"])
def test_comfy_workflow_patching(tmp_path, backend):
    gen = ComfyGenerator(backend=backend, workflow_path=REPO / "workflows" / f"{backend}.json",
                         cache_dir=tmp_path, batch_size=4)
    wf = gen.build_workflow("a cat behind a dog", 123)
    texts = [wf[n]["inputs"][k] for n, k in gen.node_map["positive"]]
    assert texts and all(t == "a cat behind a dog" for t in texts)
    assert [wf[n]["inputs"][k] for n, k in gen.node_map["seed"]][0] == 123
    node, key = gen.node_map["batch"]
    assert wf[node]["inputs"][key] == 4
    # the prompt must reach every CLIPTextEncode that feeds a positive input
    positives = {v["inputs"]["positive"][0] for v in wf.values() if "positive" in v["inputs"]}
    for nid in positives:
        if wf[nid]["class_type"] == "CLIPTextEncode":
            assert wf[nid]["inputs"]["text"] == "a cat behind a dog"
