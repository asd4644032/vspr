"""Algorithm 1 (VSPR), executed stage by stage over the whole dataset.

For every description D (M = vspr.max_iterations, B = t2i.batch_size):

    P_1 <- D
    for i = 1 .. M+1:
        I_i <- G(P_i)                                    B images, same seed in every round
        R_i <- phi_ans(I_i^b)  for every image b         fixed questions (vspr.judge)
        s_i <- phi_sco(R_i) = a * DMS_i + (1 - a) * OPS_i
        if i <= M:
            T_i, P_{i+1} <- phi_rea(D, P_i, feedback(R_i))
    return the round with the highest s_i (ties -> the earlier round)

"VSPR iter k" is the best of rounds 1..k+1. In the w/o VQA ablation (vspr.use_vqa = false)
the images are not inspected, phi_rea sees only D and P_i, and the output is the last round.

Each stage runs for all descriptions before the next stage starts, so every model is loaded
once per stage and only one model needs to be in GPU memory at a time. All intermediate
results are written to disk; re-running the same command resumes.

Files per round (outputs/vspr/<run>/items/<idx>_<slug>/round_<i>/):
    images.json     prompt, seed, negative, image paths
    judge.json      the B probe results and their diagnosis            (VSPR only)
    score.json      DMS, OPS, s, a, per-image values                     (VSPR only)
    reasoning.json  phi_rea input feedback, explanation, raw output,     (rounds 1..M)
                    refined prompt and whether it was truncated
"""

from __future__ import annotations

import logging
import os
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .comfy import DEFAULT_NEGATIVE, ImageGenerator
from .functions import RoleConfig, judge_batch, phi_rea, phi_sco, round_scores
from .io_utils import read_json, slugify, stable_seed, write_json
from .judge import Judge, diagnose, feedback_text
from .llm import LLMClient

log = logging.getLogger(__name__)


def progress(it: Iterable, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(list(it), desc=desc)
    except ImportError:
        return it


@dataclass
class VSPRSettings:
    max_iterations: int = 3  # M  (M + 1 rounds)
    batch_size: int = 4  # B
    use_vqa: bool = True
    dms_weight: float = 0.5  # a in s = a * DMS + (1 - a) * OPS
    base_seed: int = 42
    negative: str = DEFAULT_NEGATIVE  # fixed; VSPR does not write negative prompts


class VSPRPipeline:
    def __init__(
        self,
        llm: LLMClient,
        generator: ImageGenerator,
        reasoner: RoleConfig,
        judge: Judge | None,
        settings: VSPRSettings,
        run_dir: str | Path,
    ) -> None:
        if settings.use_vqa and judge is None:
            raise ValueError("VSPR with VQA needs a judge (MLLM)")
        self.llm = llm
        self.gen = generator
        self.reasoner = reasoner
        self.judge = judge
        self.s = settings
        self.run_dir = Path(run_dir)
        self.errors: list[dict] = []

    # ------------------------------------------------------------- layout
    def item_dir(self, idx: int, description: str) -> Path:
        return self.run_dir / "items" / f"{idx:04d}_{slugify(description, 80)}"

    def round_dir(self, idx: int, description: str, i: int) -> Path:
        return self.item_dir(idx, description) / f"round_{i}"

    def prompt_for_round(self, idx: int, description: str, i: int) -> str | None:
        if i == 1:
            return description  # P_1 = D
        prev = self.round_dir(idx, description, i - 1) / "reasoning.json"
        return read_json(prev)["next_prompt"] if prev.exists() else None

    # ------------------------------------------------------------- helpers
    def _run_stage(self, name: str, items: list[tuple[int, str]], fn: Callable[[int, str], None],
                   before: Callable[[], None] | None = None) -> None:
        if not items:
            return
        if before:
            before()
        for idx, desc in progress(items, name):
            try:
                fn(idx, desc)
            except Exception as err:  # keep going; the item is retried on the next run
                log.error("[%s] item %d (%s) failed: %s", name, idx, desc, err)
                self.errors.append({"stage": name, "index": idx, "description": desc,
                                    "error": repr(err), "trace": traceback.format_exc(limit=3)})

    def _release_models(self) -> None:
        self.llm.unload(self.reasoner.model)
        if self.judge is not None:
            self.judge.unload()

    def _release_generator(self) -> None:
        self.gen.release()

    # ------------------------------------------------------------- stages
    def stage_generate(self, items: list[tuple[int, str]], i: int) -> None:
        todo = []
        for idx, desc in items:
            if (self.round_dir(idx, desc, i) / "images.json").exists():
                continue
            if self.prompt_for_round(idx, desc, i) is not None:
                todo.append((idx, desc))

        def fn(idx: int, desc: str) -> None:
            prompt = self.prompt_for_round(idx, desc, i)
            seed = stable_seed(self.s.base_seed, desc)
            images = self.gen.generate(prompt, seed, negative=self.s.negative)
            if len(images) != self.s.batch_size:
                raise RuntimeError(f"expected {self.s.batch_size} images, got {len(images)}")
            write_json(self.round_dir(idx, desc, i) / "images.json",
                       {"prompt": prompt, "seed": seed, "negative": self.s.negative, "images": images})

        self._run_stage(f"G round {i}", todo, fn, self._release_models)

    def stage_judge(self, items: list[tuple[int, str]], i: int) -> None:
        """phi_que / phi_ans on every image, then phi_sco of the round (no model call)."""
        todo = [(idx, d) for idx, d in items
                if (self.round_dir(idx, d, i) / "images.json").exists()
                and not (self.round_dir(idx, d, i) / "score.json").exists()]

        def fn(idx: int, desc: str) -> None:
            rd = self.round_dir(idx, desc, i)
            if (rd / "judge.json").exists():
                results = read_json(rd / "judge.json")["results"]
            else:
                images = read_json(rd / "images.json")["images"]
                results = judge_batch(self.judge, desc, images)
                write_json(rd / "judge.json", {"model": self.judge.model, "results": results,
                                               "diagnose": [diagnose(desc, r) for r in results]})
            sc = round_scores(desc, results)
            score = phi_sco(sc["dms"], sc["ops"], self.s.dms_weight)
            write_json(rd / "score.json", {"score": score, "dms": sc["dms"], "ops": sc["ops"],
                                           "dms_weight": self.s.dms_weight, "per_image": sc["per_image"]})

        self._run_stage(f"phi_ans round {i}", todo, fn, self._release_generator)

    def stage_reason(self, items: list[tuple[int, str]], i: int) -> None:
        need = "judge.json" if self.s.use_vqa else "images.json"
        todo = [(idx, d) for idx, d in items
                if (self.round_dir(idx, d, i) / need).exists()
                and not (self.round_dir(idx, d, i) / "reasoning.json").exists()]

        def fn(idx: int, desc: str) -> None:
            rd = self.round_dir(idx, desc, i)
            prompt = read_json(rd / "images.json")["prompt"]
            feedback = (feedback_text(desc, read_json(rd / "judge.json")["results"])
                        if self.s.use_vqa else None)
            out = phi_rea(self.llm, self.reasoner, desc, prompt, feedback, self.s.batch_size,
                          use_vqa=self.s.use_vqa)
            write_json(rd / "reasoning.json", {"prompt": prompt, "feedback": feedback, **out})

        self._run_stage(f"phi_rea round {i}", todo, fn, self._release_generator)

    # ------------------------------------------------------------- driver
    def run(self, descriptions: list[str]) -> dict:
        items = list(enumerate(descriptions))
        models = [self.reasoner.model] + ([self.judge.model] if self.s.use_vqa else [])
        describe = getattr(self.llm, "describe", None)
        if describe:  # model digests, appended once per invocation (resumed runs included)
            prov_path = self.run_dir / "provenance.json"
            prov = read_json(prov_path) if prov_path.exists() else []
            prov.append(describe(sorted(set(models))))
            write_json(prov_path, prov)
        write_json(self.run_dir / "settings.json", {
            **self.s.__dict__,
            "reasoner": self.reasoner.__dict__,
            "mllm": self.judge.model if self.s.use_vqa else None,
            "generator": getattr(self.gen, "name", "?"),
            "num_descriptions": len(descriptions),
        })
        M = self.s.max_iterations
        for i in range(1, M + 2):  # M + 1 rounds
            self.stage_generate(items, i)
            if self.s.use_vqa:
                self.stage_judge(items, i)
            if i <= M:
                self.stage_reason(items, i)
        if self.errors:
            write_json(self.run_dir / "errors.json", self.errors)
        return summarize(self.run_dir, descriptions, M + 1, self.s.use_vqa)


# ----------------------------------------------------------------- selection
def load_rounds(run_dir: Path, idx: int, description: str, max_round: int,
                use_vqa: bool = True) -> list[dict]:
    """Finished rounds 1..max_round (a VSPR round is finished once it is scored)."""
    base = Path(run_dir) / "items" / f"{idx:04d}_{slugify(description, 80)}"
    rounds = []
    for i in range(1, max_round + 1):
        rd = base / f"round_{i}"
        done = rd / ("score.json" if use_vqa else "images.json")
        if not done.exists():
            break
        r = {"round": i, **read_json(rd / "images.json")}
        r["score"] = read_json(rd / "score.json") if (rd / "score.json").exists() else None
        rounds.append(r)
    return rounds


def select_best(rounds: list[dict]) -> dict:
    """argmax_i s_i over the given rounds; ties -> the earlier round."""
    return max(rounds, key=lambda r: (r["score"]["score"], -r["round"]))


def summarize(run_dir: Path, descriptions: list[str], max_round: int, use_vqa: bool = True) -> dict:
    done, complete, chosen = 0, 0, {}
    for idx, desc in enumerate(descriptions):
        rounds = load_rounds(run_dir, idx, desc, max_round, use_vqa)
        if rounds:
            done += 1
            complete += len(rounds) == max_round
            r = (select_best(rounds) if use_vqa else rounds[-1])["round"]
            chosen[r] = chosen.get(r, 0) + 1
    summary = {"descriptions": len(descriptions), "with_results": done,
               "all_rounds_complete": complete, "selected_round_histogram": chosen,
               "selection": "best phi_sco" if use_vqa else "last round"}
    write_json(Path(run_dir) / "summary.json", summary)
    return summary


def link_or_copy(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_samples(
    run_dir: str | Path,
    descriptions: list[str],
    out_dir: str | Path,
    upto_round: int,
    batch_size: int,
    select: str = "best",
) -> dict:
    """Write images in T2I-CompBench layout: ``<out>/samples/<D>_<id:06d>.png``.

    ``upto_round = k + 1`` with ``select="best"`` reproduces "VSPR iter k" (best of rounds
    1..k+1); ``select="last"`` takes round ``upto_round`` without selection (w/o VQA, and the
    per-round exports used by validate_scorer.py).
    """
    out_dir = Path(out_dir)
    if (out_dir / "samples").exists():
        shutil.rmtree(out_dir / "samples")
    manifest, missing = [], []
    for idx, desc in enumerate(descriptions):
        rounds = load_rounds(Path(run_dir), idx, desc, upto_round, use_vqa=(select == "best"))
        if len(rounds) < upto_round:
            missing.append(desc)
            if not rounds:
                continue
        chosen = select_best(rounds) if select == "best" else rounds[-1]
        sc = chosen["score"] or {}
        for b, img in enumerate(chosen["images"]):
            qid = idx * batch_size + b
            link_or_copy(img, out_dir / "samples" / f"{desc}_{qid:06d}.png")
            manifest.append({"question_id": qid, "description": desc, "prompt": chosen["prompt"],
                             "round": chosen["round"], "score": sc.get("score"),
                             "dms": sc.get("dms"), "ops": sc.get("ops")})
    write_json(out_dir / "manifest.json", manifest)
    if missing:
        write_json(out_dir / "incomplete.json", missing)
    return {"images": len(manifest), "incomplete": len(missing)}
