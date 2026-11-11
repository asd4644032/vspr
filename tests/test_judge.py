"""vspr.judge (phi_que / phi_ans) against the experiment judge (vqa_test/vqa_v4.py).

Golden files:
* golden/judge_messages.json -- every request and probe result of the experiment judge on a
  deterministic fake Ollama client (tests/fake_ollama.py), with and without logprobs;
* golden/judge_v4.json -- real qwen2.5vl:7b probe results of 5 descriptions x 3 rounds of the
  v5-noneg trajectories (vqa_test/pilot2/sdxl/state.json) with their diagnosis, feedback text,
  DMS, OPS and 0.5*DMS + 0.5*OPS as computed by the experiment code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fake_ollama import FakeOllama
from vspr import judge as J
from vspr.functions import phi_sco, round_scores

GOLDEN = Path(__file__).resolve().parent / "golden"
MSG = json.loads((GOLDEN / "judge_messages.json").read_text(encoding="utf-8"))
V4 = json.loads((GOLDEN / "judge_v4.json").read_text(encoding="utf-8"))


def test_texts_verbatim():
    assert J.SYSTEM_V4 == MSG["system"]
    assert J.FIXES == MSG["fixes"]
    assert J.FIX_ORDER == MSG["fix_order"]


@pytest.mark.parametrize("logprobs", [True, False])
def test_probe_matches_experiment_judge(logprobs):
    client = FakeOllama(logprobs=logprobs)
    judge = J.Judge("qwen2.5vl:7b", client=client)
    cases = [c for c in MSG["cases"] if c["logprobs"] == logprobs]
    assert cases
    for c in cases:
        n0 = len(client.calls)
        r = judge.probe(c["description"], c["image"], c["b"])
        assert r.pop("seconds") >= 0
        assert r == c["result"], c["description"]
        assert client.calls[n0:] == c["calls"], c["description"]  # same questions, options, schema
        assert J.diagnose(c["description"], r) == c["diagnose"]


def test_question_count():
    client = FakeOllama()
    judge = J.Judge("m", client=client)
    judge.probe("a cat in front of a dog", "x.png", 1)
    assert len(client.calls) == 8
    judge.probe("a candle hidden by a candle", "x.png", 1)
    assert len(client.calls) == 8 + 4


def test_logprob_parsing():
    resp = {"message": {"content": "Yes"}, "logprobs": [
        {"token": "Yes", "logprob": -0.1, "top_logprobs": [
            {"token": "Yes", "logprob": -0.5}, {"token": " yes", "logprob": -2.0},
            {"token": "No", "logprob": -1.5}, {"token": "Maybe", "logprob": -0.1}]}]}
    d = J.Judge._first_token_dist(resp, ("yes", "no"))
    import math
    y = math.exp(-0.5) + math.exp(-2.0)
    assert d["yes"] == pytest.approx(y / (y + math.exp(-1.5)))
    assert J.Judge._first_token_dist({"message": {"content": "x"}}, ("yes", "no")) is None
    assert J.Judge.yn('{"answer": "No"}') == "no" and J.Judge.yn("maybe") == "other"


def test_thinking_model_detected():
    judge = J.Judge("m", client=FakeOllama(capabilities=["vision", "thinking"]))
    assert judge.thinking


def test_diagnosis_and_feedback_of_real_rounds():
    for rnd in V4["rounds"]:
        d, res = rnd["description"], rnd["judge"]
        assert [J.diagnose(d, r) for r in res] == rnd["diagnose"]
        assert J.feedback_text(d, res) == rnd["feedback"]
        sc = round_scores(d, res)
        assert abs(sc["dms"] - rnd["dms"]) < 1e-12 and abs(sc["ops"] - rnd["ops"]) < 1e-12
        assert abs(phi_sco(sc["dms"], sc["ops"], 0.5) - rnd["score05"]) < 1e-12


def test_selection_of_real_trajectories():
    # best05 of the experiments: max over rounds of 0.5*DMS + 0.5*OPS, ties -> earlier round
    from vspr.pipeline import select_best

    for d in V4["descriptions"]:
        rs = [r for r in V4["rounds"] if r["description"] == d]
        expect = max(rs, key=lambda r: (r["score05"], -r["round"]))["round"]
        rounds = [{"round": r["round"], "score": {"score": phi_sco(*map(round_scores(d, r["judge"]).get,
                                                                           ("dms", "ops")), 0.5)}}
                  for r in rs]
        assert select_best(rounds)["round"] == expect


def test_same_object_presence():
    r = {"count_a": {"0": 0.1, "1": 0.3, "2": 0.4, "3": 0.2}, "same_rel_p": 0.9, "overall_p": 0.7}
    g = J.diagnose("a candle hidden by a candle", r)
    assert g["presence"] == pytest.approx(0.6 * 0.6) and g["fails"] == []
    assert J.image_scores("a candle hidden by a candle", r) == {"dms": 0.7, "ops": pytest.approx(0.36)}
