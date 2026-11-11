"""Round export + validate_scorer, external prompt import, provenance recording."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.test_pipeline import DESCS, FakeLLM, make
from vspr.baselines.rewriters import ExternalPrompts
from vspr.io_utils import read_json, write_json
from vspr.pipeline import export_samples

REPO = Path(__file__).resolve().parent.parent


def test_round_export_and_validate(tmp_path):
    pipe, _, _ = make(tmp_path)  # phi_sco peaks at round 3
    pipe.run(DESCS)
    root = tmp_path / "eval"
    for i in range(1, 5):
        d = root / f"vspr_x_round{i}"
        export_samples(pipe.run_dir, DESCS, d, i, 4, select="last")
        man = read_json(d / "manifest.json")
        assert {m["round"] for m in man} == {i}
        # fake metric that peaks at round 3, like the scorer
        write_json(d / "scores" / "Spatial.json", {m["question_id"]: [0.1, 0.3, 0.6, 0.2][i - 1] for m in man})
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "validate_scorer.py"),
                          "--eval-root", str(root), "--method", "vspr_x"],
                         capture_output=True, text=True, check=True).stdout
    assert "oracle best round" in out and "selected by s (phi_sco)" in out
    sel = next(l for l in out.splitlines() if "selected by s (phi_sco)" in l)
    assert "60.00" in sel  # phi_sco picked round 3
    assert "selected by DMS only" in out


def test_external_prompts_txt_dir(tmp_path):
    (tmp_path / "a cat behind a dog.txt").write_text("a cat behind a dog, 8k\n", encoding="utf-8")
    ext = ExternalPrompts(tmp_path, name="pag")
    assert ext("a cat behind a dog") == "a cat behind a dog, 8k"


def test_shipped_pag_prompts_cover_dataset():
    descs = [l.strip() for l in (REPO / "data/t2i_compbench/3d_spatial.txt").read_text().splitlines() if l.strip()]
    pag = ExternalPrompts(REPO / "data/baseline_prompts/pag.json", name="pag")
    assert all(d in pag.table for d in descs)


def test_provenance_written(tmp_path):
    class DescLLM(FakeLLM):
        def describe(self, models):
            return {m: {"digest": "abc"} for m in models}

    pipe, _, _ = make(tmp_path, llm=DescLLM(), max_iterations=1)
    pipe.run(DESCS[:1])
    pipe.run(DESCS[:1])
    prov = read_json(pipe.run_dir / "provenance.json")
    assert len(prov) == 2 and prov[0]["reasoner"]["digest"] == "abc"
