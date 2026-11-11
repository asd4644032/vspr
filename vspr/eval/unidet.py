"""Spatial metric drivers.

``run_spatial`` (default): run scripts/unidet_extract.py in the UniDet environment, then
score with vspr/eval/spatial.py (fixed object matching and depth indexing).

``run_official``: T2I-CompBench's own UniDet_eval/3D_spatial_eval.py, kept for comparison.
Known issues of that script: substring object matching, depth map indexed as [x, y],
partial credit when only one object is found, and paths split on "/" (fails on Windows).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..io_utils import read_json
from .spatial import score_image

EXTRACT = Path(__file__).resolve().parents[2] / "scripts" / "unidet_extract.py"


REPO_ROOT = Path(__file__).resolve().parents[2]


def _unidet_dir(path: Path | None) -> Path:
    """Folder that contains ``experts/model_bank_3d.py`` (the UniDet + DPT code and weights).

    Accepted: the repo itself (``VSPR/experts`` copied from T2I-CompBench, the default when
    ``path`` is None), a T2I-CompBench clone, or its ``UniDet_eval`` folder.
    """
    candidates = [REPO_ROOT] if path is None else [Path(path).resolve(), Path(path).resolve() / "UniDet_eval"]
    for d in candidates:
        if (d / "experts" / "model_bank_3d.py").exists():
            return d
    raise FileNotFoundError(
        "UniDet experts not found. Copy T2I-CompBench/UniDet_eval/experts (code + expert_weights) into "
        f"{REPO_ROOT / 'experts'}, or pass --unidet-dir <T2I-CompBench clone>")


def run_spatial(method_dir: Path, t2i_compbench: Path | None, python: str = "python", batch_size: int = 8,
                unscorable: str = "exclude") -> dict[int, float | None]:
    from PIL import Image

    method_dir = Path(method_dir).resolve()
    extract_dir = method_dir / "unidet"
    subprocess.run([python, str(EXTRACT), "--samples", str(method_dir / "samples"),
                    "--out", str(extract_dir), "--batch-size", str(batch_size)],
                   cwd=_unidet_dir(t2i_compbench), check=True)
    scores: dict[int, float | None] = {}
    for row in read_json(method_dir / "manifest.json"):
        stem = f"{row['description']}_{row['question_id']:06d}"
        det = read_json(extract_dir / "det" / f"{stem}.json")
        depth = np.asarray(Image.open(extract_dir / "depth" / f"{stem}.png").convert("L"), dtype=np.float32) / 255.0
        scores[row["question_id"]] = score_image(row["description"], det, depth, unscorable=unscorable)
    return scores


def run_official(method_dir: Path, t2i_compbench: Path, python: str = "python",
                 batch_size: int = 8, fresh: bool = True) -> dict[int, float]:
    method_dir = Path(method_dir).resolve()
    labels = method_dir / "labels"
    if fresh and labels.exists():
        shutil.rmtree(labels)  # the script silently reuses stale depth maps otherwise
    d = _unidet_dir(t2i_compbench)
    if not (d / "3D_spatial_eval.py").exists():
        raise FileNotFoundError(f"{d} has no 3D_spatial_eval.py; the official script needs a T2I-CompBench clone")
    cmd = [python, "3D_spatial_eval.py", "--outpath", method_dir.as_posix().rstrip("/"),
           "--batch_size", str(batch_size)]
    subprocess.run(cmd, cwd=d, check=True)
    with open(labels / "annotation_obj_detection_3d" / "vqa_result.json", "r", encoding="utf-8") as f:
        return {int(r["question_id"]): float(r["answer"]) for r in json.load(f)}
