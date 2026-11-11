"""Evaluate exported images with the metrics of the paper.

    python scripts/evaluate.py --eval-root outputs/eval/sdxl
    python scripts/evaluate.py --eval-root outputs/eval/sdxl --metrics CS ITC ITM VQA AS
    python scripts/evaluate.py --eval-root outputs/eval/sdxl --metrics Spatial \
        --unidet-python /path/to/unidet-env/python      # uses VSPR/experts if present
    (or --unidet-dir ../T2I-CompBench to use the experts of a T2I-CompBench clone)

Each method folder (outputs/eval/<backend>/<method>/) must contain samples/ and
manifest.json, as written by run_vspr.py / run_baselines.py. Per-image scores go to
<method>/scores/<metric>.json and the scaled means to <method>/metrics.json.
Then build the tables with scripts/make_tables.py.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vspr.eval import ORDER, SCALE  # noqa: E402
from vspr.io_utils import read_json, write_json  # noqa: E402


def method_dirs(root: Path, names: list[str] | None) -> list[Path]:
    dirs = [d for d in sorted(root.iterdir()) if (d / "manifest.json").exists()]
    return [d for d in dirs if d.name in names] if names else dirs


def samples(method: Path) -> list[tuple[int, Path, str]]:
    rows = read_json(method / "manifest.json")
    return [(r["question_id"], method / "samples" / f"{r['description']}_{r['question_id']:06d}.png",
             r["description"]) for r in rows]


def score_with(scorer, method: Path, keys: tuple[str, ...], batch_size: int, force: bool) -> None:
    from PIL import Image

    out_files = {k: method / "scores" / f"{k}.json" for k in keys}
    if not force and all(f.exists() for f in out_files.values()):
        return
    rows = samples(method)
    results = {k: {} for k in keys}
    for s in range(0, len(rows), batch_size):
        chunk = rows[s : s + batch_size]
        images = [Image.open(p).convert("RGB") for _, p, _ in chunk]
        caps = [c for _, _, c in chunk]
        out = scorer(images, caps)
        if not isinstance(out, dict):
            out = {keys[0]: out}
        for k in keys:
            for (qid, _, _), v in zip(chunk, out[k]):
                results[k][qid] = float(v)
    for k in keys:
        write_json(out_files[k], results[k])
    print(f"  {method.name}: {', '.join(keys)} done ({len(rows)} images)")


def summarize(method: Path) -> dict:
    summary = {}
    for k in ORDER:
        f = method / "scores" / f"{k}.json"
        if f.exists():
            raw = list(read_json(f).values())
            vals = [v for v in raw if v is not None]  # None = not scorable (e.g. "wallet" for Spatial)
            summary[k] = SCALE[k] * statistics.fmean(vals) if vals else None
            summary[f"{k}_n"] = len(vals)
            if len(vals) != len(raw):
                summary[f"{k}_excluded"] = len(raw) - len(vals)
    write_json(method / "metrics.json", summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True, help="e.g. outputs/eval/sdxl")
    ap.add_argument("--methods", nargs="*", default=None, help="method folders (default: all)")
    ap.add_argument("--metrics", nargs="*", default=ORDER, choices=ORDER)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--unidet-dir", "--t2i-compbench", dest="unidet_dir", default=None,
                    help="folder with experts/ (UniDet + DPT): default VSPR/experts; or a T2I-CompBench clone")
    ap.add_argument("--unidet-python", default=sys.executable, help="python with detectron2 for UniDet")
    ap.add_argument("--spatial-impl", choices=["vspr", "official"], default="vspr",
                    help="vspr: fixed scorer (vspr/eval/spatial.py); official: T2I-CompBench script as is")
    ap.add_argument("--spatial-unscorable", choices=["exclude", "zero"], default="exclude",
                    help="images whose object has no UniDet class (wallet): exclude from the mean or score 0")
    ap.add_argument("--force", action="store_true", help="recompute existing scores")
    args = ap.parse_args()

    root = Path(args.eval_root)
    methods = method_dirs(root, args.methods)
    if not methods:
        sys.exit(f"No method folders with manifest.json under {root}")
    print("Methods:", ", ".join(m.name for m in methods))

    from vspr.eval import metrics as M

    jobs = []
    if "CS" in args.metrics:
        jobs.append((M.CLIPScore, ("CS",)))
    if {"ITC", "ITM"} & set(args.metrics):
        jobs.append((M.BLIP2ITCITM, ("ITC", "ITM")))
    if "VQA" in args.metrics:
        jobs.append((M.BLIPVQAScore, ("VQA",)))
    if "AS" in args.metrics:
        jobs.append((M.AestheticScore, ("AS",)))

    for cls, keys in jobs:  # load each model once, then score every method
        print(f"[{'/'.join(keys)}] loading {cls.__name__}")
        scorer = cls()
        for m in methods:
            score_with(scorer, m, keys, args.batch_size, args.force)
        del scorer
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    if "Spatial" in args.metrics:
        from vspr.eval.unidet import _unidet_dir, run_official, run_spatial

        unidet_dir = Path(args.unidet_dir) if args.unidet_dir else None
        try:
            print(f"[Spatial] UniDet experts: {_unidet_dir(unidet_dir) / 'experts'}")
        except FileNotFoundError as err:
            sys.exit(str(err))

        for m in methods:
            out = m / "scores" / "Spatial.json"
            if out.exists() and not args.force:
                continue
            print(f"[Spatial/{args.spatial_impl}] {m.name}")
            if args.spatial_impl == "official":
                res = run_official(m, unidet_dir, args.unidet_python)
            else:
                res = run_spatial(m, unidet_dir, args.unidet_python,
                                  unscorable=args.spatial_unscorable)
            write_json(out, res)

    for m in methods:
        s = summarize(m)
        print(f"{m.name:40s} " + " ".join(f"{k}={s[k]:.2f}" for k in ORDER if s.get(k) is not None))


if __name__ == "__main__":
    main()
