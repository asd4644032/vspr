"""Run VSPR (Algorithm 1) on a description file and export the results for evaluation.

Examples
--------
    # VSPR with Llama 3.1 on SDXL (paper setting M=3, B=4, a=0.5; MLLM qwen2.5vl:7b)
    python scripts/run_vspr.py

    # DeepSeek-R1 on Stable Cascade
    python scripts/run_vspr.py --set llm.reasoner=deepseek-r1:14b t2i.backend=cascade

    # "w/o VQA" ablation
    python scripts/run_vspr.py --set vspr.use_vqa=false

    # Only re-export already computed results
    python scripts/run_vspr.py --export-only

Re-running the same command resumes from the last finished step.
Exports go to outputs/eval/<backend>/<method>/samples (T2I-CompBench layout):
    <method>          best prompt over all M+1 rounds by phi_sco (P*, I*)
    <method>_iter<k>  best over rounds 1..k+1, i.e. "VSPR iter k"; iter1 = "w/o iterative"
With vspr.use_vqa=false (w/o VQA) there is no phi_sco: <method> is the last round M+1 and
<method>_iter<k> is round k+1.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vspr import factory  # noqa: E402
from vspr.config import load_config, resolve  # noqa: E402
from vspr.pipeline import export_samples, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--export-only", action="store_true", help="skip generation, only export")
    ap.add_argument("--export-rounds", action="store_true",
                    help="also export every round on its own as <method>_round<i> (for validate_scorer.py)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config, args.set)
    descs = factory.descriptions(cfg)
    rdir = factory.run_dir(cfg)
    M = int(cfg["vspr"]["max_iterations"])
    B = int(cfg["t2i"]["batch_size"])
    print(f"Run directory: {rdir}\nDescriptions: {len(descs)} | M={M} B={B} "
          f"use_vqa={cfg['vspr']['use_vqa']} a={cfg['vspr']['dms_weight']}")

    use_vqa = bool(cfg["vspr"]["use_vqa"])
    if args.export_only:
        summary = summarize(rdir, descs, M + 1, use_vqa)
    else:
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = factory.build_pipeline(cfg).run(descs)
    print("Summary:", json.dumps(summary, indent=2))

    name, backend = factory.run_name(cfg), cfg["t2i"]["backend"]
    method = name[len(backend) + 1:] if name.startswith(backend + "_") else name
    eval_root = resolve(cfg["output_root"]) / "eval" / cfg["t2i"]["backend"]
    exports = {f"vspr_{method}": M + 1}
    exports.update({f"vspr_{method}_iter{k}": k + 1 for k in range(1, M)})
    for name, upto in exports.items():
        res = export_samples(rdir, descs, eval_root / name, upto, B, select="best" if use_vqa else "last")
        print(f"exported {name}: {res}")
    if args.export_rounds:
        for i in range(1, M + 2):
            name = f"vspr_{method}_round{i}"
            res = export_samples(rdir, descs, eval_root / name, i, B, select="last")
            print(f"exported {name}: {res}")
    if (rdir / "errors.json").exists():
        print(f"Some items failed; see {rdir / 'errors.json'} and re-run to retry.")


if __name__ == "__main__":
    main()
