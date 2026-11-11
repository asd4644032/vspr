"""Generate prompts and images for the baseline methods.

Examples
--------
    python scripts/run_baselines.py --methods none beautiful magic promptist
    python scripts/run_baselines.py --methods gpt4            # needs OPENAI_API_KEY
    python scripts/run_baselines.py --methods sspo=path/to/sspo_prompts.json pag=path/to/pag.json
    python scripts/run_baselines.py --methods none --set t2i.backend=cascade

Every method uses the same per-description seed as VSPR, so "none" reproduces the
initial round of VSPR (served from the image cache when VSPR ran first).
Outputs: outputs/baselines/<method>/prompts.json and
         outputs/eval/<backend>/<method>/samples (T2I-CompBench layout).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vspr import factory  # noqa: E402
from vspr.baselines import build_rewriter  # noqa: E402
from vspr.config import load_config, resolve  # noqa: E402
from vspr.io_utils import read_json, stable_seed, write_json  # noqa: E402
from vspr.pipeline import link_or_copy, progress  # noqa: E402

log = logging.getLogger("baselines")


def make_prompts(spec: str, cfg: dict, descs: list[str]) -> tuple[str, dict]:
    name = spec.split("=", 1)[0]
    path = resolve(cfg["output_root"]) / "baselines" / name / "prompts.json"
    prompts = read_json(path) if path.exists() else {}
    todo = [d for d in descs if d not in prompts]
    if todo:
        rewriter = build_rewriter(spec, cfg)
        try:
            for i, d in enumerate(progress(todo, f"prompts {name}")):
                prompts[d] = rewriter(d)
                if i % 25 == 0:
                    write_json(path, prompts)
        finally:
            write_json(path, prompts)
            rewriter.close()
    return name, prompts


def generate(name: str, prompts: dict, cfg: dict, descs: list[str]) -> None:
    gen = factory.build_generator(cfg)
    B = int(cfg["t2i"]["batch_size"])
    out = resolve(cfg["output_root"]) / "eval" / cfg["t2i"]["backend"] / name
    manifest = []
    for idx, d in enumerate(progress(descs, f"G {name}")):
        try:
            images = gen.generate(prompts[d], stable_seed(int(cfg["t2i"]["base_seed"]), d))
        except Exception as err:  # noqa: BLE001
            log.error("generation failed for %r: %s (re-run to retry)", d, err)
            continue
        for b, img in enumerate(images):
            qid = idx * B + b
            link_or_copy(img, out / "samples" / f"{d}_{qid:06d}.png")
            manifest.append({"question_id": qid, "description": d, "prompt": prompts[d]})
    write_json(out / "manifest.json", manifest)
    print(f"{name}: {len(manifest)} images -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="none gpt4 beautiful magic promptist, or name=path for external prompts")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--prompts-only", action="store_true", help="do not generate images")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.set)
    descs = factory.descriptions(cfg)
    produced = [make_prompts(spec, cfg, descs) for spec in args.methods]  # all text models first
    if not args.prompts_only:
        for name, prompts in produced:
            generate(name, prompts, cfg, descs)


if __name__ == "__main__":
    main()
