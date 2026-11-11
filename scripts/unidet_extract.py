"""Save UniDet detections and DPT depth maps for every image in a samples folder.

Runs INSIDE a T2I-CompBench clone, in the environment that has detectron2 and the expert
weights (see the T2I-CompBench README), with UniDet_eval as the working directory:

    cd T2I-CompBench/UniDet_eval
    python /path/to/VSPR/scripts/unidet_extract.py --samples /path/to/<method>/samples \
        --out /path/to/<method>/unidet

Writes <out>/det/<image>.json  {"width","height","boxes" (x0,y0,x1,y1 in image pixels),
"labels","label_ids","scores"} and <out>/depth/<image>.png (min-max normalized inverse depth,
at the image's own resolution). Models and settings are the ones of T2I-CompBench's
3D_spatial_eval.py (UniDet R50 unified label space, score threshold 0.5; DPT hybrid, 480x480).
Scoring is done afterwards by vspr/eval/spatial.py. Existing outputs are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    sys.argv = sys.argv[:1]  # load_expert_model() parses sys.argv itself
    sys.path.insert(0, os.getcwd())
    if not Path("experts/model_bank_3d.py").exists():
        sys.exit("Run this from T2I-CompBench/UniDet_eval")

    import numpy as np
    import torch
    from PIL import Image

    from experts.model_bank_3d import load_expert_model

    samples = sorted(p for p in Path(args.samples).iterdir() if p.suffix.lower() == ".png")
    out = Path(args.out)
    (out / "det").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.autocast("cuda", dtype=torch.float16) if dev == "cuda" else torch.autocast("cpu", enabled=False)

    todo = [p for p in samples if not (out / "depth" / p.name).exists()]
    if todo:
        model, tf = load_expert_model(task="depth")
        model = model.eval().to(dev)
        with torch.no_grad(), amp:
            for s in range(0, len(todo), args.batch_size):
                chunk = todo[s : s + args.batch_size]
                imgs = [Image.open(p).convert("RGB") for p in chunk]
                pred = model(torch.stack([tf(im) for im in imgs]).to(dev)).float()
                for p, im, d in zip(chunk, imgs, pred):
                    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                    d = torch.nn.functional.interpolate(d[None, None], size=(im.height, im.width),
                                                        mode="bilinear", align_corners=True)[0, 0]
                    Image.fromarray((255 * d.cpu().numpy()).astype("uint8"), "L").save(out / "depth" / p.name)
        del model
        torch.cuda.empty_cache()

    todo = [p for p in samples if not (out / "det" / (p.stem + ".json")).exists()]
    if todo:
        names = torch.load("dataset/detection_features.pt")["labels"]
        model, tf = load_expert_model(task="obj_detection")
        model = model.eval().to(dev)
        with torch.no_grad(), amp:
            for s in range(0, len(todo), args.batch_size):
                chunk = todo[s : s + args.batch_size]
                inputs, sizes = [], []
                for p in chunk:
                    im = Image.open(p).convert("RGB")
                    arr = np.array(tf(im))[:, :, ::-1]  # BGR, as in T2I-CompBench
                    inputs.append({"image": torch.as_tensor(arr.astype("float32").transpose(2, 0, 1)),
                                   "height": im.height, "width": im.width})  # boxes come back in image pixels
                    sizes.append((im.width, im.height))
                for p, (w, h), o in zip(chunk, sizes, model(inputs)):
                    inst = o["instances"].to("cpu")
                    ids = inst.pred_classes.tolist()
                    rec = {"width": w, "height": h, "boxes": inst.pred_boxes.tensor.float().tolist(),
                           "label_ids": ids, "labels": [names[i] for i in ids],
                           "scores": inst.scores.float().tolist()}
                    (out / "det" / (p.stem + ".json")).write_text(json.dumps(rec), encoding="utf-8")
    print(f"done: {len(samples)} images -> {out}")


if __name__ == "__main__":
    main()
