"""Generate and filter the TIFA questions for the dataset descriptions (once; shared by all methods).

    python scripts/tifa_questions.py                       # fp16 (needs ~14 GB GPU memory, e.g. Colab L4)
    python scripts/tifa_questions.py --load-in-8bit        # ~8 GB (RTX 3080; needs bitsandbytes)

Writes data/tifa/3d_spatial_qas.json: {"meta": {...}, "questions": {description: [qa, ...]}}.
Resumable: descriptions already in the file are skipped. See vspr/eval/tifa.py for the method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vspr import factory  # noqa: E402
from vspr.config import load_config  # noqa: E402
from vspr.eval import tifa as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(T.QAS_PATH))
    ap.add_argument("--load-in-8bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch
    import transformers
    from tqdm import tqdm

    descs = factory.descriptions(load_config(None, []))[: args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"questions": {}, "raw": {}}
    data["meta"] = {"question_generator": T.QG_MODEL, "qa_filter": T.QA_MODEL, "decoding": "beam 5, max_length 512",
                    "load_in_8bit": args.load_in_8bit, "source": "tifascore (github.com/Yushi-Hu/tifa)"}
    todo = [d for d in descs if d not in data["questions"]]
    print(f"{len(descs)} descriptions, {len(todo)} to generate")
    if not todo:
        return

    qa_model = T.UnifiedQAModel()  # loaded first: small, and fails fast if a tokenizer dependency is missing
    kw = {"device_map": "auto"}
    if args.load_in_8bit:
        kw["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
    else:
        kw["torch_dtype"] = torch.float16
    pipe = transformers.pipeline("text-generation", model=T.QG_MODEL, model_kwargs=kw)
    for i, d in enumerate(tqdm(todo)):
        prompt = T.create_qg_prompt(d)
        seq = pipe(prompt, do_sample=False, num_beams=5, num_return_sequences=1, max_length=512)
        resp = seq[0]["generated_text"][len(prompt):].split("\n\n")[0]
        qas = T.to_qas(d, T.parse_resp(resp))
        data["raw"][d] = resp
        data["questions"][d] = T.filter_question_and_answers(qa_model, qas)
        if i % 20 == 0 or i == len(todo) - 1:
            out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    n = [len(v) for v in data["questions"].values()]
    print(f"done: {len(n)} descriptions, {sum(n)} questions (mean {sum(n) / len(n):.2f}); "
          f"{sum(x == 0 for x in n)} descriptions without questions -> {out}")


if __name__ == "__main__":
    main()
