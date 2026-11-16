"""TIFA (Hu et al., ICCV 2023): text-to-image faithfulness by question answering.

    TIFA(I, C) = fraction of the questions generated from C that the VQA model answers correctly.

Pipeline, following the official ``tifascore`` package (github.com/Yushi-Hu/tifa, Apache-2.0):

1. Question generation, once per description (``scripts/tifa_questions.py``):
   the open TIFA question generator ``tifa-benchmark/llama2_tifa_question_generation`` with the
   official prompt and decoding (beam search 5), parsed with the official parser, then filtered
   with UnifiedQA (``allenai/unifiedqa-v2-t5-large-1363200``): a question is kept only if
   UnifiedQA, reading the caption, gives the same answer. The prompt, parser and filter below
   are copied from ``tifascore`` (question_gen_llama2.py, question_filter.py, unifiedqa.py).
   The questions are stored in ``data/tifa/3d_spatial_qas.json`` and shared by all methods.
2. Answering (``TIFAScore``): BLIP-2 Flan-T5-XL (``tifascore`` model name ``blip2-flant5xl``),
   multiple-choice prompt "Answer the multiple-choice question. Question: {q} Choices: {c}
   Answer:" as in ``tifascore``; the free-form answer is mapped to the closest choice with
   SBERT (all-mpnet-base-v2) when it is not one of the choices. ``tifascore`` loads BLIP-2
   through LAVIS (``blip2_t5 / pretrain_flant5xl``); here the same checkpoint is loaded from
   Hugging Face (``Salesforce/blip2-flan-t5-xl``) with the LAVIS default decoding (beam 5,
   max 30 tokens), so no LAVIS / fairseq / modelscope install is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

CATEGORIES = ['object', 'human', 'animal', 'food', 'activity', 'attribute',
              'counting', 'color', 'material', 'spatial', 'location', 'shape', 'other']
QG_MODEL = "tifa-benchmark/llama2_tifa_question_generation"
QA_MODEL = "allenai/unifiedqa-v2-t5-large-1363200"
VQA_MODEL = "Salesforce/blip2-flan-t5-xl"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"
FREE_FORM_THRESHOLD = 0.6
QAS_PATH = Path(__file__).resolve().parents[2] / "data" / "tifa" / "3d_spatial_qas.json"


# ---------------------------------------------------------------- question generation (tifascore)
def create_qg_prompt(caption: str) -> str:
    INTRO_BLURB = """Given an image description, generate one or two multiple-choice questions that verifies if the image description is correct.
Classify each concept into a type (object, human, animal, food, activity, attribute, counting, color, material, spatial, location, shape, other), and then generate a question for each type.
"""
    formated_prompt = f"<s>[INST] <<SYS>>\n{INTRO_BLURB}\n<</SYS>>\n\n"
    formated_prompt += f"Description: {caption} [/INST] Entities:"
    return formated_prompt


def parse_resp(resp: str):
    resp = resp.split('\n')
    question_instances = []
    this_entity = this_type = this_question = this_choices = this_answer = None
    for line_number in range(6, len(resp)):
        line = resp[line_number]
        if line.startswith('About '):
            whole_line = line[len('About '):-1]
            this_entity = whole_line.split(' (')[0]
            this_type = whole_line.split(' (')[1].split(')')[0]
        elif line.startswith('Q: '):
            this_question = line[3:]
        elif line.startswith('Choices: '):
            this_choices = line[9:].split(', ')
        elif line.startswith('A: '):
            this_answer = line[3:]
            if this_entity and this_question and this_choices:
                question_instances.append((this_entity, this_question, this_choices, this_answer, this_type))
            this_question = this_choices = this_answer = None
    return question_instances


def to_qas(caption: str, question_instances) -> list[dict]:
    out = []
    for entity, question, choices, answer, etype in question_instances:
        if etype not in CATEGORIES:
            continue
        out.append({"caption": caption, "element": entity, "question": question, "choices": choices,
                    "answer": answer, "element_type": "animal/human" if etype in ("animal", "human") else etype})
    return out


def compute_prf(gold, pred) -> float:
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
        recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
    else:
        F1 = 1 if len(pred) == 0 else 0
    return float(F1)


def filter_question_and_answers(qa_model, caption_qas: list[dict]) -> list[dict]:
    filtered, seen = [], set()
    for q in caption_qas:
        if q['question'] in seen:  # avoid duplicate questions
            continue
        seen.add(q['question'])
        if qa_model.mcqa(q['question'], q['caption'], choices=q['choices']) != q['answer']:
            continue
        if q['answer'] not in ['yes', 'no']:  # validate the free-form answer
            free_form_answer = qa_model.qa(q['question'], q['caption']).strip()
            if q['answer'].isnumeric():
                try:
                    from word2number import w2n

                    free_form_answer = str(w2n.word_to_num(free_form_answer))
                except Exception:  # noqa: BLE001
                    pass
            if compute_prf(q['answer'].split(), free_form_answer.split()) <= FREE_FORM_THRESHOLD:
                continue
        filtered.append(q)
    return filtered


class UnifiedQAModel:
    def __init__(self, model_name: str = QA_MODEL):
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.torch = torch
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def run_model(self, input_string, **generator_args):
        with self.torch.no_grad():
            input_ids = self.tokenizer.encode(input_string, return_tensors="pt")
            res = self.model.generate(input_ids.to(self.model.device), max_new_tokens=30, **generator_args)
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def qa(self, question, context):
        answer = self.run_model(f"{question} \n {context}")[0]
        return ''.join(c for c in answer if c.isalnum() or c.isspace()).strip().lower()

    def mcqa(self, question, context, choices=("yes", "no")):
        choice_text = ""
        headings = ["(A)", "(B)", "(C)", "(D)"]
        for i, choice in enumerate(choices):
            if i < len(headings):
                choice_text += f"{headings[i]} {choice} "
        return self.run_model(f"{question} \n {context} \n {choice_text}")[0]


def load_qas(path: str | Path = QAS_PATH) -> dict[str, list[dict]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: generate the TIFA questions first (scripts/tifa_questions.py)")
    return json.loads(p.read_text(encoding="utf-8"))["questions"]


# ---------------------------------------------------------------- answering
class SBERTModel:
    def __init__(self, ckpt: str = SBERT_MODEL, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt).eval().to(device)

    def embed(self, sentences):
        import torch
        import torch.nn.functional as F

        enc = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            tok = self.model(**enc)[0]
        mask = enc["attention_mask"].unsqueeze(-1).expand(tok.size()).float()
        emb = (tok * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=1).cpu()

    def multiple_choice(self, answer, choices):
        a, c = self.embed([answer]), self.embed(list(choices))
        return choices[int((c @ a.T).argmax())]


def mc_prompt(question: str, choices) -> str:
    return f"Answer the multiple-choice question. Question: {question} Choices: {', '.join(choices)} Answer:"


class TIFAScore:
    """Per-image TIFA (fraction correct) and its accuracy on the spatial questions only."""

    keys = ("TIFA", "TIFA_spatial")

    def __init__(self, qas_path: str | Path = QAS_PATH, model_id: str = VQA_MODEL, device: str | None = None,
                 num_beams: int = 5, max_new_tokens: int = 30):
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        self.torch = torch
        self.qas = load_qas(qas_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.proc = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=self.dtype).eval().to(self.device)
        self.sbert = SBERTModel(device=self.device)
        self.gen = dict(num_beams=num_beams, max_new_tokens=max_new_tokens, min_new_tokens=1)

    def answer(self, image, qs: list[dict]) -> list[str]:
        prompts = [mc_prompt(q["question"], q["choices"]) for q in qs]
        inp = self.proc(images=[image] * len(qs), text=prompts, return_tensors="pt", padding=True)
        inp = {k: (v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)) for k, v in inp.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inp, **self.gen)
        return [t.strip() for t in self.proc.batch_decode(out, skip_special_tokens=True)]

    def __call__(self, images, captions) -> dict[str, list]:
        tifa, spatial = [], []
        for img, cap in zip(images, captions):
            qs = self.qas.get(cap) or []
            if not qs:
                tifa.append(None)
                spatial.append(None)
                continue
            ok = []
            for q, free in zip(qs, self.answer(img, qs)):
                mc = free if free in q["choices"] else self.sbert.multiple_choice(free, q["choices"])
                ok.append(int(mc == q["answer"]))
            tifa.append(sum(ok) / len(ok))
            sp = [o for o, q in zip(ok, qs) if q["element_type"] == "spatial"]
            spatial.append(sum(sp) / len(sp) if sp else None)
        return {"TIFA": tifa, "TIFA_spatial": spatial}
