"""phi_que / phi_ans of VSPR: a fixed set of diagnostic VQA questions asked on every image.

The questions, the system prompt, the way the answer probabilities are read and the failure
diagnosis are copied verbatim from the experiment code that produced every validation result
(vqa_test/vqa_v4.py, "v4 judge"); do not edit the wording, it is part of the method.

  layer        question (per image)                                         failure types
  -----------  -----------------------------------------------------------  -------------------------------
  1 count      How many {A}s ... 0, 1, 2 or 3 (=3+)?  (and for {B})          missing_a / missing_b / crowd
               (same object twice: only {A}, 2+ needed)                      not_two
  2 integrity  Are {A} and {B} two separate objects? (p yes)                 fused
               Is one of them only a picture / print / screen image? (p yes) picture
  3 relation   in front of / behind: Which is closer: X, Y or same?          reversed / same_depth
                 (X/Y order swapped between images) + Is one on top of the
                 other? (p yes)                                              (+ stacked)
               hidden by: Does {B} cover part of {A}? Does {A} cover part    no_occlusion / reversed
                 of {B}? (p yes each)
  4 overall    Does this image show '{D}'? (p yes)                            -> DMS (phi_sco)
  5 describe   one sentence: where are {A} and {B}?  (free text, not scored) -- evidence for phi_rea

Per image: 8 questions when A != B, 4 when A == B.
Selection scores of one image (phi_sco averages them over the B images of a round):
  DMS (description match score) = overall_p = p(yes | "Does this image show 'D'?")
  OPS (object presence score)   = P(count_A >= 1) * P(count_B >= 1)   (A == B: P(count >= 2)^2)
Both need token log-probabilities from the MLLM (Ollama chat with logprobs; qwen2.5vl:7b).
"""

from __future__ import annotations

import contextlib
import json
import math
import re
import time
from collections import Counter
from typing import Any

from .eval.spatial import parse_description

SYSTEM_V4 = (
    "You are a careful visual inspector for text-to-image evaluation. Judge only what is visible in the "
    "image; do not guess from what is typical. Spatial terms refer to the camera's viewpoint: 'in front "
    "of' means closer to the camera, 'behind' means farther from the camera, 'hidden by' means partially "
    "covered by. Being on top of, above or below another object is not the same as being in front of or "
    "behind it. An object counts only if it appears as its own real object: a picture or pattern of it "
    "printed or painted on another object, or two objects fused into one, does not count.")

# failure type -> what the reasoner should change in the prompt
FIXES = {
    "missing_a": "the {a} is missing: name the {a} early in the prompt, make it a clearly visible, "
                 "reasonably large subject, and keep the prompt short so it is not dropped",
    "missing_b": "the {b} is missing: name the {b} early in the prompt, make it a clearly visible, "
                 "reasonably large subject, and keep the prompt short so it is not dropped",
    "not_two": "there are not two distinct {a}s: ask explicitly for two separate {a}s (e.g. 'two {a}s, "
               "one ... the other ...')",
    "crowd": "there are many copies of an object, overlapping each other: ask for exactly one {a} and "
             "one {b} (e.g. 'a single {a}', 'one {b}') and avoid plural words",
    "fused": "the {a} and the {b} are merged into one object: describe them as two separate, distinct "
             "objects with a visible gap or boundary, each with its own shape",
    "picture": "one object only appears as a picture/print/screen image on the other: ask for a real, "
               "physical {a} and a real, physical {b} (not a picture of it)",
    "reversed": "the depth order is reversed: put the object that must be in front in the foreground "
                "and the other in the background, explicitly ('X in the foreground, Y behind it in the background')",
    "same_depth": "the objects stand side by side at the same distance: add depth cues ('X in the "
                  "foreground, close to the camera; Y further back in the background'), a view from the "
                  "front so one is closer than the other",
    "stacked": "one object stands on top of the other instead of in front/behind it: say they stand on "
               "the ground at different distances from the camera and must not be on top of each other",
    "no_occlusion": "the objects do not overlap: ask for the {b} to stand directly in front of the {a} "
                    "and partially cover it ('the {a} is partly hidden behind the {b}, only part of the "
                    "{a} is visible')",
    "occlusion_reversed": "the wrong object is covered: the {a} must be partly hidden BY the {b} (the {b} "
                          "in front, covering part of the {a}), not the other way round",
}
FIX_ORDER = ["missing_a", "missing_b", "not_two", "picture", "fused", "crowd", "reversed",
             "occlusion_reversed", "stacked", "same_depth", "no_occlusion"]


# ---------------------------------------------------------------- the judge
class Judge:
    """MLLM inspector (phi_ans). ``client`` is an ``ollama.Client``-like object (``chat``, ``show``);
    by default one is created for ``host``."""

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", client: Any = None):
        if client is None:
            import ollama  # imported lazily so tests run without the package

            client = ollama.Client(host=host)
        self.model = model
        self.client = client
        self.system = SYSTEM_V4
        self.logprobs = True  # switched off after the first failure
        self.thinking = False
        try:
            caps = self.client.show(model)
            caps = caps.get("capabilities") if isinstance(caps, dict) else getattr(caps, "capabilities", None)
            self.thinking = bool(caps) and "thinking" in caps
        except Exception:  # noqa: BLE001
            self.thinking = "qwen3" in model

    @staticmethod
    def yn(text: str) -> str:
        try:
            text = json.loads(text).get("answer", text)
        except Exception:  # noqa: BLE001
            pass
        w = re.sub(r"[^a-z]", " ", str(text).lower()).split()
        return w[0] if w and w[0] in ("yes", "no") else "other"

    def unload(self) -> None:
        with contextlib.suppress(Exception):
            self.client.generate(model=self.model, prompt="", keep_alive=0)

    def _chat(self, image: str, question: str, num_predict: int, schema=None, logprobs=False):
        msgs = [{"role": "system", "content": self.system},
                {"role": "user", "content": question, "images": [image]}]
        kw = dict(model=self.model, messages=msgs, keep_alive="15m",
                  options={"temperature": 0, "seed": 0, "num_predict": 512 if self.thinking else num_predict})
        if schema:
            kw["format"] = schema
        use_lp = logprobs and self.logprobs and not self.thinking
        if use_lp:
            try:
                return self.client.chat(**kw, logprobs=True, top_logprobs=10), True
            except Exception:  # noqa: BLE001
                self.logprobs = False
        return self.client.chat(**kw), False

    @staticmethod
    def _text(r) -> str:
        msg = r["message"] if isinstance(r, dict) else r.message
        return (msg.get("content") if isinstance(msg, dict) else msg.content) or ""

    @staticmethod
    def _first_token_dist(r, tokens: tuple[str, ...]):
        lps = r.get("logprobs") if isinstance(r, dict) else getattr(r, "logprobs", None)
        for tok in lps or []:
            get = tok.get if isinstance(tok, dict) else (lambda k, t=tok: getattr(t, k, None))
            cands = {}
            for alt in (get("top_logprobs") or []):
                g = alt.get if isinstance(alt, dict) else (lambda k, a_=alt: getattr(a_, k, None))
                t = re.sub(r"[^a-z0-9]", "", str(g("token")).lower())
                if t in tokens:
                    cands[t] = cands.get(t, 0.0) + math.exp(g("logprob"))
            if cands:
                z = sum(cands.values())
                return {t: cands.get(t, 0.0) / z for t in tokens}
        return None

    def yes_no(self, image: str, q: str) -> float:
        r, lp = self._chat(image, q + " Answer yes or no.", 8, logprobs=True)
        d = self._first_token_dist(r, ("yes", "no")) if lp else None
        if d:
            return d["yes"]
        a = self.yn(self._text(r))
        return 1.0 if a == "yes" else 0.0 if a == "no" else 0.5

    def count(self, image: str, obj: str) -> dict:
        q = (f"How many {obj}s can you clearly see in the image? Count only real, separate {obj}s. "
             f"Answer with a single digit: 0, 1, 2, or 3 (3 means three or more).")
        r, lp = self._chat(image, q, 6, logprobs=True)
        d = self._first_token_dist(r, ("0", "1", "2", "3")) if lp else None
        if d:
            return d
        m = re.search(r"\d+", self._text(r))
        n = min(3, int(m.group())) if m else 1
        return {str(k): float(k == n) for k in range(4)}

    def closer(self, image: str, x: str, y: str) -> str:
        schema = {"type": "object", "properties": {"answer": {"type": "string", "enum": [x, y, "same"]}},
                  "required": ["answer"]}
        q = (f"Which object is closer to the camera: the {x} or the {y}? If they are at about the same "
             f"distance from the camera (for example side by side), answer 'same'. "
             f"Answer with one word: {x}, {y}, or same.")
        r, _ = self._chat(image, q, 24, schema=schema)
        t = self._text(r)
        with contextlib.suppress(Exception):
            t = json.loads(t).get("answer", t)
        t = str(t).lower()
        hits = [o for o in (x, y, "same") if o in t]
        return hits[0] if len(hits) == 1 else "other"

    def describe(self, image: str, a: str, b: str) -> str:
        q = (f"In one short sentence, describe where the {a} and the {b} are relative to each other, as seen "
             f"from the camera (which is in front, which is behind, whether they overlap or touch)."
             if a != b else
             f"In one short sentence, describe how many {a}s there are and how they are arranged relative to "
             f"each other, as seen from the camera.")
        r, _ = self._chat(image, q, 60)
        return " ".join(self._text(r).split())[:300]

    def probe(self, desc: str, image: str, b: int) -> dict:
        a, rel, bb = parse_description(desc)
        t0 = time.time()
        res = {"count_a": self.count(image, a)}
        if a != bb:
            res["count_b"] = self.count(image, bb)
            res["separate_p"] = self.yes_no(
                image, f"Are the {a} and the {bb} two separate, distinct objects (not merged into one object)?")
            res["picture_p"] = self.yes_no(
                image, f"Is the {a} or the {bb} only shown as a picture, print, screen image or pattern on another "
                       f"object, rather than as a real object?")
            if rel == "hidden by":
                res["b_covers_a_p"] = self.yes_no(image, f"Does the {bb} cover or block the view of part of the {a}?")
                res["a_covers_b_p"] = self.yes_no(image, f"Does the {a} cover or block the view of part of the {bb}?")
            else:
                x, y = (bb, a) if b % 2 == 0 else (a, bb)
                res["closer"], res["closer_order"] = self.closer(image, x, y), [x, y]
                res["stacked_p"] = self.yes_no(
                    image, f"Is one of the two objects (the {a} and the {bb}) standing on top of the other?")
        else:
            q = (f"Is one {a} partially hidden behind another {a}?" if rel == "hidden by"
                 else f"Is one {a} clearly closer to the camera than another {a}?")
            res["same_rel_p"] = self.yes_no(image, q)
        res["overall_p"] = self.yes_no(image, f"Does this image show '{desc}'?")
        res["describe"] = self.describe(image, a, bb)
        res["seconds"] = time.time() - t0
        return res


# ---------------------------------------------------------------- diagnosis
def p_present(count: dict, need: int = 1) -> float:
    return sum(v for k, v in count.items() if int(k) >= need)


def diagnose(desc: str, r: dict) -> dict:
    """Failure types (ordered), relation probability and the selection scores of one image."""
    a, rel, b = parse_description(desc)
    fails = []
    if a == b:
        pa = pb = p_present(r["count_a"], 2)
        if pa < 0.5:
            fails.append("not_two")
        p_rel = r["same_rel_p"]
        if pa >= 0.5 and p_rel < 0.5:
            fails.append("no_occlusion" if rel == "hidden by" else "same_depth")
    else:
        pa, pb = p_present(r["count_a"]), p_present(r["count_b"])
        if pa < 0.5:
            fails.append("missing_a")
        if pb < 0.5:
            fails.append("missing_b")
        if pa >= 0.5 and pb >= 0.5:
            if r["picture_p"] >= 0.5:
                fails.append("picture")
            elif r["separate_p"] < 0.5:
                fails.append("fused")
        if max(r["count_a"], key=r["count_a"].get) == "3" or max(r["count_b"], key=r["count_b"].get) == "3":
            fails.append("crowd")
        if rel == "hidden by":
            p_rel = r["b_covers_a_p"] * (1 - 0.5 * r["a_covers_b_p"])
            if not fails or fails == ["crowd"]:
                if r["b_covers_a_p"] < 0.5:
                    fails.append("occlusion_reversed" if r["a_covers_b_p"] >= 0.5 else "no_occlusion")
        else:
            want = a if rel == "in front of" else b
            p_rel = 1.0 if r["closer"] == want else 0.0
            if not fails or fails == ["crowd"]:
                if r["closer"] == "same":
                    fails.append("stacked" if r["stacked_p"] >= 0.5 else "same_depth")
                elif r["closer"] in (a, b) and r["closer"] != want:
                    fails.append("reversed")
                elif r["stacked_p"] >= 0.5:
                    fails.append("stacked")
    presence = pa * pb
    return {"fails": fails, "ok": not fails, "p_rel": p_rel, "presence": presence,
            "overall": r["overall_p"],
            "combined": (presence + r["overall_p"] + presence * p_rel) / 3}


def feedback_text(desc: str, results: list[dict]) -> str:
    """Structured feedback for phi_rea from the B probed images of one prompt."""
    a, rel, b = parse_description(desc)
    diags = [diagnose(desc, r) for r in results]
    lines = [f"Diagnosis of the {len(results)} images generated from the current prompt "
             f"(target: '{desc}'):"]
    for i, (r, d) in enumerate(zip(results, diags), 1):
        status = "matches the description" if d["ok"] else "problems: " + ", ".join(d["fails"])
        lines.append(f"- image {i}: {status}. Seen: {r.get('describe') or '(no description)'}")
    ok = sum(d["ok"] for d in diags)
    cnt = Counter(f for d in diags for f in d["fails"])
    lines.append(f"Summary: {ok}/{len(diags)} images match the description.")
    if cnt:
        lines.append("Problems, most frequent first, with the fix to apply:")
        for f in sorted(cnt, key=lambda f: (-cnt[f], FIX_ORDER.index(f))):
            lines.append(f"- {f} in {cnt[f]}/{len(diags)} images -> {FIXES[f].format(a=a, b=b)}")
    else:
        lines.append("No problem found: keep the prompt structure, change as little as possible.")
    return "\n".join(lines)


# ---------------------------------------------------------------- phi_sco inputs
def image_scores(desc: str, r: dict) -> dict:
    """DMS and OPS of one probed image."""
    d = diagnose(desc, r)
    return {"dms": r["overall_p"], "ops": d["presence"]}
