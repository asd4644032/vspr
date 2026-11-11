"""Instructions of phi_rea (the reasoner) in VSPR: the "v5" rules.

The text is copied verbatim from the experiment code that produced every validation result
(vqa_test/refine_pilot.py: V5_HEAD, V5_INPUT_VQA, V5_INPUT_NOVQA, V5_POSITIVE, V5_RULE_STEPS_P,
V5_EXAMPLES, V5_OUT_NONEG, v5_system, v5_user; the 60-word limit from vqa_test/vqa_v5.py).
Do not edit the wording; tests/test_prompts.py compares it with a golden file.

Two variants:
  "vqa"    (VSPR)     D, the current prompt P and the diagnosis of the B images (judge.feedback_text)
  "novqa"  (w/o VQA)  D and the current prompt P only; the model predicts the likely failures
No negative prompt is written; the generator always uses the fixed negative "text, watermark".
"""

MAX_PROMPT_WORDS = 60  # longer prompts are cut at a comma (truncate_prompt)
MAX_POS_WORDS = MAX_PROMPT_WORDS

# ---------------------------------------------------------------- phi_rea system prompt (v5 rules)
V5_HEAD = """You refine prompts for Stable Diffusion XL. It cannot reason; it draws what the words emphasize. \
The description D names two objects and their spatial relation as seen from the camera. {inputs}"""

V5_INPUT_VQA = """You get D, the current prompt P{and_n} and an inspection of the {batch} images made from \
them: per image the problems found and what is visible, then the problems by frequency with fixes{extra}."""

V5_INPUT_NOVQA = """You get D and the current prompt P, but NO images: predict the failures this kind of \
description usually causes (list below) and prevent them."""

V5_POSITIVE = """
POSITIVE PROMPT P (at most 60 words)
Name the front object (the occluder) first, then the hidden one:
- "A hidden by B" -> "B in front of A, B's <part> hiding part of A"
- "A behind B" -> "B in front of A, B covering part of A"
- "A in front of B" -> "A close to the camera, B behind A, A hiding part of B"
Write "a single <object>" or name each object once. End with "photo". Keep the same two objects and the \
same relation; never swap their roles.

Failure -> positive fix
- reversed / occlusion_reversed: remove the back object's support surface ("on a desk/table/shelf"), put it \
"on the ground/floor" or outdoors; name the part of the front object that hides it ("the cow's head hiding \
part of the laptop").
- missing_a / missing_b: weight the missing noun (noun:1.3), same weight for both objects; no "small"; \
"full body" instead of close-up actions; a more typical name ("picnic table").
- picture_a / picture_b / picture / fused: "a real <object>", "photo"; with a painting or screen add \
"shallow depth of field, <front object> sharp, <painting/screen> blurred".
- stacked / riding: "<A> under <B>, seen through <B>'s legs" or "<B> parked in front of <A>", low angle.
- same_depth / no_occlusion: "directly behind", "overlapping", "covering part of", name the overlapping part.
- crowd: "a single <object>".
- wrong_kind (another sense of the word, e.g. a computer mouse for an animal mouse): name the right kind \
("a live gray mouse").
- small object that must be large or hidden: low angle or close-up.
Do not: weight only the relation phrase "(partly hidden by X:1.2)"; write only "in the background"; use the \
bare word "behind" as the relation; cover a head with a lampshade or hat to show occlusion."""

V5_RULE_STEPS_P = """
Steps: 1. Name the 1-3 failures most likely for D (small object behind a large one -> missing or reversed; \
animal with a screen or painting -> picture; small animal with furniture -> stacked; mouse -> wrong_kind). \
2. Write P with the template and their positive fixes. 3. If P was already refined, change it only where a \
likely failure is not yet covered."""

V5_EXAMPLES = """
Verified examples (failures seen -> {what})
1. D: a computer hidden by a cow. Seen: laptop on a desk in front of the cow (reversed); cow on the screen (picture_b).
   P: photo, close-up of a cow's head in front of an open laptop computer on the grass, the cow's head hiding part of the laptop screen{n12}
2. D: a chair hidden by a mouse. Seen: a computer mouse (wrong_kind); mouse on the chair (stacked).
   P: low angle photo, a small gray mouse on the floor close to the camera, a wooden chair behind the mouse, (the chair leg partly hidden by the mouse:1.2){n15}
3. D: a sheep hidden by a bicycle. Seen: the sheep rides the bicycle (riding).
   P: a bicycle parked close to the camera in the foreground, a sheep standing behind the bicycle, (the sheep partly hidden by the bicycle wheels and frame:1.2), photo{n19}
4. D: a painting hidden by a butterfly. Seen: the butterfly is inside the painting (picture_b).
   P: close-up of a real butterfly flying in front of a framed painting on a wall, the butterfly's wings blocking part of the painting, butterfly in sharp focus, painting blurred
5. D: a frog hidden by a chair. Seen: the frog sits on the chair (stacked).
   P: low angle photo, a frog sitting under a wooden chair, seen through the chair legs, (the frog partly hidden behind a chair leg:1.3)
6. D: a girl hidden by a computer. Seen: her face on the screen (picture_a); no overlap.
   P: photo, a girl peeking over the top of a computer monitor, the monitor hiding the lower half of her face, the screen turned off and black{neg_only}"""

V5_OUT_NONEG = """
Return JSON: "explanation" (short step-by-step analysis) and "prompt" (P)."""

# Keys of the few-shot examples that only the negative-prompt variant fills in (empty here).
_NEG_EXAMPLE_KEYS = ("n12", "n15", "n19", "neg_only")


def v5_system(mode: str = "vqa", batch: int = 4) -> str:
    """System prompt of phi_rea. mode: "vqa" (VSPR; called "noneg" in the experiments) or "novqa"."""
    if mode == "noneg":
        mode = "vqa"
    if mode not in ("vqa", "novqa"):
        raise ValueError(f"unknown phi_rea mode {mode!r}")
    if mode == "novqa":
        inputs = V5_INPUT_NOVQA
    else:
        inputs = V5_INPUT_VQA.format(and_n="", batch=batch, extra="")
    parts = [V5_HEAD.format(inputs=inputs), V5_POSITIVE]
    if mode == "novqa":
        parts.append(V5_RULE_STEPS_P)
    parts.append(V5_EXAMPLES.format(what="P", **{k: "" for k in _NEG_EXAMPLE_KEYS}))
    parts.append(V5_OUT_NONEG)
    return "\n".join(parts)


def v5_user(description: str, prompt: str, feedback: str | None) -> str:
    """User message of phi_rea; ``feedback=None`` for the w/o VQA variant."""
    lines = [f"Description D: {description}", f"Current prompt P: {prompt}"]
    lines.append("")
    lines.append(feedback if feedback else "(no images: predict the likely failures)")
    return "\n".join(lines)


def truncate_prompt(p: str, limit: int = MAX_POS_WORDS) -> tuple[str, bool]:
    if len(p.split()) <= limit:
        return p, False
    parts, out = p.split(","), []
    for part in parts:
        if len(" ".join(out + [part]).split()) > limit:
            break
        out.append(part)
    s = ",".join(out).strip(" ,") if out else " ".join(p.split()[:limit])
    return s, True


REASON_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "prompt": {"type": "string"},
    },
    "required": ["explanation", "prompt"],
}
