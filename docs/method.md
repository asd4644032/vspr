# Method: the four task functions

VSPR refines a prompt for a frozen text-to-image model *G*. Every round generates *B* images
from the current prompt (same seed in every round), inspects them with an MLLM, scores the
round, and lets an LLM rewrite the prompt from the inspection. The best round is returned.

```
P_1 <- D
for i = 1 .. M+1:
    I_i <- G(P_i)                                   B = 4 images
    R_i <- phi_ans(I_i^b, phi_que(D))  for all b    fixed questions, qwen2.5vl:7b
    s_i <- phi_sco(R_i)                             a * DMS_i + (1 - a) * OPS_i
    if i <= M:  T_i, P_{i+1} <- phi_rea(D, P_i, feedback(R_i))
return the round with the highest s_i (ties -> earlier round)
```

Defaults (`configs/default.yaml`): *M* = 3, *B* = 4, *a* = 0.5, reasoner `llama3.1:8b`
(alternative `deepseek-r1:14b`), MLLM `qwen2.5vl:7b`, fixed negative prompt "text, watermark",
SDXL guidance scale 8 (the workflow's value), seed = hash(42, *D*).

## φ_que / φ_ans: fixed diagnostic questions

Code: [`vspr/judge.py`](../vspr/judge.py) (`Judge.probe`, `diagnose`, `feedback_text`).

The questions are not written by an LLM. *D* always has the form "a(n) A *relation* a(n) B"
with relation ∈ {in front of, behind, hidden by}; `vspr.eval.spatial.parse_description` reads
A, the relation and B, and the same questions are asked on every image, with the system prompt
`SYSTEM_V4` (camera viewpoint; a picture of an object or two fused objects do not count):

| Layer | Question (A ≠ B) | Answer read as | Failure types |
|---|---|---|---|
| count | "How many {A}s can you clearly see …? Answer with a single digit: 0, 1, 2, or 3" (and B) | distribution over the first token | missing_a, missing_b, crowd |
| integrity | "Are the {A} and the {B} two separate, distinct objects …?" | p(yes) | fused |
| | "Is the {A} or the {B} only shown as a picture, print, screen image or pattern …?" | p(yes) | picture |
| relation (in front of / behind) | "Which object is closer to the camera: the {X} or the {Y}? … same" (X/Y order alternates between images) | {X, Y, same} | reversed, same_depth |
| | "Is one of the two objects … standing on top of the other?" | p(yes) | stacked |
| relation (hidden by) | "Does the {B} cover or block the view of part of the {A}?" and the reverse | p(yes) each | no_occlusion, occlusion_reversed |
| overall | "Does this image show '{D}'?" | p(yes) = **DMS** | — |
| describe | "In one short sentence, describe where the {A} and the {B} are …" | free text | evidence for φ_rea |

When A = B (e.g. "a candle hidden by a candle") the questions are: count of A (2 needed), one
relation question, overall, describe. So each image takes 8 MLLM calls (4 when A = B).

`diagnose` turns the answers of one image into an ordered list of failure types and the
selection scores; `feedback_text` writes the reasoner's input: per image its failures and the
one-sentence description, then the failures by frequency with the fix for each (`FIXES`).
The wording is exactly the one used in all validation experiments; `tests/test_judge.py`
checks it against a golden file.

## φ_rea: the v5 rules

Code: [`vspr/prompts.py`](../vspr/prompts.py) (`v5_system`, `v5_user`, `truncate_prompt`) and
`vspr.functions.phi_rea`.

The system prompt (`v5_system("vqa")`, about 3.7 k characters) has four parts, kept verbatim:

1. `V5_HEAD` + `V5_INPUT_VQA`: the task and what the reasoner receives.
2. `V5_POSITIVE`: the prompt template (front object / occluder first; "a single <object>";
   end with "photo"; at most 60 words; never swap the roles) and a failure → fix table for
   every failure type of φ_ans, plus a short "Do not" list.
3. `V5_EXAMPLES`: six verified examples (failures seen → prompt that fixed them).
4. `V5_OUT_NONEG`: return JSON with "explanation" (*T*) and "prompt" (*P'*).

The user message is *D*, the current prompt *P_i* and `feedback_text` of round *i*.
*P'* longer than 60 words is cut at the last comma that keeps it within 60 words
(`truncate_prompt`; CLIP truncates long prompts anyway). VSPR writes no negative prompt.

For DeepSeek-R1 the reasoning trace (`<think>`) is stored as `raw_thinking`; *T* is the
`explanation` field for both reasoners. The v5 rules were developed and validated with
`llama3.1:8b` only.

**w/o VQA ablation** (`vspr.use_vqa: false`): `v5_system("novqa")` replaces the input part
with `V5_INPUT_NOVQA` ("you get D and P, but no images: predict the failures this kind of
description usually causes") and adds `V5_RULE_STEPS_P`; the user message ends with
"(no images: predict the likely failures)". No image is inspected and there is no φ_sco: the
output is the last round.

## φ_sco: DMS and OPS

Code: `vspr.functions.round_scores`, `phi_sco`, `vspr.pipeline.select_best`.
s_i = a·DMS_i + (1 − a)·OPS_i with a = 0.5; definitions, validation and limitations are in
[scoring.md](scoring.md).

## Computational cost per description

| Calls | Formula | *M* = 3, *B* = 4 |
|---|---|---|
| MLLM (φ_que / φ_ans) | (questions per image) · *B* · (*M*+1) = 8·*B*·(*M*+1) | 128 (64 when A = B) |
| LLM (φ_rea) | *M* | 3 |
| generated images | *B*·(*M*+1) | 16 |

φ_sco needs no model call. Measured in the validation runs (SDXL base + refiner): about 38 s per batch
of 4 images, 10 s to inspect a batch, 5 s per φ_rea call.

## Files of a run

`outputs/vspr/<backend>_<reasoner>[_novqa]/items/<id>_<slug>/round_<i>/`:

| File | Content |
|---|---|
| images.json | prompt *P_i*, seed, negative, image paths |
| judge.json | the probe results of the *B* images and their diagnosis (VSPR only) |
| score.json | DMS, OPS, *s*, *a*, per-image values (VSPR only) |
| reasoning.json | feedback sent to φ_rea, explanation, raw output, raw thinking, the refined prompt before and after truncation (rounds 1..*M*) |
