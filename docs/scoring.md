# The scoring function φ_sco

φ_sco decides which round's prompt–image pair (*P\**, *I\**) VSPR returns. It reads the
images, not the reasoner's text, and it needs no language-model call: both terms come from
the answers of the MLLM to the fixed questions of φ_que / φ_ans ([`vspr/judge.py`](../vspr/judge.py)).

## Definition

For round *i* with prompt *P_i* and images *I_i^1..I_i^B*:

```
m(I, D) = p_MLLM(yes | I, "Does this image show 'D'?")           # description match score, DMS
o(I, D) = P(count_A >= 1) * P(count_B >= 1)                        # object presence score, OPS
          (A = B, e.g. "a candle hidden by a candle": P(count_A >= 2)^2)

DMS_i = (1/B) Σ_b m(I_i^b, D)
OPS_i = (1/B) Σ_b o(I_i^b, D)
s_i   = a * DMS_i + (1 - a) * OPS_i                                # a = 0.5
```

* *m* is the probability of the first answer token: the log-probabilities of the top-10
  candidates whose text is "yes" or "no" (case and spaces ignored) are exponentiated, summed
  per answer and normalised, `Judge.yes_no`.
* *P(count ≥ k)* comes from the distribution over the first answer token "0", "1", "2", "3"
  of "How many {A}s can you clearly see in the image? …", `Judge.count`.
* Both need token log-probabilities (an Ollama version whose chat API accepts `logprobs`), with `qwen2.5vl:7b` decoded greedily
  (temperature 0, seed 0). Without log-probabilities the judge falls back to the text answer
  (0 or 1), which makes the scores coarse; every result in the paper used log-probabilities.

**Selection:** *i\* = argmax_i s_i* over rounds 1..*M*+1; ties go to the earlier round.
"VSPR iter *k*" is the best of rounds 1..*k*+1, so iter 1 (best of *P_1 = D* and *P_2*) is the
*w/o iterative* ablation.

**w/o VQA:** the images are not inspected at all, so there is no φ_sco; the output is the
last round (*P_{M+1}*), and iter *k* is round *k*+1.

The VQA results also drive φ_rea: the diagnosis of every image (failure types and a one-line
description) and the fixes ordered by frequency (`judge.feedback_text`). They are the
reasoner's input, not part of the score.

## Why DMS and OPS

The two terms measure different things, and human labels confirm it.

Human validation of *a* (`claude/a-results.md`): 30 descriptions × rounds 1–3 of the final
method (v5 rules, SDXL) × 4 images = 360 images, all rated by a person on a 0–3 scale
(3 = matches the description; 0 = an object is missing). The selector picks one round per
description; the table reports the 4 images of the picked round.

| Selection | strictly matches (grade 3) | mean grade | grade 0 (object problem) |
|---|---|---|---|
| random round (expected) | 18.9 % | 1.28 | 20.6 % |
| always round 1 (*P = D*) | 17.5 % | 1.17 | 25.0 % |
| always round 3 | 21.7 % | 1.38 | 17.5 % |
| a = 0 (OPS only) | 21.7 % | 1.37 | 17.5 % |
| a = 0.25 | 22.5 % | **1.43** | 16.7 % |
| **a = 0.5** | 22.5 % | **1.40** | 19.2 % |
| a = 0.75 | 21.7 % | 1.35 | 21.7 % |
| a = 1 (DMS only) | 22.5 % | 1.38 | 20.8 % |
| oracle (human best round) | 25.0 % | 1.54 | 10.8 % |

Against a random round (bootstrap over descriptions, 95 % CI), only a = 0.25
(+0.14 [+0.04, +0.27]) and a = 0.5 (+0.12 [+0.01, +0.24]) improve the mean grade
significantly. a = 0.5 was fixed beforehand (in E1) and was not tuned on these images.

Image-level AUC on the same 360 images:

| Score | predicts "strictly matches" | predicts "both objects present" (grade ≥ 1) |
|---|---|---|
| DMS | 0.877 | 0.610 |
| OPS | 0.602 | **0.815** |
| 0.5·DMS + 0.5·OPS | **0.886** | 0.710 |

DMS judges whether the image matches the description; OPS judges whether both objects are
there. The combination is the best predictor of a strict match.

The earlier design, s = 0.5·acc + 0.5·q (acc = agreement with five LLM-written yes/no
questions, q = an LLM rating of the reasoner's analysis), was dropped after the first
validation (E1: 30 descriptions × 4 rewrites × 4 images = 479 human-labelled images, plus 296
images of the old experiments):

| Score | image AUC, 479 blind-rated | image AUC, 296 old | rounds: picked − random (×100) |
|---|---|---|---|
| q (LLM rating) | 0.44 | 0.54 | +2.3 [−2.5, +7.7] |
| acc (LLM questions) | 0.61 | 0.58 | +2.9 [−3.5, +9.4] |
| 0.5·acc + 0.5·q (old) | 0.54 | 0.58 | +2.7 [−2.1, +7.9] |
| DMS | **0.80** | **0.82** | +6.9 [+0.2, +14.6] |
| 0.5·DMS + 0.5·OPS | 0.81 | 0.79 | +7.7 [+1.0, +16.0] |

Reproduce the E1 image-level table with
`python scripts/validate_scorer.py human --vqa-test <vqa_test>` (needs the experiment
folders; the acc / q rows came from the pre-final code and are not part of this repository).

## Configuration

* `vspr.dms_weight` is *a* (default 0.5).
* `llm.mllm` is the judge (default `qwen2.5vl:7b`); it must return token log-probabilities.
* `vspr.use_vqa: false` runs the *w/o VQA* ablation (no judge, last round).
* `scripts/validate_scorer.py` (metric mode) reports, per evaluation metric, the round picked by
  *s*, by DMS alone and by OPS alone, next to round 1, the last round and the oracle round
  (`run_vspr.py --export-rounds` first).

## Limitations

* Validated only with `qwen2.5vl:7b` on SDXL images. Another MLLM needs log-probabilities
  and should be validated again.
* "hidden by" descriptions: no score separates good from bad rounds (image AUC ≤ 0.69 in E1);
  SDXL rarely draws the occlusion at all, and even the oracle round matches only 28 %.
* With 30 descriptions the values of *a* cannot be told apart, and the selector is not clearly
  better than always taking the last round on the human labels (strict match 22.5 % vs 21.7 %).
  On the automatic metrics of 97 descriptions, the selected round beats round 3 (ITM 81.1 vs
  75.4, BLIP-VQA 38.4 vs 32.8).
* The selector still misses rounds in which an object disappears (grade 0: 19.2 % vs 10.8 % for
  the oracle).
