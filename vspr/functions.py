"""The four task functions of VSPR (Section 3, Algorithm 1).

phi_que / phi_ans  fixed diagnostic VQA questions asked on every image (vspr.judge.Judge.probe)
phi_rea            LLM reasoner with the v5 rules (vspr.prompts) -> explanation T, refined prompt P'
phi_sco            s = a * DMS + (1 - a) * OPS, averaged over the B images of a round
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass

from . import prompts as P
from .judge import Judge, diagnose
from .llm import LLMClient


@dataclass
class RoleConfig:
    model: str
    options: dict
    # where the v5 rules go: "system" (a system message) or "user" (prepended to the user message).
    # DeepSeek-R1's model card advises against system prompts, so llm.reasoner_system = auto picks
    # "user" for it (vspr.factory.system_placement); Llama 3.1 keeps the system message of the paper.
    system_placement: str = "system"


def reasoner_messages(system: str, user: str, placement: str = "system") -> list[dict]:
    """Chat messages for phi_rea; ``placement="user"`` sends no system message."""
    if placement == "system":
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if placement == "user":
        return [{"role": "user", "content": f"{system}\n\n{user}"}]
    raise ValueError(f"system_placement must be 'system' or 'user', not {placement!r}")


# ------------------------------------------------------------ phi_que / phi_ans
def judge_batch(judge: Judge, description: str, images: list[str]) -> list[dict]:
    """Ask the fixed questions on every image of the batch (image index b starts at 1)."""
    return [judge.probe(description, img, b) for b, img in enumerate(images, 1)]


# ------------------------------------------------------------------ phi_rea
def phi_rea(
    llm: LLMClient,
    role: RoleConfig,
    description: str,
    prompt: str,
    feedback: str | None,
    batch: int,
    use_vqa: bool = True,
) -> dict:
    """Return the explanation T and the refined prompt P' (cut to 60 words at a comma).

    With ``use_vqa`` the reasoner gets the diagnosis of the B images (``judge.feedback_text``);
    without it (w/o VQA ablation) it only sees D and the current prompt.
    """
    mode = "vqa" if use_vqa else "novqa"
    messages = reasoner_messages(P.v5_system(mode, batch),
                                 P.v5_user(description, prompt, feedback if use_vqa else None),
                                 role.system_placement)
    res = llm.chat_json(role.model, messages, P.REASON_SCHEMA, role.options)
    raw_prompt = " ".join(str(res.data.get("prompt", "")).split()) or prompt
    next_prompt, truncated = P.truncate_prompt(raw_prompt)
    return {
        "mode": mode,
        "system_placement": role.system_placement,
        "explanation": str(res.data.get("explanation", "")).strip(),
        "raw_thinking": res.thinking,  # DeepSeek-R1 <think> trace; empty for Llama 3.1
        "raw_output": res.raw,
        "prompt_raw": raw_prompt,
        "next_prompt": next_prompt,
        "truncated": truncated,
        "words": len(next_prompt.split()),
    }


# ------------------------------------------------------------------ phi_sco
def round_scores(description: str, results: list[dict]) -> dict:
    """DMS and OPS of one round: the mean over its B probed images."""
    diags = [diagnose(description, r) for r in results]
    return {
        "dms": st.fmean(r["overall_p"] for r in results),
        "ops": st.fmean(d["presence"] for d in diags),
        "per_image": [{"dms": r["overall_p"], "ops": d["presence"], "fails": d["fails"]}
                      for r, d in zip(results, diags)],
    }


def phi_sco(dms: float, ops: float, a: float) -> float:
    """s = a * DMS + (1 - a) * OPS  (a = vspr.dms_weight, 0.5 in the paper)."""
    return a * dms + (1 - a) * ops
