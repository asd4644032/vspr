"""Deterministic stand-in for ``ollama.Client`` (chat with logprobs, show, generate).

Answers depend only on the question text, so the frozen experiment judge (vqa_test/vqa_v4.py)
and vspr.judge can be fed the same responses; the golden file tests/golden/judge_messages.json
was produced with this client and the experiment code (vqa_test/e5_check/make_judge_golden.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import re


def _u(text: str, k: int = 0) -> float:
    """Pseudo-random number in (0, 1) from the text."""
    h = hashlib.sha256(f"{k}:{text}".encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + 1) / (2**32 + 2)


class FakeOllama:
    def __init__(self, logprobs: bool = True, capabilities=("completion", "vision")):
        self.logprobs = logprobs
        self.capabilities = list(capabilities)
        self.calls: list[dict] = []

    def show(self, model):
        return {"capabilities": self.capabilities}

    def generate(self, **kw):
        return {"response": ""}

    def chat(self, **kw):
        if kw.get("logprobs") and not self.logprobs:
            raise TypeError("logprobs not supported")
        self.calls.append(json.loads(json.dumps(kw, default=str)))
        q = kw["messages"][-1]["content"]
        if "format" in kw:  # closer(): JSON enum answer
            enum = kw["format"]["properties"]["answer"]["enum"]
            ans = enum[int(_u(q) * len(enum))]
            return {"message": {"content": json.dumps({"answer": ans})}}
        if "single digit" in q:
            ps = [_u(q, k) for k in range(4)]
            content = str(max(range(4), key=lambda k: ps[k]))
            top = [{"token": str(k), "logprob": math.log(ps[k] / 4)} for k in range(4)]
            top.append({"token": "The", "logprob": math.log(0.05)})
        elif "yes or no" in q:
            p = _u(q)
            content = "Yes" if p >= 0.5 else "No"
            top = [{"token": "Yes", "logprob": math.log(p * 0.9)},
                   {"token": "No", "logprob": math.log((1 - p) * 0.9)},
                   {"token": " yes", "logprob": math.log(0.01)}]
        else:  # describe()
            words = re.findall(r"the (\w+)", q)
            return {"message": {"content": f"  The {' and the '.join(words[:2])} are\n side by side. "}}
        msg = {"message": {"content": content}}
        if kw.get("logprobs"):
            msg["logprobs"] = [{"token": content, "logprob": -0.1, "top_logprobs": top}]
        return msg
