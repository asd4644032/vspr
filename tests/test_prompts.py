"""phi_rea messages against the experiment code (vqa_test/refine_pilot.py, v5-noneg / v5-noneg-novqa).

golden/v5_phi_rea.json holds the system prompts produced by refine_pilot.v5_system("noneg") and
v5_system("novqa"), and the user messages of 20 real calls (5 descriptions x rounds 2-3 x both
variants) with the feedback text that the experiment actually sent (vqa_test/pilot2/sdxl/state.json).
"""

from __future__ import annotations

import json
from pathlib import Path

from vspr import prompts as P
from vspr.functions import RoleConfig, phi_rea
from vspr.llm import LLMResult

GOLDEN = json.loads((Path(__file__).resolve().parent / "golden" / "v5_phi_rea.json").read_text(encoding="utf-8"))


class RecordingLLM:
    def __init__(self, reply):
        self.reply, self.messages = reply, []

    def chat_json(self, model, messages, schema, options=None):
        self.messages.append((messages, schema, options))
        return LLMResult(self.reply, raw=json.dumps(self.reply))


def test_system_prompts_verbatim():
    assert P.v5_system("vqa", 4) == GOLDEN["system_noneg"]
    assert P.v5_system("noneg", 4) == GOLDEN["system_noneg"]
    assert P.v5_system("novqa", 4) == GOLDEN["system_novqa"]
    assert "NEGATIVE PROMPT" not in P.v5_system("vqa") and "\n   N: " not in P.v5_system("vqa")


def test_phi_rea_messages_verbatim():
    assert len(GOLDEN["calls"]) == 20
    for c in GOLDEN["calls"]:
        use_vqa = c["mode"] == "noneg"
        llm = RecordingLLM({"explanation": "e", "prompt": c["next_prompt_pilot2"]})
        out = phi_rea(llm, RoleConfig("llama3.1:8b", {"seed": 42}), c["description"], c["prompt"],
                      c["feedback"], 4, use_vqa=use_vqa)
        (messages, schema, options), = llm.messages
        assert messages[0] == {"role": "system", "content": GOLDEN["system_noneg" if use_vqa else "system_novqa"]}
        assert messages[1] == {"role": "user", "content": c["user"]}
        assert schema == P.REASON_SCHEMA and options == {"seed": 42}
        assert out["next_prompt"] == c["next_prompt_pilot2"] and not out["truncated"]


def test_truncate_prompt():
    short = "a cat, a dog, photo"
    assert P.truncate_prompt(short) == (short, False)
    long = ", ".join(f"part {k} with words" for k in range(30))  # 4 words per part
    cut, t = P.truncate_prompt(long)
    assert t and len(cut.split()) <= 60 and long.startswith(cut) and not cut.endswith(",")
    no_comma = " ".join(["word"] * 70)
    assert P.truncate_prompt(no_comma) == (" ".join(["word"] * 60), True)
    first_too_long = " ".join(["w"] * 61) + ", tail"
    assert P.truncate_prompt(first_too_long) == (" ".join(["w"] * 60), True)


def test_empty_prompt_falls_back_to_current():
    llm = RecordingLLM({"explanation": "e", "prompt": "  "})
    out = phi_rea(llm, RoleConfig("m", {}), "a cat behind a dog", "old prompt", "fb", 4)
    assert out["next_prompt"] == "old prompt"


def test_rules_in_user_message_for_deepseek():
    from vspr.factory import system_placement
    from vspr.functions import reasoner_messages

    assert system_placement("llama3.1:8b") == "system"
    assert system_placement("deepseek-r1:14b") == "user"
    assert system_placement("deepseek-r1:14b", "system") == "system"
    sys_msg, user_msg = P.v5_system("vqa", 4), P.v5_user("a cat behind a dog", "p", "fb")
    assert reasoner_messages(sys_msg, user_msg) == [{"role": "system", "content": sys_msg},
                                                   {"role": "user", "content": user_msg}]
    llm = RecordingLLM({"explanation": "e", "prompt": "a dog in front of a cat, photo"})
    out = phi_rea(llm, RoleConfig("deepseek-r1:14b", {}, "user"), "a cat behind a dog", "p", "fb", 4)
    (msgs, _, _), = llm.messages
    assert msgs == [{"role": "user", "content": sys_msg + "\n\n" + user_msg}]
    assert out["system_placement"] == "user"
