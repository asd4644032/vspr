"""TIFA question parsing and filtering (copied from tifascore) without loading any model."""

from vspr.eval import tifa as T

RESP = """ a, sofa, plate
Activites:
Colors:
Counting:
Other attributes:
Questions and answers are below:
About sofa (furniture):
Q: is this a sofa?
Choices: yes, no
A: yes
About sofa (object):
Q: is there a sofa?
Choices: yes, no
A: yes
About plate (object):
Q: what is in front of the plate?
Choices: sofa, chair, table, bed
A: sofa
About in front of (spatial):
Q: is the sofa in front of the plate?
Choices: yes, no
A: yes"""


def test_parse_and_categories():
    qas = T.to_qas("a sofa in front of a plate", T.parse_resp(RESP))
    # "furniture" is not a TIFA category and is dropped, as in tifascore
    assert [q["element_type"] for q in qas] == ["object", "object", "spatial"]
    assert qas[1]["choices"] == ["sofa", "chair", "table", "bed"] and qas[1]["answer"] == "sofa"


class FakeQA:
    def mcqa(self, question, context, choices=("yes", "no")):
        return "no" if "plate" in question and "sofa in front" not in question and choices[0] == "yes" else (
            "sofa" if choices[0] == "sofa" else "yes")

    def qa(self, question, context):
        return "a sofa"


def test_filter_keeps_consistent_and_drops_duplicates():
    qas = T.to_qas("a sofa in front of a plate", T.parse_resp(RESP))
    qas.append(dict(qas[0]))  # duplicate question
    kept = T.filter_question_and_answers(FakeQA(), qas)
    assert [q["question"] for q in kept] == ["is there a sofa?", "what is in front of the plate?",
                                              "is the sofa in front of the plate?"]


def test_prompt_format():
    p = T.create_qg_prompt("a cat behind a dog")
    assert p.startswith("<s>[INST] <<SYS>>\n") and p.endswith("Description: a cat behind a dog [/INST] Entities:")
    assert T.mc_prompt("is there a cat?", ["yes", "no"]) == \
        "Answer the multiple-choice question. Question: is there a cat? Choices: yes, no Answer:"
