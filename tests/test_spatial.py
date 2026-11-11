"""Spatial (UniDet) scorer: label coverage, depth indexing, pair selection, formula."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vspr.eval.spatial import OBJECT_LABELS, parse_description, score_image

REPO = Path(__file__).resolve().parent.parent
DESCS = [l.strip() for l in (REPO / "data/t2i_compbench/3d_spatial.txt").read_text().splitlines() if l.strip()]
UNIDET = set(json.loads((REPO / "vspr/eval/unidet_labels.json").read_text()))


def test_every_description_parses_and_every_object_is_mapped():
    for d in DESCS:
        o1, rel, o2 = parse_description(d)
        assert o1 in OBJECT_LABELS and o2 in OBJECT_LABELS, d


def test_mapped_labels_exist_in_unidet():
    for obj, labels in OBJECT_LABELS.items():
        for lab in labels:
            assert lab in UNIDET, (obj, lab)


def test_only_wallet_is_unscorable():
    assert [o for o, l in OBJECT_LABELS.items() if not l] == ["wallet"]


def det(*items, w=100, h=100):
    return {"width": w, "height": h, "boxes": [b for _, b, _ in items],
            "labels": [l for l, _, _ in items], "scores": [s for _, _, s in items]}


def depth_map(near_box, w=100, h=100):
    d = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = near_box
    d[y0:y1, x0:x1] = 1.0  # larger = nearer (DPT inverse depth)
    return d


def test_in_front_correct_order_full_overlap():
    a, b = (10, 10, 60, 60), (12, 12, 62, 62)  # IoU > 0.5
    s = score_image("a cat in front of a dog", det(("cat", a, 0.8), ("dog", b, 0.6)), depth_map((10, 10, 40, 40)))
    assert s == pytest.approx(0.5 * (0.8 + 0.6) / 2 + 0.5 * 1.0)


def test_wrong_order_gives_object_score_only():
    a, b = (10, 10, 60, 60), (12, 12, 62, 62)
    s = score_image("a cat behind a dog", det(("cat", a, 0.8), ("dog", b, 0.6)), depth_map((10, 10, 40, 40)))
    assert s == pytest.approx(0.35)


def test_depth_indexed_by_row_y_col_x():
    # tall thin box on the left vs wide flat box at the top: a transposed lookup would swap them
    a, b = (0, 20, 10, 100), (20, 0, 100, 10)
    d = depth_map((0, 20, 10, 100))  # only box a is near
    s = score_image("a cat in front of a dog", det(("cat", a, 1.0), ("dog", b, 1.0)), d)
    # IoU = 0 -> s_loc = 0 by definition, so check the order through a small overlap instead
    a2, b2 = (0, 0, 30, 100), (20, 0, 100, 30)
    d2 = np.zeros((100, 100), dtype=np.float32)
    d2[:, 0:20] = 1.0  # left strip (x < 20) is near
    s_ok = score_image("a cat in front of a dog", det(("cat", a2, 1.0), ("dog", b2, 1.0)), d2)
    s_bad = score_image("a cat behind a dog", det(("cat", a2, 1.0), ("dog", b2, 1.0)), d2)
    assert s == pytest.approx(0.5)
    assert s_ok > 0.5 and s_bad == pytest.approx(0.5)


def test_same_class_needs_two_detections():
    one = det(("microwave", (10, 10, 60, 60), 0.9))
    assert score_image("a microwave in front of a microwave", one, depth_map((0, 0, 1, 1))) == 0.0
    two = det(("microwave", (10, 10, 60, 60), 0.9), ("microwave", (15, 15, 65, 65), 0.7))
    assert score_image("a microwave in front of a microwave", two, depth_map((10, 10, 40, 40))) > 0.4


def test_missing_object_scores_zero_and_no_substring_match():
    # "cattle, cow" must not count as "cat"; "beer" must not count as "bee"
    d = det(("cattle, cow", (10, 10, 60, 60), 0.9), ("dog", (12, 12, 62, 62), 0.9))
    assert score_image("a cat in front of a dog", d, depth_map((10, 10, 40, 40))) == 0.0
    d = det(("beer", (10, 10, 60, 60), 0.9), ("dog", (12, 12, 62, 62), 0.9))
    assert score_image("a bee in front of a dog", d, depth_map((10, 10, 40, 40))) == 0.0


def test_synonyms_person_sofa_phone():
    d = det(("woman", (10, 10, 60, 60), 0.9), ("couch", (12, 12, 62, 62), 0.9),
            ("mobile phone, cell phone", (30, 30, 50, 50), 0.9))
    assert score_image("a girl in front of a sofa", d, depth_map((10, 10, 40, 40))) > 0.9
    assert score_image("a phone in front of a sofa", d, depth_map((30, 30, 50, 50))) > 0


def test_unscorable_wallet():
    d = det(("book", (10, 10, 60, 60), 0.9))
    assert score_image("a book hidden by a wallet", d, depth_map((0, 0, 1, 1))) is None
    assert score_image("a book hidden by a wallet", d, depth_map((0, 0, 1, 1)), unscorable="zero") == 0.0
