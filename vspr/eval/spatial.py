"""UniDet-based 3D spatial score (Eq. 10), computed from saved detections and depth maps.

    Spatial(I, D) = alpha * (s_obj1 + s_obj2) / 2 + (1 - alpha) * s_loc,   alpha = 0.5
    s_loc = 1             if the depth order matches the relation and IoU > tau
          = IoU / tau     if the depth order matches and 0 < IoU <= tau
          = 0             otherwise                                          (tau = 0.5)
    and Spatial = 0 if either object is not detected.

Detections (UniDet, unified label space) and depth maps (DPT, inverse depth: larger =
closer) are produced by scripts/unidet_extract.py inside the T2I-CompBench UniDet_eval
environment. Scoring is done here so that the following defects of the earlier
implementations are fixed and testable:

* objects are parsed from the description with the dataset's fixed grammar
  ("a(n) X <relation> a(n) Y"), not with spaCy (spaCy mis-tags "key", and the old
  evaluate.py also cut the last character of the description);
* each object is matched to an explicit list of UniDet labels (exact matches), instead of
  substring matching (official script: "cat" matches "cattle, cow", "bee" matches "beer",
  "phone" matches "microphone"...) or the old manual rules (which mapped "phone" only to
  "telephone", "computer" only to "tablet computer" and could never match painting/bag);
* the two objects must be two different detections (13 descriptions name the same class
  twice, e.g. "a microwave in front of a microwave");
* the depth map is indexed as [y, x] (both earlier implementations used [x, y]);
* boxes and depth map are in the same pixel coordinates (no hard-coded 480 -> 1024 rescale).

Objects with no UniDet class at all (``wallet``) cannot be scored; by default such images
are excluded from the mean (``unscorable="exclude"``) and the count is reported.
"""

from __future__ import annotations

import re
from itertools import product

import numpy as np

RELATION_SIGN = {"in front of": +1, "behind": -1, "hidden by": -1}
_DESC_RE = re.compile(r"^an? (.+?) (in front of|behind|hidden by) an? (.+?)\.?$", re.IGNORECASE)

_PERSON = ("person", "man", "woman", "girl", "boy")

# Object word in T2I-CompBench 3d_spatial -> accepted UniDet labels (exact strings).
OBJECT_LABELS: dict[str, tuple[str, ...]] = {
    "airplane": ("airplane",),
    "backpack": ("backpack",),
    "bag": ("handbag", "luggage and bags", "plastic bag"),
    "balloon": ("balloon", "hotair balloon"),
    "bee": ("bee",),
    "bicycle": ("bicycle",),
    "bird": ("bird", "wild bird"),
    "book": ("book",),
    "bowl": ("bowl",),
    "boy": _PERSON, "girl": _PERSON, "man": _PERSON, "woman": _PERSON, "person": _PERSON,
    "butterfly": ("butterfly",),
    "candle": ("candle",),
    "car": ("car", "sports car", "race car"),
    "cat": ("cat",),
    "chair": ("chair",),
    "chicken": ("chicken", "turkey, chicken"),
    "clock": ("clock", "wall clock", "digital clock", "alarm clock"),
    "computer": ("laptop", "computer monitor", "tablet computer, tablet", "computer box"),
    "couch": ("couch", "loveseat, couch", "studio couch", "sofa bed"),
    "sofa": ("couch", "loveseat, couch", "studio couch", "sofa bed"),
    "cow": ("cattle, cow",),
    "cup": ("cup", "coffee cup", "mug", "measuring cup"),
    "desk": ("desk", "table, desk"),
    "dog": ("dog",),
    "fish": ("fish", "goldfish"),
    "frog": ("frog",),
    "giraffe": ("giraffe",),
    "horse": ("horse",),
    "key": ("key",),
    "lamp": ("lamp",),
    "microwave": ("microwave", "microwave oven"),
    # UniDet: "mouse2, mouse" = computer mouse (COCO / OID /m/020lf), "mouse1" = animal (OID /m/04rmv)
    "mouse": ("mouse1", "mouse2, mouse"),
    "painting": ("picture, frame",),
    "phone": ("telephone", "corded phone, telephone", "mobile phone, cell phone"),
    "pig": ("pig",),
    "plate": ("plate",),
    "rabbit": ("rabbit",),
    "refrigerator": ("refrigerator",),
    "sheep": ("sheep",),
    "suitcase": ("suitcase",),
    "table": ("table, desk", "dining table", "coffee table", "kitchen & dining room table"),
    "television": ("television", "tv"),
    "train": ("train",),
    "turtle": ("turtle", "sea turtle", "tortoise, tortoise/turtle"),
    "vase": ("vase",),
    "wallet": (),  # no wallet / purse class in the UniDet label space
}


def parse_description(desc: str) -> tuple[str, str, str]:
    m = _DESC_RE.match(desc.strip())
    if not m:
        raise ValueError(f"Not a 3D-spatial description: {desc!r}")
    return m.group(1).lower(), m.group(2).lower(), m.group(3).lower()


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mean_depth(depth: np.ndarray, box) -> float:
    h, w = depth.shape
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, max(x0 + 1, int(np.ceil(box[2])))), min(h, max(y0 + 1, int(np.ceil(box[3]))))
    return float(depth[y0:y1, x0:x1].mean())  # rows = y, columns = x


def select_pair(det: dict, labels1, labels2):
    """Highest s1 + s2 over pairs of *different* detections; None if impossible."""
    names, scores = det["labels"], det["scores"]
    c1 = [i for i, n in enumerate(names) if n in labels1]
    c2 = [j for j, n in enumerate(names) if n in labels2]
    pairs = [(i, j) for i, j in product(c1, c2) if i != j]
    if not pairs:
        return None
    return max(pairs, key=lambda p: (scores[p[0]] + scores[p[1]], -p[0], -p[1]))


def score_image(description: str, det: dict, depth: np.ndarray, alpha: float = 0.5, tau: float = 0.5,
                unscorable: str = "exclude") -> float | None:
    obj1, rel, obj2 = parse_description(description)
    l1, l2 = OBJECT_LABELS.get(obj1), OBJECT_LABELS.get(obj2)
    if l1 is None or l2 is None:
        raise KeyError(f"No label mapping for {obj1!r} or {obj2!r}; add it to OBJECT_LABELS")
    if not l1 or not l2:
        return None if unscorable == "exclude" else 0.0
    pair = select_pair(det, l1, l2)
    if pair is None:
        return 0.0
    i, j = pair
    b1, b2 = det["boxes"][i], det["boxes"][j]
    if depth.shape != (det["height"], det["width"]):
        raise ValueError(f"depth map {depth.shape} does not match image {(det['height'], det['width'])}")
    iou = box_iou(b1, b2)
    diff = mean_depth(depth, b1) - mean_depth(depth, b2)  # > 0: obj1 nearer
    order_ok = diff * RELATION_SIGN[rel] > 0
    if order_ok and iou > tau:
        s_loc = 1.0
    elif order_ok and iou > 0:
        s_loc = iou / tau
    else:
        s_loc = 0.0
    return alpha * (det["scores"][i] + det["scores"][j]) / 2 + (1 - alpha) * s_loc
