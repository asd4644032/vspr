"""Evaluation metrics of Section 4.1 (CLIP, BLIP-ITC/ITM, BLIP-VQA, UniDet, aesthetic)."""

# Metric -> multiplier used in Tables 1-3 of the paper.
SCALE = {"CS": 1.0, "ITC": 100.0, "ITM": 1000.0, "VQA": 100.0, "Spatial": 100.0, "AS": 10.0}
ORDER = ["CS", "ITC", "ITM", "VQA", "Spatial", "AS"]
