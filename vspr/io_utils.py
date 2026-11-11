"""Small file helpers shared by the pipeline, baselines and evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def load_descriptions(path: str | os.PathLike) -> list[str]:
    """Read one description per line, dropping blank lines and surrounding whitespace."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def slugify(text: str, max_len: int = 120) -> str:
    """File-system-safe name for a description (Windows-safe as well)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return slug[:max_len] or "empty"


def stable_seed(base_seed: int, text: str) -> int:
    """Deterministic 32-bit seed derived from the description.

    The same description always gets the same noise, in every round and for every
    method, so differences between methods/rounds come from the prompt only.
    """
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def read_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    """Atomic JSON write so an interrupted run never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
