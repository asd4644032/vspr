"""Collect metrics.json files into the paper's tables (Markdown and LaTeX rows).

    python scripts/make_tables.py --eval-root outputs/eval --out outputs/tables.md

Table 1: all methods per backend.  Table 3: VSPR iterations vs. the no-alignment
baseline, with differences computed before rounding (as in the paper).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vspr.eval import ORDER  # noqa: E402
from vspr.io_utils import read_json  # noqa: E402

PREFERRED = ["none", "gpt4", "beautiful", "magic", "sspo", "promptist", "pag"]


def fmt(v) -> str:
    return "–" if v is None else f"{v:.2f}"


def collect(backend_dir: Path) -> dict[str, dict]:
    return {d.name: read_json(d / "metrics.json") for d in sorted(backend_dir.iterdir())
            if (d / "metrics.json").exists()}


def ordered(names: list[str]) -> list[str]:
    head = [n for n in PREFERRED if n in names]
    return head + sorted(n for n in names if n not in head)


def main_table(backend: str, rows: dict[str, dict]) -> list[str]:
    names = [n for n in ordered(list(rows)) if "_iter" not in n]
    best = {k: max((rows[n].get(k) for n in names if rows[n].get(k) is not None), default=None) for k in ORDER}
    out = [f"### {backend}", "", "| Method | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|"]
    tex = []
    for n in names:
        md, tx = [], []
        for k in ORDER:
            v = rows[n].get(k)
            is_best = v is not None and v == best[k]
            md.append(f"**{fmt(v)}**" if is_best else fmt(v))
            tx.append(f"\\textbf{{{fmt(v)}}}" if is_best else fmt(v))
        out.append(f"| {n} | " + " | ".join(md) + " |")
        tex.append(f"{n} & " + " & ".join(tx) + r" \\")
    return out + ["", "```latex", *tex, "```", ""]


def iteration_table(backend: str, rows: dict[str, dict]) -> list[str]:
    base = rows.get("none")
    out = []
    for full in [n for n in rows if n.startswith("vspr_") and "_iter" not in n]:
        iters = sorted([n for n in rows if n.startswith(full + "_iter")],
                       key=lambda s: int(s.rsplit("iter", 1)[1]))
        if not iters or not base:
            continue
        out += [f"### {backend}: {full} by iteration", "",
                "| Method | " + " | ".join(ORDER) + " |", "|---" * (len(ORDER) + 1) + "|",
                "| baseline (none) | " + " | ".join(fmt(base.get(k)) for k in ORDER) + " |"]
        for n in iters + [full]:
            label = n.replace(full + "_", "") if n != full else f"iter{len(iters) + 1}"
            cells = []
            for k in ORDER:
                v, b = rows[n].get(k), base.get(k)
                cells.append(fmt(v) if v is None or b is None else f"{v:.2f} ({v - b:+.2f})")
            out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    lines = ["# Results", ""]
    for backend_dir in sorted(Path(args.eval_root).iterdir()):
        if backend_dir.is_dir():
            rows = collect(backend_dir)
            if rows:
                lines += main_table(backend_dir.name, rows) + iteration_table(backend_dir.name, rows)
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
