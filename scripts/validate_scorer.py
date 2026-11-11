"""Check whether phi_sco picks better rounds than not selecting at all.

Two sub-commands (metric is the default):

DMS = p(yes | "Does this image show 'D'?") and OPS = p(A present) * p(B present), both from the
fixed questions of vspr/judge.py (qwen2.5vl:7b); phi_sco = a * DMS + (1 - a) * OPS. In the human
mode they keep their E1 names: overall = DMS, pApB = OPS.


1. metric (default) -- phi_sco against the evaluation metrics of a pilot run:

    python scripts/run_vspr.py --set limit=100 run_name=sdxl_pilot --export-rounds
    python scripts/evaluate.py --eval-root outputs/eval/sdxl --metrics Spatial CS ITM \
        --methods vspr_pilot_round1 vspr_pilot_round2 vspr_pilot_round3 vspr_pilot_round4 ...
    python scripts/validate_scorer.py --eval-root outputs/eval/sdxl --method vspr_pilot

   For each metric it reports, per description (metric averaged over the B images of a round):
   Spearman rho between the selector and the metric; pairwise accuracy within a description;
   and the mean metric of round 1, the last round, the mean over rounds, the round chosen by the
   selector and the oracle best round. Selectors: s (phi_sco), DMS only and OPS only.

2. human -- validate every candidate phi_sco against the human labels (E1; stdlib only):

    python scripts/validate_scorer.py human --vqa-test ..\\vqa_test

   Candidate scores per image: pApB (OPS) and overall (DMS) from the v4 judge, and
   w*overall + (1-w)*pApB (w = a). The acc / q columns of E1 (old design, LLM questions and LLM
   judgment) came from the removed `fill` command; the pre-E5 script is kept in
   vqa_test/_vspr_legacy/scripts/validate_scorer.py.
   The round score is the mean over the B = 4 images. Evaluation metrics are reference only.
   Reports (tables also saved as CSV in <vqa_test>/scorer_validation/dms_ops/):
   * image level: AUC for the human "matches" label, bootstrap 95% CI resampling descriptions;
   * round level (main): each of the 30 pilot descriptions picks 1 of its 4 conditions by the
     score; mean human match rate of the picked condition vs random / oracle / always v4 /
     always round 1 (paired bootstrap CI); ties count as a uniform random pick among the tied;
   * choice of w: 2-fold split of the descriptions, repeated 5 times (w chosen on one half,
     reported on the other); plus w chosen on the 296 old-experiment images (image AUC);
   * the same per relation type (in front of / behind / hidden by).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vspr.eval import ORDER, SCALE  # noqa: E402
from vspr.io_utils import read_json  # noqa: E402


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


# ====================================================================== 1. metric (original)
def load(eval_root: Path, method: str):
    """{metric: {description: {round: (metric_mean, s, DMS, OPS)}}}"""
    rounds = sorted((d for d in eval_root.iterdir() if d.name.startswith(method + "_round")),
                    key=lambda d: int(d.name.rsplit("round", 1)[1]))
    if not rounds:
        sys.exit(f"No {method}_round<i> folders under {eval_root}; run run_vspr.py --export-rounds")
    table: dict = defaultdict(lambda: defaultdict(dict))
    for d in rounds:
        i = int(d.name.rsplit("round", 1)[1])
        manifest = read_json(d / "manifest.json")
        for metric in ORDER:
            f = d / "scores" / f"{metric}.json"
            if not f.exists():
                continue
            scores = {int(k): v for k, v in read_json(f).items()}
            per_desc = defaultdict(list)
            meta = {}
            for row in manifest:
                if scores.get(row["question_id"]) is not None:
                    per_desc[row["description"]].append(scores[row["question_id"]])
                    meta[row["description"]] = (row["score"], row.get("dms"), row.get("ops"))
            for desc, vals in per_desc.items():
                table[metric][desc][i] = (SCALE[metric] * statistics.fmean(vals), *meta[desc])
    return table, len(rounds)


def report(metric: str, data: dict, n_rounds: int) -> None:
    descs = [d for d, r in data.items() if len(r) == n_rounds]
    if not descs:
        return
    print(f"\n== {metric}  ({len(descs)} descriptions x {n_rounds} rounds)")
    m = lambda d, i: data[d][i][0]  # noqa: E731
    base = {
        "round 1 (no alignment)": statistics.fmean(m(d, 1) for d in descs),
        "last round": statistics.fmean(m(d, n_rounds) for d in descs),
        "mean over rounds": statistics.fmean(statistics.fmean(m(d, i) for i in data[d]) for d in descs),
        "oracle best round": statistics.fmean(max(m(d, i) for i in data[d]) for d in descs),
    }
    for k, v in base.items():
        print(f"  {k:28s} {v:8.2f}")
    for name, col in (("s (phi_sco)", 1), ("DMS only", 2), ("OPS only", 3)):
        if any(data[d][i][col] is None for d in descs for i in data[d]):
            continue
        sel = statistics.fmean(
            m(d, max(data[d], key=lambda i: (data[d][i][col], -i))) for d in descs)
        xs = [data[d][i][col] for d in descs for i in data[d]]
        ys = [m(d, i) for d in descs for i in data[d]]
        agree = total = 0
        for d in descs:
            rs = sorted(data[d])
            for a in rs:
                for b in rs:
                    if a < b and data[d][a][col] != data[d][b][col] and m(d, a) != m(d, b):
                        total += 1
                        agree += (data[d][a][col] > data[d][b][col]) == (m(d, a) > m(d, b))
        pa = agree / total if total else float("nan")
        print(f"  selected by {name:16s} {sel:8.2f}   rho={spearman(xs, ys):+.3f}   pairwise acc={pa:.3f}")


def cmd_metric(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--method", required=True, help="e.g. vspr_llama3_1_8b (without _round<i>)")
    args = ap.parse_args(argv)
    table, n = load(Path(args.eval_root), args.method)
    for metric in ORDER:
        if metric in table:
            report(metric, table[metric], n)


# ====================================================================== shared (E1)
CONDS = ("round1", "novqa", "repo", "v4")
BASE_SCORES = ("pApB", "overall")
PAIRS = (("overall", "pApB"),)
WS = (0.0, 0.25, 0.5, 0.75, 1.0)
REF_METRICS = ("CS", "ITC", "ITM", "VQA", "Spatial", "AS")
RELS = ("in front of", "behind", "hidden by")
OLD_DATA = r"F:\ComfyUI-master\script_examples\project\output\3d_spatial"
DESC_RE = re.compile(r"^an? (.+?) (in front of|behind|hidden by) an? (.+)$")


def parse_desc(d: str) -> tuple[str, str, str]:
    m = DESC_RE.match(d.strip())
    if not m:
        raise ValueError(f"cannot parse {d!r}")
    return m.group(1), m.group(2), m.group(3)


def vt_paths(vt: Path) -> dict:
    out = vt / "scorer_validation" / "dms_ops"  # the E1 tables (with acc / q) stay one level up
    return {"out": out, "labels": out / "pilot_eval_labels.json",
            "state": vt / "pilot" / "sdxl" / "state.json", "mapping": vt / "pilot" / "sdxl" / "pilot_mapping.json",
            "descs": vt / "pilot" / "sdxl" / "descriptions.json", "eval": vt / "pilot" / "eval" / "sdxl",
            "probe": vt / "probe_cache" / "qwen2.5vl_7b__v4.jsonl",
            "probe_v3": vt / "probe_cache" / "qwen2.5vl_7b__v3.jsonl",
            "keys": [vt / "human_keys.json", vt / "human_keys_random.json"]}


def combo_name(x: str, y: str, w: float) -> str:
    return f"{w:g}*{x}+{1 - w:g}*{y}"


# ====================================================================== 2. human (E1)
def p_present(count: dict, need: int = 1) -> float:
    return sum(v for k, v in count.items() if int(k) >= need)


def presence(desc: str, judge: dict) -> float:
    a, _, b = parse_desc(desc)
    if a == b:
        p2 = p_present(judge["count_a"], 2)
        return p2 * p2
    return p_present(judge["count_a"]) * p_present(judge["count_b"])


def human_cat(h: dict) -> str | None:  # same rule as vqa_test/vqa_v4.py
    if not h or "rel" not in h:
        return None
    if h.get("pa") == "no" or h.get("pb") == "no":
        return "missing"
    if h.get("sep") == "no" and h["rel"] in ("na", "neither"):
        return "fused/picture"
    if h["rel"] == "na":
        return "fused/picture"
    return {"want": "ok", "reverse": "reversed", "neither": "neither"}[h["rel"]]


def load_rows(p: dict) -> tuple[list[dict], list[str]]:
    """One row per human-labelled image: set, group (description), cond, rel, y, scores, metrics."""
    rows, notes = [], []
    # --- pilot blind rating
    state, mapping = read_json(p["state"]), read_json(p["mapping"])
    labels = read_json(p["labels"])["labels"]
    descs100 = read_json(p["descs"])
    metrics = {}
    for c in CONDS:
        for m in REF_METRICS:
            f = p["eval"] / c / "scores" / f"{m}.json"
            if f.exists():
                metrics[(c, m)] = {int(k): v for k, v in read_json(f).items()}
    for d, order in mapping.items():
        it = state[d]
        for bi, c in enumerate(order):
            judges = it["r1_judge"] if c == "round1" else it["r2_judge"][c]
            for b in range(4):
                lab = labels.get(f"{d}|{bi}|{b}")
                if lab not in ("ok", "bad"):
                    continue
                s = {"pApB": presence(d, judges[b]), "overall": judges[b]["overall_p"]}
                qid = descs100.index(d) * 4 + b
                ref = {m: metrics[(c, m)][qid] for (cc, m) in metrics if cc == c and qid in metrics[(c, m)]}
                rows.append({"set": "pilot", "group": d, "cond": c, "b": b, "rel": parse_desc(d)[1],
                             "y": int(lab == "ok"), "s": s, "ref": ref})
    # --- old-experiment human keys (one batch of 4 per description, prompt = D)
    probe = {}
    if p["probe"].exists():
        for line in p["probe"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                probe[(r["t2i"], r["desc"], r["b"])] = r
    probe_v3 = {}
    if p["probe_v3"].exists():
        for line in p["probe_v3"].read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "pres_a_p" in r and "pres_b_p" in r and "overall_p" in r:
                    probe_v3[(r["t2i"], r["desc"], r["b"])] = r
    for f in p["keys"]:
        if not f.exists():
            continue
        set_name = "random200" if "random" in f.name else "keys96"
        for k, h in read_json(f)["keys"].items():
            t2i, d, b = k.split("|")
            b = int(b)
            cat = human_cat(h)
            if cat is None:
                continue
            s = {}
            r = probe.get((t2i, d, b))
            if r:
                s["pApB"], s["overall"] = presence(d, r), r["overall_p"]
            elif (t2i, d, b) in probe_v3:  # human_keys.json (audit set) was only probed by v3
                r = probe_v3[(t2i, d, b)]
                s["pApB"], s["overall"] = r["pres_a_p"] * r["pres_b_p"], r["overall_p"]
            rows.append({"set": set_name, "group": f"{t2i}|{d}", "cond": "round1", "b": b - 1,
                         "rel": parse_desc(d)[1], "y": int(cat == "ok"), "s": s, "ref": {}})
    for name in BASE_SCORES:
        for st in ("pilot", "random200", "keys96"):
            sub = [r for r in rows if r["set"] == st]
            miss = sum(name not in r["s"] for r in sub)
            if sub and miss:
                notes.append(f"{st}: '{name}' missing for {miss}/{len(sub)} images")
    return rows, notes


def add_combos(rows: list[dict]) -> list[str]:
    """Adds w*X+(1-w)*Y image scores (w in 0.25/0.5/0.75; w=1 / w=0 are the single scores)."""
    names = list(BASE_SCORES)
    for x, y in PAIRS:
        for w in WS[1:-1]:
            n = combo_name(x, y, w)
            names.append(n)
            for r in rows:
                if x in r["s"] and y in r["s"]:
                    r["s"][n] = w * r["s"][x] + (1 - w) * r["s"][y]
    return names


def score_of(name: str, x: str, y: str, w: float) -> str:
    return x if w == 1 else y if w == 0 else combo_name(x, y, w)


def auc(s: list[float], y: list[int]) -> float:
    pos = sum(y)
    if pos == 0 or pos == len(y):
        return float("nan")
    r = ranks(s)
    return (sum(ri for ri, yi in zip(r, y) if yi) - pos * (pos - 1) / 2) / (pos * (len(y) - pos))


def cluster_boot(groups: dict, stat, n_boot: int, seed: int = 0) -> tuple[float, float, float]:
    """stat(list of group values) on the full data and its percentile 95% CI, resampling groups."""
    keys = sorted(groups)
    full = stat([groups[k] for k in keys])
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        v = stat([groups[rng.choice(keys)] for _ in keys])
        if v == v:
            vals.append(v)
    vals.sort()
    if not vals:
        return full, float("nan"), float("nan")
    return full, vals[int(0.025 * len(vals))], vals[min(len(vals) - 1, int(0.975 * len(vals)))]


def image_auc(rows: list[dict], name: str, n_boot: int, ref: bool = False) -> tuple:
    by = defaultdict(list)
    for r in rows:
        src = r["ref"] if ref else r["s"]
        if name in src:
            by[r["group"]].append((src[name], r["y"]))
    if not by:
        return None

    def stat(gs):
        pairs = [p for g in gs for p in g]
        return auc([p[0] for p in pairs], [p[1] for p in pairs])

    n = sum(len(v) for v in by.values())
    return (*cluster_boot(by, stat, n_boot), n, sum(y for v in by.values() for _, y in v))


def round_table(rows: list[dict]) -> dict:
    """{description: {cond: {"h": human rate, "rel": ..., scores: round score}}} for the pilot."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["set"] == "pilot":
            acc[r["group"]][r["cond"]].append(r)
    out = {}
    for d, cs in acc.items():
        out[d] = {}
        for c, rs in cs.items():
            e = {"h": statistics.fmean(r["y"] for r in rs), "rel": rs[0]["rel"]}
            for name in rs[0]["s"]:
                if all(name in r["s"] for r in rs):
                    e[name] = statistics.fmean(r["s"][name] for r in rs)
            out[d][c] = e
    return out


def picked(cands: dict, name: str) -> float | None:
    """Human rate of the condition picked by score `name` (expected value over ties)."""
    if any(name not in e for e in cands.values()):
        return None
    best = max(e[name] for e in cands.values())
    tied = [e["h"] for e in cands.values() if abs(e[name] - best) < 1e-12]
    return statistics.fmean(tied)


def paired(descs: list[str], f, n_boot: int) -> tuple[float, float, float]:
    vals = {d: f(d) for d in descs}
    vals = {d: v for d, v in vals.items() if v is not None}
    return cluster_boot(vals, lambda xs: statistics.fmean(xs) if xs else float("nan"), n_boot, seed=1)


def fmt_ci(t, scale=100.0, sign=False) -> str:
    v, lo, hi = t
    s = "+" if sign else ""
    return f"{v * scale:{s}.1f} [{lo * scale:{s}.1f}, {hi * scale:{s}.1f}]"


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def cmd_human(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="validate phi_sco candidates against human labels")
    ap.add_argument("--vqa-test", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--subset", nargs="*", default=None, help="restrict the pilot to these conditions")
    args = ap.parse_args(argv)
    p = vt_paths(Path(args.vqa_test))
    p["out"].mkdir(parents=True, exist_ok=True)
    rows, notes = load_rows(p)
    names = add_combos(rows)
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s)
        lines.append(s)

    out("# E1 phi_sco validation against human labels")
    for n in notes:
        out(f"NOTE {n}")
    sets = Counter(r["set"] for r in rows)
    out(f"images: {dict(sets)}; positives: " +
        ", ".join(f"{s} {sum(r['y'] for r in rows if r['set'] == s)}" for s in sets))

    # ---------------- 1. image-level AUC
    out("\n## 1. Image-level AUC for 'matches the description' (95% CI, descriptions resampled)")
    datasets = {"pilot": [r for r in rows if r["set"] == "pilot"],
                "old296": [r for r in rows if r["set"] in ("random200", "keys96")],
                "random200": [r for r in rows if r["set"] == "random200"],
                "keys96": [r for r in rows if r["set"] == "keys96"]}
    datasets["all"] = datasets["pilot"] + datasets["old296"]
    auc_rows = []
    out(f"{'score':28s}" + "".join(f"{k:>26s}" for k in datasets))
    for name in names:
        line, any_ok = f"{name:28s}", False
        for k, rs in datasets.items():
            t = image_auc(rs, name, args.boot) if rs else None
            if t is None or t[0] != t[0]:
                line += f"{'-':>26s}"
                continue
            any_ok = True
            line += f"{t[0]:9.3f} [{t[1]:.3f}, {t[2]:.3f}] n={t[3]:<3d}"
            auc_rows.append([name, k, round(t[0], 4), round(t[1], 4), round(t[2], 4), t[3], t[4]])
        if any_ok:
            out(line)
    out("reference metrics (pilot only; never used as selectors):")
    for m in REF_METRICS:
        t = image_auc(datasets["pilot"], m, args.boot, ref=True)
        if t and t[0] == t[0]:
            out(f"  {m:26s}{t[0]:9.3f} [{t[1]:.3f}, {t[2]:.3f}] n={t[3]}")
            auc_rows.append([f"ref:{m}", "pilot", round(t[0], 4), round(t[1], 4), round(t[2], 4), t[3], t[4]])
    write_csv(p["out"] / "image_auc.csv", ["score", "set", "auc", "lo", "hi", "n", "positives"], auc_rows)

    # ---------------- 2. round-level selection
    rt = round_table(rows)
    if args.subset:
        rt = {d: {c: e for c, e in cs.items() if c in args.subset} for d, cs in rt.items()}
    descs = sorted(d for d, cs in rt.items() if len(cs) >= 2)
    out(f"\n## 2. Round level: pick 1 of {'/'.join(args.subset or CONDS)} per description "
        f"({len(descs)} descriptions); human match rate x100")
    rand = lambda d: statistics.fmean(e["h"] for e in rt[d].values())  # noqa: E731
    orac = lambda d: max(e["h"] for e in rt[d].values())  # noqa: E731
    fixed = {c: (lambda d, c=c: rt[d][c]["h"] if c in rt[d] else None) for c in CONDS}
    out(f"  random pick            {fmt_ci(paired(descs, rand, args.boot))}")
    out(f"  oracle (human best)    {fmt_ci(paired(descs, orac, args.boot))}")
    for c in CONDS:
        if any(c in rt[d] for d in descs):
            out(f"  always {c:15s} {fmt_ci(paired(descs, fixed[c], args.boot))}")
    informative = [d for d in descs if len({e['h'] for e in rt[d].values()}) > 1]
    out(f"  descriptions where the choice matters (human rates differ): {len(informative)}/{len(descs)}")
    sel_rows = []
    out(f"\n  {'selector':28s}{'picked':>22s}{'- random':>24s}{'- always v4':>24s}{'- round1':>24s}{'regret vs oracle':>20s}")
    for name in names:
        if any(picked(rt[d], name) is None for d in descs):
            continue
        sel = paired(descs, lambda d: picked(rt[d], name), args.boot)
        dr = paired(descs, lambda d: picked(rt[d], name) - rand(d), args.boot)
        dv = paired(descs, lambda d: picked(rt[d], name) - rt[d]["v4"]["h"] if "v4" in rt[d] else None, args.boot)
        d1 = paired(descs, lambda d: picked(rt[d], name) - rt[d]["round1"]["h"] if "round1" in rt[d] else None,
                    args.boot)
        rg = statistics.fmean(orac(d) - picked(rt[d], name) for d in descs)
        out(f"  {name:28s}{fmt_ci(sel):>22s}{fmt_ci(dr, sign=True):>24s}{fmt_ci(dv, sign=True):>24s}"
            f"{fmt_ci(d1, sign=True):>24s}{rg * 100:20.1f}")
        sel_rows.append([name] + [round(x * 100, 2) for t in (sel, dr, dv, d1) for x in t] + [round(rg * 100, 2)])
    write_csv(p["out"] / "round_selection.csv",
              ["selector", "picked", "lo", "hi", "minus_random", "lo", "hi", "minus_v4", "lo", "hi",
               "minus_round1", "lo", "hi", "regret"], sel_rows)

    # best-of-2 = what a 2-round VSPR with that rewriter would return
    b2_rows = []
    out("\n  best-of-2 {round1, X} (a 2-round VSPR run with rewriter X); picked - always X, picked - round1")
    for c in ("v4", "repo", "novqa"):
        sub = {d: {k: rt[d][k] for k in ("round1", c) if k in rt[d]} for d in descs}
        ds = [d for d in descs if len(sub[d]) == 2]
        line = f"  X={c:6s}"
        for name in BASE_SCORES + (combo_name("overall", "pApB", 0.5),):
            if not ds or any(picked(sub[d], name) is None for d in ds):
                continue
            dx = paired(ds, lambda d: picked(sub[d], name) - sub[d][c]["h"], args.boot)
            d1 = paired(ds, lambda d: picked(sub[d], name) - sub[d]["round1"]["h"], args.boot)
            out(f"{line} {name:26s} {fmt_ci(dx, sign=True):>22s} {fmt_ci(d1, sign=True):>22s}")
            b2_rows.append([c, name] + [round(x * 100, 2) for t in (dx, d1) for x in t])
    write_csv(p["out"] / "best_of_2.csv", ["X", "selector", "minus_X", "lo", "hi", "minus_round1", "lo", "hi"],
              b2_rows)

    # ---------------- 3. choice of w (2-fold x 5)
    out("\n## 3. Choice of w: 2-fold split of the descriptions, repeated 5 times "
        "(w picked on one half, picked - random reported on the other half, x100)")
    cv_rows = []

    def gain(ds, name):
        return statistics.fmean(picked(rt[d], name) - rand(d) for d in ds)

    def choose(ds, cand):  # best mean gain on ds; ties -> the earliest in cand (cand ordered by preference)
        vals = [(gain(ds, n), -i, n) for i, n in enumerate(cand)]
        return max(vals)[2]

    families = {f"{x}+{y}": [score_of("", x, y, w) for w in (0.5, 0.25, 0.75, 0.0, 1.0)] for x, y in PAIRS}
    families["any single or combination"] = [n for n in names]
    for fam, cand in families.items():
        if any(picked(rt[d], n) is None for d in descs for n in cand):
            continue
        tests, chosen = [], Counter()
        for rep in range(5):
            ds = descs[:]
            random.Random(100 + rep).shuffle(ds)
            halves = (ds[: len(ds) // 2], ds[len(ds) // 2:])
            for tr, te in (halves, halves[::-1]):
                n = choose(tr, cand)
                chosen[n] += 1
                tests.append(gain(te, n))
        m, sd = statistics.fmean(tests), statistics.stdev(tests)
        out(f"  {fam:28s} held-out gain {m * 100:+6.1f} (sd over 10 folds {sd * 100:4.1f}); chosen: " +
            ", ".join(f"{k} x{v}" for k, v in chosen.most_common()))
        cv_rows.append([fam, round(m * 100, 2), round(sd * 100, 2),
                        "; ".join(f"{k} x{v}" for k, v in chosen.most_common())])
    write_csv(p["out"] / "w_cv.csv", ["family", "heldout_gain_vs_random", "sd", "chosen"], cv_rows)

    # w chosen on the 296 old images (independent data), applied to the pilot rounds
    old = datasets["old296"]
    if old:
        out("\n  w chosen by image AUC on the 296 old-experiment images, applied to the pilot rounds:")
        for x, y in PAIRS:
            cand = [score_of("", x, y, w) for w in WS]
            aucs = {n: auc([r["s"][n] for r in old if n in r["s"]], [r["y"] for r in old if n in r["s"]])
                    for n in cand if sum(n in r["s"] for r in old) == len(old)}
            if len(aucs) < len(cand) or any(picked(rt[d], n) is None for d in descs for n in cand):
                continue
            best = max(aucs, key=aucs.get)
            out(f"    {x}+{y}: chosen {best} (AUC old {aucs[best]:.3f}) -> pilot picked - random "
                f"{fmt_ci(paired(descs, lambda d: picked(rt[d], best) - rand(d), args.boot), sign=True)}")

    # ---------------- 4. per relation type
    out("\n## 4. Per relation type")
    rel_rows = []
    for rel in RELS:
        ds = [d for d in descs if next(iter(rt[d].values()))["rel"] == rel]
        img = [r for r in datasets["all"] if r["rel"] == rel]
        out(f"\n  -- {rel}: {len(ds)} pilot descriptions, {len(img)} labelled images "
            f"({sum(r['y'] for r in img)} positive); random {statistics.fmean(rand(d) for d in ds) * 100:.1f}, "
            f"oracle {statistics.fmean(orac(d) for d in ds) * 100:.1f}, always v4 "
            f"{statistics.fmean(rt[d]['v4']['h'] for d in ds) * 100:.1f}")
        for name in names:
            t = image_auc(img, name, max(200, args.boot // 4))
            if t is None or t[0] != t[0] or any(picked(rt[d], name) is None for d in ds):
                continue
            pk = statistics.fmean(picked(rt[d], name) for d in ds)
            g = statistics.fmean(picked(rt[d], name) - rand(d) for d in ds)
            out(f"    {name:28s} AUC(all) {t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]   picked {pk * 100:5.1f}  "
                f"- random {g * 100:+5.1f}")
            rel_rows.append([rel, name, round(t[0], 4), round(t[1], 4), round(t[2], 4), round(pk * 100, 2),
                             round(g * 100, 2), len(ds)])
    write_csv(p["out"] / "by_relation.csv",
              ["relation", "score", "auc_all", "lo", "hi", "picked", "minus_random", "n_desc"], rel_rows)
    (p["out"] / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsaved tables and report.txt in {p['out']}")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] == "human":
        cmd_human(argv[1:])
    else:
        cmd_metric(argv[1:] if argv and argv[0] == "metric" else argv)


if __name__ == "__main__":
    main()
