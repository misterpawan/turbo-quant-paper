"""
Phase 57: weight-fingerprint x sensitivity correlation (CPU only).

Phase 45 produced an 18 264-row CSV of per-(model, layer) weight
statistics: max_over_rms, p999_over_rms, stable_rank,
mean_abs_over_rms, frac_above_5rms. Phases 30 and 48 produced
per-(model, family) accuracy under per-family isolated quantization.

This script computes whether the static weight fingerprint actually
predicts the dynamic quantization fragility at the family level. We
aggregate weight stats by (model, family) and correlate them against
the BF16-relative accuracy drop on each task. A strong negative
correlation between max_over_rms and accuracy-drop is the textbook
"outlier-causes-fragility" claim.

Output: runs/overnight/phase57_fingerprint_corr.csv with one row per
(model, weight-stat, dataset, correlation kind).

CPU only -- no GPU contention with other phases.
"""
from __future__ import annotations

import csv, json, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "overnight"

WEIGHT_STATS = ["max_over_rms", "p999_over_rms", "stable_rank",
                "mean_abs_over_rms", "frac_above_5rms"]


def load_fingerprint() -> dict:
    """Returns {(model, family) -> {stat -> mean across layers}}."""
    path = RUNS / "phase45_arch_fingerprint.csv"
    bucket: dict = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["family"])
            for stat in WEIGHT_STATS:
                v = row.get(stat)
                try:
                    bucket[key][stat].append(float(v))
                except (TypeError, ValueError):
                    continue
    return {k: {s: mean(v) for s, v in d.items() if v} for k, d in bucket.items()}


def load_modsens(path: Path, model_key: str) -> dict:
    """Returns {(family, dataset) -> mean score across rows}."""
    if not path.exists():
        return {}
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    bucket: dict = defaultdict(list)
    for r in rows:
        if r.get("model") != model_key:
            continue
        fam = r.get("family")
        ds = r.get("dataset")
        s = r.get("score_value")
        if fam is None or ds is None or s is None:
            continue
        bucket[(fam, ds)].append(float(s))
    return {k: mean(v) for k, v in bucket.items() if v}


def baseline_score(rows: list, dataset: str) -> float | None:
    """Best BF16 reference for the given dataset across all the per-family rows."""
    bf16_scores = [float(r.get("score_value")) for r in rows
                   if r.get("dataset") == dataset and "bf16" in str(r.get("baseline", "")).lower()]
    if bf16_scores:
        return mean(bf16_scores)
    # Fall back: highest seen score on the dataset.
    seen = [float(r.get("score_value")) for r in rows
            if r.get("dataset") == dataset and r.get("score_value") is not None]
    return max(seen) if seen else None


def spearman(xs: list, ys: list) -> float:
    """Tiny-dependency Spearman rho. Returns nan if degenerate."""
    n = len(xs)
    if n < 3:
        return float("nan")

    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return float("nan")
    return num / (dx * dy) ** 0.5


def correlate_for_model(model_key: str, modsens_path: Path, fingerprint: dict) -> list:
    """Returns one row per (stat, dataset)."""
    out_rows = []
    if not modsens_path.exists():
        return out_rows
    with modsens_path.open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if r.get("model") == model_key]
    if not rows:
        return out_rows
    fam_ds_score = load_modsens(modsens_path, model_key)
    datasets = sorted({d for (_, d) in fam_ds_score})
    for dataset in datasets:
        bf = baseline_score(rows, dataset)
        if bf is None:
            continue
        families = sorted({f for (f, d) in fam_ds_score if d == dataset})
        # For each weight stat, build (stat_value, accuracy_drop) per family.
        for stat in WEIGHT_STATS:
            xs, ys = [], []
            for fam in families:
                fp = fingerprint.get((model_key, fam))
                if not fp or stat not in fp:
                    continue
                drop = bf - fam_ds_score[(fam, dataset)]
                xs.append(fp[stat]); ys.append(drop)
            if len(xs) < 3:
                continue
            rho = spearman(xs, ys)
            out_rows.append({
                "model": model_key, "dataset": dataset, "stat": stat,
                "n_families": len(xs), "spearman_rho": rho,
                "ref_bf16_score": bf,
                "families": ",".join(families[:len(xs)]),
            })
    return out_rows


def main() -> None:
    fingerprint = load_fingerprint()
    print(f"loaded fingerprint: {len(fingerprint)} (model, family) cells", file=sys.stderr)
    rows = []
    rows.extend(correlate_for_model(
        "llada_8b_instruct",
        RUNS / "phase30_modsens_multitask.jsonl", fingerprint))
    rows.extend(correlate_for_model(
        "dream_7b_instruct",
        RUNS / "phase48_dream_modsens.jsonl", fingerprint))
    out = RUNS / "phase57_fingerprint_corr.csv"
    if not rows:
        print("no rows produced; check input paths", file=sys.stderr)
        return
    fields = ["model", "dataset", "stat", "n_families", "spearman_rho",
              "ref_bf16_score", "families"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Phase 57 done; wrote {out} ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
