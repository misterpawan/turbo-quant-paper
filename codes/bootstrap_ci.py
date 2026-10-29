# Release change: make the generated caption valid for either phase 28 or phase 34 inputs.
"""
Paired-bootstrap 95% CI computation over per-example correctness arrays
stored in Phase 28's JSONL log.

Loads ``runs/overnight/phase28_main_ci.jsonl`` and computes, per
(method, dataset):

  * mean accuracy across seeds
  * seed-stdev
  * paired bootstrap 95% CI against BF16 on the matched prompt IDs
    (same seed, same dataset)
  * same against the nearest storage-matched baseline

Usage:
  python3 codes/bootstrap_ci.py   \\
    --in runs/overnight/phase28_main_ci.jsonl   \\
    --out papers/gen_table_main_ci.tex
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

BASELINE = "bf16"


def load_rows(path: Path) -> list:
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def per_example_from_extra(row: dict) -> list:
    ex = row.get("extra", {})
    if isinstance(ex, str):
        try:
            ex = json.loads(ex)
        except Exception:
            ex = {}
    return list(ex.get("correct_per_example", []))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, alpha: float = 0.05,
                      rng: np.random.Generator | None = None) -> tuple:
    if len(a) != len(b) or len(a) == 0:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(0)
    diff = a - b
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="runs/overnight/phase28_main_ci.jsonl")
    ap.add_argument("--out", default="papers/gen_table_main_ci.tex")
    args = ap.parse_args()
    rows = load_rows(Path(args.inp))
    if not rows:
        print("no rows found")
        return
    # Group per (method, dataset); collect scores and per-example arrays.
    by_method_dataset = defaultdict(list)
    for r in rows:
        method = r.get("method") or r.get("baseline", "").split("_s")[0]
        dataset = r.get("dataset", "?")
        seed = r.get("seed", -1)
        score = float(r.get("score_value", 0))
        pex = per_example_from_extra(r)
        bits = float(r.get("avg_bits_per_param", 0))
        storage = float(r.get("param_storage_gb", 0))
        by_method_dataset[(method, dataset)].append(
            dict(seed=seed, score=score, pex=pex, bits=bits, storage=storage))
    # Summarise.
    methods = sorted({k[0] for k in by_method_dataset})
    datasets = sorted({k[1] for k in by_method_dataset})
    summary = {}
    for m in methods:
        for d in datasets:
            entries = by_method_dataset.get((m, d), [])
            if not entries:
                continue
            scores = [e["score"] for e in entries]
            mean_s = mean(scores)
            sd_s = stdev(scores) if len(scores) > 1 else 0.0
            # bits/storage are deterministic in most rows; take median.
            bits = float(np.median([e["bits"] for e in entries]))
            storage = float(np.median([e["storage"] for e in entries]))
            summary[(m, d)] = dict(mean=mean_s, sd=sd_s, bits=bits, storage=storage,
                                   entries=entries)
    # Paired bootstrap CI against BF16 per dataset, per seed.
    rng = np.random.default_rng(0)
    bf16_by_seed_dataset = {}
    for e in by_method_dataset.get((BASELINE, ""), []):  # no-op
        pass
    for (m, d), entries in by_method_dataset.items():
        if m != BASELINE:
            continue
        for e in entries:
            bf16_by_seed_dataset[(e["seed"], d)] = e["pex"]
    for (m, d), info in summary.items():
        if m == BASELINE:
            info["ci_vs_bf16"] = (0.0, 0.0)
            continue
        diffs = []
        for e in info["entries"]:
            ref = bf16_by_seed_dataset.get((e["seed"], d))
            if not ref or not e["pex"] or len(ref) != len(e["pex"]):
                continue
            a = np.array(e["pex"], dtype=float)
            b = np.array(ref, dtype=float)
            diffs.append(a - b)
        if diffs:
            pooled = np.concatenate(diffs)
            lo = float(np.quantile(pooled, 0.025))
            hi = float(np.quantile(pooled, 0.975))
            # proper bootstrap over the pooled diff vector
            n = 2000
            idx = rng.integers(0, len(pooled), size=(n, len(pooled)))
            boot = pooled[idx].mean(axis=1)
            ci_lo = float(np.quantile(boot, 0.025))
            ci_hi = float(np.quantile(boot, 0.975))
            info["ci_vs_bf16"] = (ci_lo * 100, ci_hi * 100)   # pp
        else:
            info["ci_vs_bf16"] = (float("nan"), float("nan"))
    # Render LaTeX
    lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Main quality--storage trade-off on LLaDA-8B-Instruct \\emph{with 3 seeds and paired bootstrap 95\\% CIs}. Each cell is the across-seed mean accuracy $\\pm$ the seed standard deviation; the trailing [lo, hi] is the paired bootstrap 95\\% CI on the accuracy gap to BF16 in percentage points (negative = method is worse than BF16). A gap whose 95\\% CI straddles zero cannot be distinguished from BF16 at the sample sizes recorded in the input rows. Source: \\texttt{the input JSONL passed with --in}.}",
        "\\label{tab:main-ci}",
    ]
    colspec = "l" + ("c" * (2 + 2 * len(datasets)))
    lines.append("\\begin{tabular}{" + colspec + "}")
    lines.append("\\toprule")
    header = ["Method", "Bits", "Total GiB"]
    for d in datasets:
        header.append(d.replace("_", "\\_"))
        header.append("CI vs BF16 (pp)")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    method_order = ["bf16", "w3", "w4", "gptq_diag_4", "awq_4", "qdiff4", "turboldm_3p1", "turboldm_4p1"]
    for m in method_order:
        row_vals = [m.replace("_", "\\_")]
        # Use any dataset to get bits/storage (they're method-level).
        bits_storage_entry = None
        for d in datasets:
            s = summary.get((m, d))
            if s is not None:
                bits_storage_entry = s; break
        if bits_storage_entry is None:
            continue
        row_vals.append(f"{bits_storage_entry['bits']:.2f}")
        row_vals.append(f"{bits_storage_entry['storage']:.2f}")
        for d in datasets:
            s = summary.get((m, d))
            if s is None:
                row_vals.append("--"); row_vals.append("--"); continue
            row_vals.append(f"{s['mean']*100:.1f}$\\pm${s['sd']*100:.1f}")
            ci = s.get("ci_vs_bf16", (float("nan"), float("nan")))
            if any(np.isnan(x) for x in ci):
                row_vals.append("--")
            elif m == BASELINE:
                row_vals.append("(ref)")
            else:
                row_vals.append(f"[{ci[0]:.1f}, {ci[1]:.1f}]")
        lines.append(" & ".join(row_vals) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
