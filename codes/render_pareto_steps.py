# Release change: create generated table/figure directories in a code-only checkout.
"""
Render a single-panel PDF figure showing accuracy vs wall-clock for
each (method, T) pair in Phase 31 on \textsc{gsm8k} and MMLU.

Output: papers/fig_steps_pareto.pdf and papers/neurips2026/fig_steps_pareto.pdf
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "overnight" / "phase31_steps_pareto.jsonl"


def main() -> None:
    Path("papers/neurips2026").mkdir(parents=True, exist_ok=True)
    rows = []
    with SRC.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=False)
    method_names = {
        "bf16": "BF16",
        "w4": "W4 uniform",
        "awq_4": "AWQ-4",
        "turboldm_3p1": "TurboLDM (3+1)",
        "turboldm_4p1": "TurboLDM (4+1)",
    }
    colors = {
        "bf16": "#000000",
        "w4": "#1f77b4",
        "awq_4": "#2ca02c",
        "turboldm_3p1": "#d62728",
        "turboldm_4p1": "#ff7f0e",
    }
    markers = {"bf16": "s", "w4": "o", "awq_4": "^", "turboldm_3p1": "D", "turboldm_4p1": "v"}
    for ax_idx, dataset in enumerate(("gsm8k", "mmlu")):
        ax = axes[ax_idx]
        per_method = defaultdict(list)
        for r in rows:
            if r.get("dataset") != dataset: continue
            T = r.get("T", 0); method = r.get("method", "?")
            wall = r.get("wall_sec", 0)
            score = r.get("score_value", 0) * 100
            per_method[method].append((T, wall, score))
        for method, vals in per_method.items():
            vals.sort()
            xs = [v[1] for v in vals]
            ys = [v[2] for v in vals]
            ts = [v[0] for v in vals]
            ax.plot(xs, ys, marker=markers.get(method, "o"),
                    color=colors.get(method, "#888"),
                    label=method_names.get(method, method), linewidth=1.4, markersize=6)
            for x, y, t in zip(xs, ys, ts):
                ax.annotate(f"T={t}", (x, y), fontsize=6, alpha=0.6,
                             xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Wall-clock (s) per query")
        ax.set_ylabel("Accuracy (\\%)" if ax_idx == 0 else "")
        ax.set_title(("\\textsc{gsm8k}" if dataset == "gsm8k" else "MMLU") + f" ($n{{=}}{{500 if dataset=='gsm8k' else 200}}$)".replace("{500 if dataset=='gsm8k' else 200}", str(100 if dataset == "gsm8k" else 200)))
        ax.grid(True, alpha=0.3)
        if ax_idx == 1:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Accuracy vs wall-clock at $T \\in \\{8, 16, 32, 64, 128\\}$", fontsize=10)
    fig.tight_layout()
    out_a = ROOT / "papers" / "fig_steps_pareto.pdf"
    out_b = ROOT / "papers" / "neurips2026" / "fig_steps_pareto.pdf"
    fig.savefig(out_a, bbox_inches="tight")
    fig.savefig(out_b, bbox_inches="tight")
    print(f"wrote {out_a} and {out_b}")


if __name__ == "__main__":
    main()
