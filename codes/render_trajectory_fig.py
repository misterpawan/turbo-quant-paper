# Release change: create generated table/figure directories in a code-only checkout.
"""
Render the per-step KL trajectory figure from Phase 44 data.
Saves papers/{,neurips2026/}fig_trajectory_lens.pdf
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "runs" / "overnight" / "phase44_trajectory.jsonl"


def main() -> None:
    Path("papers/neurips2026").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in SRC.open() if l.strip()]
    by_task = defaultdict(list)
    for r in rows:
        task = "gsm8k" if r["prompt_id"].startswith("gsm8k") else "mmlu"
        by_task[task].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    # Left: per-step KL (one curve per prompt + bold mean).
    for ax_idx, task in enumerate(("mmlu", "gsm8k")):
        ax = axes[ax_idx]
        T = 64
        all_kls = np.full((len(by_task[task]), T), np.nan)
        for i, r in enumerate(by_task[task]):
            for s in r["trajectory"]:
                if s["step"] < T:
                    all_kls[i, s["step"]] = s["kl_bf16_to_quant"]
        # plot each curve faint, then mean
        for i in range(all_kls.shape[0]):
            ax.plot(range(T), all_kls[i], color="#888", alpha=0.3, linewidth=0.8)
        mean_kl = np.nanmean(all_kls, axis=0)
        ax.plot(range(T), mean_kl, color="#d62728", linewidth=2.0, label="mean (8 prompts)")
        # fraction-disagree as a secondary curve (right axis)
        ax2 = ax.twinx()
        agree_per_step = []
        for t in range(T):
            agrees = []
            for r in by_task[task]:
                for s in r["trajectory"]:
                    if s["step"] == t:
                        agrees.append(s["top1_agreement"])
            agree_per_step.append(1.0 - (np.mean(agrees) if agrees else 0))
        ax2.plot(range(T), agree_per_step, color="#1f77b4", linewidth=1.4, linestyle="--",
                 label="fraction-disagree")
        ax2.set_ylabel("Fraction-disagree", color="#1f77b4")
        ax2.tick_params(axis='y', labelcolor="#1f77b4")
        ax2.set_ylim(0, 1.0)
        ax.set_xlabel("Denoising step $t$")
        if ax_idx == 0:
            ax.set_ylabel("KL(BF16 || quant) at filled position")
        ax.set_title("MMLU" if task == "mmlu" else "GSM8K")
        ax.set_yscale("symlog", linthresh=0.001)
        ax.grid(True, alpha=0.25)
    fig.suptitle(r"Phase 44: per-step quantisation divergence at $T{=}64$ (BF16 vs TurboLDM (3+1))",
                  fontsize=10)
    fig.tight_layout()
    out_a = ROOT / "papers" / "fig_trajectory_lens.pdf"
    out_b = ROOT / "papers" / "neurips2026" / "fig_trajectory_lens.pdf"
    fig.savefig(out_a, bbox_inches="tight")
    fig.savefig(out_b, bbox_inches="tight")
    print(f"wrote {out_a} and {out_b}")


if __name__ == "__main__":
    main()
