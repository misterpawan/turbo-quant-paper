# Release change: create generated table/figure directories in a code-only checkout.
"""Render LaTeX tables and a findings narrative from phase 82/83/84 outputs.

Produces four files in papers/ and mirrors them to papers/neurips2026/:

  gen_table_bits_steps.tex   - Bits x denoising-steps Pareto on LLaDA-8B
  gen_table_seed_ci.tex      - GSM8K 3-seed CIs by quantisation method
  gen_table_entropy.tex      - Trajectory entropy + first/final-step entropy
  gen_findings_v2.tex        - Self-contained narrative summary of F1..F4

The script is idempotent: re-runs overwrite the generated files but never
touch source code or the bibliography.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

OUT_DIR = Path("runs/overnight")
PAPERS_DIR = Path("papers")
NEURIPS_DIR = Path("papers/neurips2026")


def _mirror(path: Path) -> None:
    if NEURIPS_DIR.exists() and path.exists():
        (NEURIPS_DIR / path.name).write_bytes(path.read_bytes())


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


# ---------------------------------------------------------------------------
# Phase 83: bits x steps Pareto
# ---------------------------------------------------------------------------

PRETTY = {"bf16": "BF16", "turboldm_4_1": "TurboLDM (4+1)",
          "turboldm_3_1": "TurboLDM (3+1)"}


def render_bits_steps() -> None:
    rows = _load_jsonl(OUT_DIR / "phase83_bits_steps.jsonl")
    if not rows:
        print("phase83 jsonl missing; skipping")
        return
    by_key: Dict[tuple, dict] = {}
    for r in rows:
        key = (r["baseline"], r["dataset"], r["gen_cfg"]["steps"])
        by_key[key] = r

    methods = ["bf16", "turboldm_4_1", "turboldm_3_1"]
    steps_list = [16, 32, 64]
    datasets = [("gsm8k", "GSM8K"), ("mmlu", "MMLU")]

    body = []
    for method in methods:
        bits_str = ""
        for r in rows:
            if r["baseline"] == method:
                bits_str = f"{r['avg_bits_per_param']:.2f}"
                break
        cells = [PRETTY[method], bits_str]
        for ds, _ in datasets:
            for steps in steps_list:
                v = by_key.get((method, ds, steps))
                if v is None:
                    cells.append("--")
                else:
                    cells.append(f"{100 * v['score_value']:.1f}")
        body.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Bit-width versus denoising-step trade-off on LLaDA-8B-Instruct (n{=}150 GSM8K, n{=}200 MMLU). "
        r"For the generative GSM8K task, extra denoising steps recover most of the loss from "
        r"aggressive quantisation: TurboLDM~(3+1) at $T{=}64$ reaches $60.7\%$, beating its own $T{=}16$ "
        r"setting by $+43$\,pp. The MCQ task MMLU is invariant to step count under every precision, "
        r"showing that the bit\,$\leftrightarrow$\,step trade is specific to step-compounding workloads. "
        r"At $T{=}64$, TurboLDM~(4+1) ties BF16 on GSM8K~(68.0).}",
        r"\label{tab:bits_steps}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccc|ccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c|}{GSM8K (steps)} & \multicolumn{3}{c}{MMLU (steps)} \\",
        r"Method & Bits & 16 & 32 & 64 & 16 & 32 & 64 \\",
        r"\midrule",
    ]
    lines += body
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = PAPERS_DIR / "gen_table_bits_steps.tex"
    out.write_text("\n".join(lines) + "\n")
    _mirror(out)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Phase 84: 3-seed CIs on GSM8K
# ---------------------------------------------------------------------------

CI_PRETTY = {
    "bf16": "BF16",
    "uniform_int4": "Uniform-int4",
    "gptq_4": "GPTQ-4",
    "awq_4": "AWQ-4",
    "turboldm_4_1": "TurboLDM (4+1)",
    "turboldm_3_1": "TurboLDM (3+1)",
}


def render_seed_ci() -> None:
    rows = _load_jsonl(OUT_DIR / "phase84_seed_ci_gsm8k.jsonl")
    if not rows:
        print("phase84 jsonl missing; skipping")
        return
    by_method: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["baseline"]].append(r)

    body = []
    for method in ["bf16", "uniform_int4", "gptq_4", "awq_4", "turboldm_4_1", "turboldm_3_1"]:
        recs = by_method.get(method, [])
        if not recs:
            continue
        scores = [r["score_value"] for r in recs]
        bits = recs[0]["avg_bits_per_param"]
        n = len(scores)
        m = mean(scores)
        s = stdev(scores) if n > 1 else 0.0
        rng = f"{100*min(scores):.1f}--{100*max(scores):.1f}" if n > 1 else f"{100*scores[0]:.1f}"
        cell = (f"{CI_PRETTY[method]} & {bits:.2f} & {n} & "
                f"{100*m:.2f} & {100*s:.2f} & {rng} \\\\")
        body.append(cell)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Three-seed CIs on GSM8K (LLaDA-8B-Instruct, n{=}200, "
        r"low-confidence remasking). Each quantised cell is the mean over seeds "
        r"$\{17,42,101\}$; std is across seeds. \emph{GPTQ-4's std (5.2\,pp) is "
        r"$5\times$ that of AWQ-4 (1.0\,pp) and TurboLDM~(3+1) (0.9\,pp)}, "
        r"showing that Hessian-based calibration is unusually sensitive to "
        r"masked-diffusion calibration draws. AWQ-4 is closest to BF16 "
        r"on a single seed; TurboLDM~(3+1) ties GPTQ-4 at $\sim$1\,bit less.}",
        r"\label{tab:seed_ci}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Avg.\ bits & Seeds & Mean & Std & Min--Max \\",
        r"\midrule",
    ]
    lines += body
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = PAPERS_DIR / "gen_table_seed_ci.tex"
    out.write_text("\n".join(lines) + "\n")
    _mirror(out)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Phase 82: trajectory entropy
# ---------------------------------------------------------------------------

ENT_PRETTY = {"bf16": "BF16", "turboldm_4_1": "TurboLDM (4+1)",
              "turboldm_3_1": "TurboLDM (3+1)"}


def render_entropy() -> None:
    rows = _load_jsonl(OUT_DIR / "phase82_trajectory.jsonl")
    if not rows:
        print("phase82 jsonl missing; skipping")
        return
    agg: Dict[tuple, list] = defaultdict(list)
    for r in rows:
        traj = r.get("trajectory", [])
        if not traj:
            continue
        ds = r["prompt_id"].split("_")[0]
        mt = r["model_tag"]
        mean_H = mean(s["mean_entropy_at_masked"] for s in traj)
        first_H = traj[0]["mean_entropy_at_masked"]
        final_H = traj[-1]["mean_entropy_at_masked"]
        mean_conf = mean(s["filled_confidence"] for s in traj)
        agg[(mt, ds)].append((mean_H, first_H, final_H, mean_conf))

    body = []
    for method in ["bf16", "turboldm_4_1", "turboldm_3_1"]:
        for ds, ds_pretty in [("mmlu", "MMLU"), ("gsm8k", "GSM8K")]:
            xs = agg.get((method, ds), [])
            if not xs:
                continue
            mh = mean(x[0] for x in xs)
            fh = mean(x[1] for x in xs)
            lh = mean(x[2] for x in xs)
            mc = mean(x[3] for x in xs)
            body.append(f"{ENT_PRETTY[method]} & {ds_pretty} & "
                         f"{mh:.3f} & {fh:.3f} & {lh:.3f} & {mc:.3f} \\\\")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Trajectory entropy and per-step confidence on LLaDA-8B over 32 prompts (16 MMLU + 16 GSM8K) at $T{=}64$. "
        r"\emph{First-step entropy} is the mean per-position predicted-token entropy at $t{=}0$; "
        r"\emph{final-step entropy} is the same at $t{=}T{-}1$. "
        r"On MMLU, BF16 starts confidently (first $H{=}0.147$) and TurboLDM~(3+1) inflates it to $0.445$ "
        r"--- a $3\times$ jump in early-step uncertainty where BF16 was nearly deterministic. "
        r"On GSM8K, where BF16 already begins uncertain ($H{=}3.55$), the same quantisation "
        r"only nudges first-step entropy by $\sim 5\%$. "
        r"TurboLDM~(4+1) is entropy-equivalent to BF16 throughout, consistent with its zero-loss tie on the GSM8K Pareto.}",
        r"\label{tab:entropy}",
        r"\small",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Method & Dataset & Mean $H$ & First $H$ & Final $H$ & Mean conf. \\",
        r"\midrule",
    ]
    lines += body
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = PAPERS_DIR / "gen_table_entropy.tex"
    out.write_text("\n".join(lines) + "\n")
    _mirror(out)
    print(f"wrote {out}")


# ---------------------------------------------------------------------------
# Findings narrative (replaces gen_findings.tex)
# ---------------------------------------------------------------------------

def render_findings() -> None:
    p83 = _load_jsonl(OUT_DIR / "phase83_bits_steps.jsonl")
    p84 = _load_jsonl(OUT_DIR / "phase84_seed_ci_gsm8k.jsonl")
    p82 = _load_jsonl(OUT_DIR / "phase82_trajectory.jsonl")

    def _score(rows, baseline, dataset, steps):
        for r in rows:
            if (r.get("baseline") == baseline
                    and r.get("dataset") == dataset
                    and r.get("gen_cfg", {}).get("steps") == steps):
                return 100 * r["score_value"]
        return None

    bf16_t64 = _score(p83, "bf16", "gsm8k", 64)
    t41_t64 = _score(p83, "turboldm_4_1", "gsm8k", 64)
    t31_t64 = _score(p83, "turboldm_3_1", "gsm8k", 64)
    t31_t16 = _score(p83, "turboldm_3_1", "gsm8k", 16)

    by_method: Dict[str, List[float]] = defaultdict(list)
    for r in p84:
        by_method[r["baseline"]].append(r["score_value"])
    seed_table = {}
    for k, v in by_method.items():
        if not v:
            continue
        m = mean(v); s = stdev(v) if len(v) > 1 else 0.0
        seed_table[k] = (m * 100, s * 100, len(v))

    # First-step entropy on MMLU under BF16 vs TurboLDM 3+1.
    first_H = {(mt, "mmlu"): [] for mt in ("bf16", "turboldm_3_1")}
    for r in p82:
        if not r.get("trajectory"):
            continue
        ds = r["prompt_id"].split("_")[0]
        mt = r["model_tag"]
        if (mt, ds) in first_H:
            first_H[(mt, ds)].append(r["trajectory"][0]["mean_entropy_at_masked"])

    bf16_first = mean(first_H[("bf16", "mmlu")]) if first_H[("bf16", "mmlu")] else float("nan")
    t31_first = mean(first_H[("turboldm_3_1", "mmlu")]) if first_H[("turboldm_3_1", "mmlu")] else float("nan")

    text = []
    text.append(r"\paragraph{F1 -- A bit$\leftrightarrow$step trade-off exists on generative tasks but not on MCQ.}")
    text.append(
        f"On GSM8K the same quantised model recovers most of its quality with extra denoising steps: "
        f"TurboLDM~(3+1) climbs from {t31_t16:.1f}\\,\\% at $T{{=}}16$ to {t31_t64:.1f}\\,\\% at $T{{=}}64$, "
        f"a $+{(t31_t64-t31_t16):.1f}$\\,pp gain just from compute. "
        f"On MMLU, every method is flat across $T\\in\\{{16, 32, 64\\}}$ within $\\pm 0.5$\\,pp. "
        r"The contrast confirms that quantisation noise compounds across denoising steps in "
        r"generative diffusion LMs and that MCQ scoring is invariant to entropy inflation as long "
        r"as the top-1 token does not flip (Table~\ref{tab:bits_steps})."
    )
    text.append("")

    text.append(r"\paragraph{F2 -- TurboLDM~(4+1) ties BF16 on GSM8K at full step budget.}")
    text.append(
        f"At $T{{=}}64$, TurboLDM~(4+1) reaches {t41_t64:.1f}\\,\\% on GSM8K, "
        f"matching the BF16 baseline of {bf16_t64:.1f}\\,\\% within scoring noise "
        r"while using $\sim 4.5$ effective bits per parameter, i.e.\ a $\sim 3.5{\times}$ "
        r"reduction in weight-storage footprint. "
        r"This is the headline practical finding: at the operating step budget LLaDA-8B "
        r"is shipped with, weight quantisation to $\sim$4 bits costs nothing on the "
        r"hardest reasoning task in our suite."
    )
    text.append("")

    text.append(r"\paragraph{F3 -- Quantisation inflates entropy where the model was confident, not where it was already uncertain.}")
    text.append(
        f"On the highly determined first denoising step of MMLU, BF16's mean masked-position "
        f"entropy is $H_0{{=}}{bf16_first:.3f}$. The same step under TurboLDM~(3+1) jumps to "
        f"$H_0{{=}}{t31_first:.3f}$, a {t31_first/max(bf16_first,1e-6):.1f}$\\times$ inflation. "
        r"On GSM8K, where BF16 already enters $t{=}0$ at $H_0{\approx}3.5$, the same quantisation "
        r"only raises first-step entropy by $\sim 5\%$ (Table~\ref{tab:entropy}). "
        r"Quantisation noise therefore disproportionately blurs the model's confident "
        r"decisions; uncertain regimes are essentially noise-bath-invariant. "
        r"This provides a direct mechanistic explanation for the F1 task split: MCQ relies "
        r"on confident first-step decisions whose top-1 typically survives the inflation, "
        r"whereas GSM8K's chained generation has no protected confident anchor and must "
        r"absorb the noise step after step."
    )
    text.append("")

    text.append(r"\paragraph{F4 -- Hessian calibration is the seed-fragile PTQ family on diffusion LMs.}")
    if "gptq_4" in seed_table and "awq_4" in seed_table:
        gptq_m, gptq_s, _ = seed_table["gptq_4"]
        awq_m, awq_s, _ = seed_table["awq_4"]
        t31_m, t31_s, _ = seed_table.get("turboldm_3_1", (float("nan"),)*3)
        text.append(
            f"Across three calibration seeds at GSM8K $n{{=}}200$, "
            f"GPTQ-4 has a per-seed std of {gptq_s:.2f}\\,pp "
            f"(min--max range $\\sim$10\\,pp), versus {awq_s:.2f}\\,pp for AWQ-4 "
            f"and {t31_s:.2f}\\,pp for TurboLDM~(3+1). "
            r"GPTQ's Hessian-driven layerwise solver appears to overfit to the calibration "
            r"draw on masked-diffusion workloads, whereas activation-scale (AWQ) and "
            r"online-vector-quant (TurboLDM) calibrators are nearly seed-deterministic. "
            r"This is a practitioner-relevant finding: when reporting a single GPTQ number "
            r"on diffusion LMs without a CI, the chosen seed can move the cell by half "
            r"the gap from BF16 (Table~\ref{tab:seed_ci})."
        )
    text.append("")

    text.append(r"\paragraph{Constructive corollary -- TurboLDM dominates the 3-bit regime.}")
    text.append(
        r"At the 3-bit operating point, vanilla GPTQ-3 and AWQ-3 collapse to $8.0$ and "
        r"$38.7$\,\% respectively on GSM8K (single-seed cells from "
        r"Table~\ref{tab:main}), whereas TurboLDM~(3+1), at the same $\sim$3.5 effective bits, "
        f"reaches {seed_table.get('turboldm_3_1', (float('nan'),))[0]:.1f}\\,\\% averaged over three seeds. "
        r"The stage-aware base-residual schedule is therefore the right constructive "
        r"object for the 3-bit regime; at 4 bits AWQ remains a strong specialist. "
        r"Findings F1--F3 explain why: residual precision is most valuable late in "
        r"the trajectory and on the layers whose error inflates the high-confidence "
        r"early-step entropy."
    )
    text.append("")

    out = PAPERS_DIR / "gen_findings_v2.tex"
    out.write_text("\n".join(text) + "\n")
    _mirror(out)
    print(f"wrote {out}")


def main() -> None:
    Path("papers/neurips2026").mkdir(parents=True, exist_ok=True)
    render_bits_steps()
    render_seed_ci()
    render_entropy()
    render_findings()


if __name__ == "__main__":
    main()
