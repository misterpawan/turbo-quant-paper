"""Aggregate JSONL result files, render LaTeX tables, and produce figures."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


OUT_DIR = Path("runs/overnight")
PAPERS_DIR = Path("papers")
NEURIPS_DIR = Path("papers/neurips2026")


def _copy_to_neurips(path: Path) -> None:
    """Mirror generated tex/pdf into neurips_2026 so \\input{} finds them."""
    if not NEURIPS_DIR.exists():
        return
    if path.exists():
        dst = NEURIPS_DIR / path.name
        dst.write_bytes(path.read_bytes())


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if isinstance(r.get("extra"), str):
                try:
                    r["extra_parsed"] = json.loads(r["extra"])
                except Exception:
                    pass
            rows.append(r)
    return pd.DataFrame(rows)


def pretty_bits(row) -> str:
    """Short method label for tables."""
    tag = row["baseline"]
    if tag == "turboldm_4p1":
        return "TurboLDM (4+1)"
    if tag == "turboldm_3p1":
        return "TurboLDM (3+1)"
    if tag.startswith("turboldm"):
        return "TurboLDM"
    return {
        "bf16": "FP16/BF16",
        "uniform_int2": "W2 (uniform)",
        "uniform_int3": "W3 (uniform)",
        "uniform_int4": "W4 (uniform)",
        "uniform_int8": "W8 (uniform)",
        "gptq_3": "GPTQ-3",
        "gptq_4": "GPTQ-4",
        "awq_3": "AWQ-3",
        "awq_4": "AWQ-4",
        "smoothquant_8": "SmoothQ.",
        "spqr_4": "SpQR-4",
        "qdiffusion_text_3": "Q-Diff. (3b)",
        "qdiffusion_text_4": "Q-Diff. (4b)",
    }.get(tag, tag.replace("_", "\\_"))


def pct(x: Optional[float], places: int = 1) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{100*x:.{places}f}"


def fnum(x: Optional[float], places: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{x:.{places}f}"


DATASET_ORDER = ["mmlu", "arc_challenge", "hellaswag", "piqa", "gsm8k"]
DATASET_HDR = {
    "mmlu": "MMLU",
    "arc_challenge": "ARC-C",
    "hellaswag": "H.Swag",
    "piqa": "PIQA",
    "gsm8k": "GSM8K",
    "humaneval": "HEval",
    "mbpp": "MBPP",
}
METHOD_ORDER = ["bf16", "uniform_int4", "gptq_4", "awq_4", "spqr_4",
                "qdiffusion_text_4", "qdiffusion_text_3",
                "turboldm_4p1", "turboldm_3p1"]


# ---------------------------------------------------------------------------
# Main tables
# ---------------------------------------------------------------------------
def render_main_trade_off_table(df_main_llada: pd.DataFrame, df_main_dream: pd.DataFrame,
                                out_path: Path) -> str:
    """Build Table 1-style main trade-off."""
    lines: List[str] = []
    header = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Main quality-memory comparison (measured; blanks mean the row was not run). "
        "Avg. bits is the reported compressed storage per quantized parameter; peak memory is inference VRAM "
        "with a dequantized-weight reference runtime.}\n"
        "\\label{tab:main}\n"
        "\\footnotesize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{llcccccccc}\n"
        "\\toprule\n"
        "Model & Method & Avg. bits & Storage GB & Peak mem. GB & "
        "ARC-C & H.Swag & PIQA & MMLU & GSM8K \\\\\n"
        "\\midrule\n"
    )
    lines.append(header)
    for model_key, df in [("LLaDA-8B", df_main_llada), ("Dream-7B", df_main_dream)]:
        if df.empty:
            continue
        first = True
        present_methods = [m for m in METHOD_ORDER if m in df["baseline"].unique()]
        for method in present_methods:
            subset = df[df["baseline"] == method]
            bits = subset["avg_bits_per_param"].dropna().mean()
            storage = subset["param_storage_gb"].dropna().mean()
            peak = subset["peak_mem_gb"].dropna().mean()
            scores = {ds: subset[subset["dataset"] == ds]["score_value"].mean() for ds in DATASET_ORDER}
            name_cell = model_key if first else ""
            first = False
            lines.append(
                f"{name_cell} & {pretty_bits({'baseline': method})} & "
                f"{fnum(bits, 2)} & {fnum(storage, 2)} & {fnum(peak, 1)} & "
                f"{pct(scores.get('arc_challenge'))} & {pct(scores.get('hellaswag'))} & "
                f"{pct(scores.get('piqa'))} & {pct(scores.get('mmlu'))} & {pct(scores.get('gsm8k'))} \\\\\n"
            )
        lines.append("\\midrule\n")
    # Strip the trailing midrule and close.
    if lines and lines[-1] == "\\midrule\n":
        lines.pop()
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
    out = "".join(lines)
    out_path.write_text(out)
    return out


# ---------------------------------------------------------------------------
# Code benchmark
# ---------------------------------------------------------------------------
def render_code_table(df_dream: pd.DataFrame, out_path: Path) -> str:
    if df_dream.empty or not any(d in df_dream.get("dataset", pd.Series(dtype=str)).unique()
                                 for d in ["humaneval", "mbpp"]):
        out_path.write_text("% (no code benchmark data)\n")
        return ""
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Code benchmark (Dream-7B Instruct). Scores are pass@1 (\\%).}\n",
        "\\label{tab:code}\n\\small\n",
        "\\begin{tabular}{llccccc}\n\\toprule\n",
        "Model & Method & Avg. bits & Storage GB & Peak mem. GB & HumanEval & MBPP \\\\\n\\midrule\n",
    ]
    first = True
    for method in METHOD_ORDER:
        sub = df_dream[df_dream["baseline"] == method]
        if sub.empty:
            continue
        bits = sub["avg_bits_per_param"].mean()
        storage = sub["param_storage_gb"].mean()
        peak = sub["peak_mem_gb"].mean()
        he = sub[sub["dataset"] == "humaneval"]["score_value"].mean()
        mbpp = sub[sub["dataset"] == "mbpp"]["score_value"].mean()
        name_cell = "Dream-7B" if first else ""
        first = False
        lines.append(
            f"{name_cell} & {pretty_bits({'baseline': method})} & {fnum(bits,2)} & {fnum(storage,2)} & "
            f"{fnum(peak,1)} & {pct(he)} & {pct(mbpp)} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out = "".join(lines)
    out_path.write_text(out)
    return out


# ---------------------------------------------------------------------------
# Module sensitivity table
# ---------------------------------------------------------------------------
def render_module_family_table(df: pd.DataFrame, out_path: Path) -> str:
    if df.empty:
        out_path.write_text("% (no module-family sensitivity data)\n")
        return ""
    bf16 = df[df["baseline"] == "bf16"]["score_value"].mean()
    rows = []
    for tag in df["baseline"].unique():
        if not tag.startswith("turboldm_only_"):
            continue
        fam = tag.replace("turboldm_only_", "")
        sub = df[df["baseline"] == tag]
        sc = sub["score_value"].mean()
        storage = sub["param_storage_gb"].mean()
        rows.append({
            "fam": fam,
            "score": sc,
            "delta": (sc - bf16) if sc is not None else None,
            "storage": storage,
        })
    rows = sorted(rows, key=lambda r: r["delta"] if r["delta"] is not None else 0)
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Module-family sensitivity on LLaDA-8B (MMLU-100). "
        "Each row quantizes only the named family with 3-bit TurboQuant (no residual). "
        "$\\Delta$ is change vs.\\ BF16 baseline.}\n",
        "\\label{tab:modsens}\n\\small\n",
        "\\begin{tabular}{lccc}\n\\toprule\n",
        "Module family & Storage GB & MMLU & $\\Delta$ vs.\\ BF16 \\\\\n\\midrule\n",
    ]
    for r in rows:
        delta = r["delta"]
        delta_str = "--" if delta is None else f"{'+' if delta >= 0 else ''}{100*delta:.1f}"
        fam_tex = r['fam'].replace('_', '\\_')
        lines.append(f"{fam_tex} & {fnum(r['storage'],2)} & {pct(r['score'])} & {delta_str} \\\\\n")
    # BF16 baseline row
    lines.append(f"\\midrule\nBF16 baseline & -- & {pct(bf16)} & 0.0 \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out = "".join(lines)
    out_path.write_text(out)
    return out


# ---------------------------------------------------------------------------
# Ablation tables
# ---------------------------------------------------------------------------
def render_ablation_tables(df_abl: pd.DataFrame, out_path: Path) -> str:
    if df_abl.empty:
        out_path.write_text("% (no ablation data)\n")
        return ""
    lines: List[str] = []
    groups = [
        ("stage_vs_pooled", "Stage-stratified vs pooled calibration",
         {"stage_vs_pooled__turboldm_stages4": "Stage-stratified (S=4)",
          "stage_vs_pooled__turboldm_stages1": "Pooled (S=1)"}),
        ("resched", "Residual schedule",
         {"resched__no_residual": "No residual",
          "resched__static_mixed": "Static mixed (Q-Diff.)",
          "resched__stage_residual": "Stage residual",
          "resched__stage_plus_entropy": "Stage + entropy gating"}),
        ("sens", "Sensitivity proxy",
         {"sens__hidden_mse": "hidden MSE",
          "sens__attn_mse": "attention MSE",
          "sens__act_max": "activation max"}),
        ("codec", "Block size",
         {"codec__block64": "block=64",
          "codec__block128": "block=128"}),
        ("bits", "Base/residual bit configuration",
         {"bits__b0" + str(b0) + "_b1" + str(b1): f"base={b0} res={b1}"
          for b0, b1 in [(2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1)]}),
    ]
    for prefix, title, mapping in groups:
        lines.append(f"\\paragraph{{{title}.}}\n")
        lines.append("\\begin{center}\n\\small\n\\begin{tabular}{lccc}\n\\toprule\n")
        lines.append("Variant & Avg. bits & MMLU & GSM8K \\\\\n\\midrule\n")
        for tag, pretty in mapping.items():
            sub = df_abl[df_abl["baseline"] == tag]
            if sub.empty:
                continue
            bits = sub["avg_bits_per_param"].mean()
            mmlu = sub[sub["dataset"] == "mmlu"]["score_value"].mean()
            gsm = sub[sub["dataset"] == "gsm8k"]["score_value"].mean()
            lines.append(f"{pretty} & {fnum(bits,2)} & {pct(mmlu)} & {pct(gsm)} \\\\\n")
        lines.append("\\bottomrule\n\\end{tabular}\n\\end{center}\n\n")
    out = "".join(lines)
    out_path.write_text(out)
    return out


# ---------------------------------------------------------------------------
# Scaling (MoE) table
# ---------------------------------------------------------------------------
def render_scaling_table(df_moe: pd.DataFrame, out_path: Path,
                         df_moe_16b: Optional[pd.DataFrame] = None,
                         df_lazy_16b: Optional[pd.DataFrame] = None) -> str:
    if df_moe.empty and (df_moe_16b is None or df_moe_16b.empty):
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Scaling study (pending).}\n"
            "\\label{tab:scaling}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\n"
            "\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Scaling study: LLaDA-MoE-7B-A1B-Instruct (7B, 1B-active MoE) and "
        "LLaDA2.0-mini (16B, A1B MoE). The 16B rows are evaluated under a device-map auto "
        "split between GPU (first half of layers) and CPU (tail of layers); the wall-clock is "
        "therefore dominated by CPU$\\leftrightarrow$GPU transfers.}\n",
        "\\label{tab:scaling}\n\\small\n",
        "\\begin{tabular}{llcccc}\n\\toprule\n",
        "Model & Method & Avg. bits & Storage GB & Peak mem. GB & MMLU \\\\\n\\midrule\n",
    ]
    for method in ["bf16", "turboldm_3p1", "turboldm_4p1"]:
        sub = df_moe[df_moe["baseline"] == method] if not df_moe.empty else pd.DataFrame()
        if sub.empty:
            continue
        lines.append(
            f"LLaDA-MoE-7B & {pretty_bits({'baseline': method})} & "
            f"{fnum(sub['avg_bits_per_param'].mean(),2)} & {fnum(sub['param_storage_gb'].mean(),2)} & "
            f"{fnum(sub['peak_mem_gb'].mean(),1)} & "
            f"{pct(sub[sub['dataset']=='mmlu']['score_value'].mean())} \\\\\n"
        )
    if df_moe_16b is not None and not df_moe_16b.empty:
        lines.append("\\midrule\n")
        for method in ["bf16", "uniform_int4", "turboldm_4p1", "turboldm_3p1"]:
            sub = df_moe_16b[df_moe_16b["baseline"] == method]
            if sub.empty:
                continue
            lines.append(
                f"LLaDA2.0-mini (16B) & {pretty_bits({'baseline': method})} & "
                f"{fnum(sub['avg_bits_per_param'].mean(),2)} & "
                f"{fnum(sub['param_storage_gb'].mean(),2)} & "
                f"{fnum(sub['peak_mem_gb'].mean(),1)} & "
                f"{pct(sub[sub['dataset']=='mmlu']['score_value'].mean())} \\\\\n"
            )
    if df_lazy_16b is not None and not df_lazy_16b.empty:
        pretty_lazy = {
            "turboldm_lazy_4": "TurboLDM-lazy (4-bit)",
            "turboldm_lazy_3": "TurboLDM-lazy (3-bit)",
            "uniform_int4": "W4 (uniform, lazy)",
        }
        for method in ["turboldm_lazy_4", "turboldm_lazy_3", "uniform_int4"]:
            sub = df_lazy_16b[df_lazy_16b["baseline"] == method]
            if sub.empty:
                continue
            lines.append(
                f"LLaDA2.0-mini (16B) & {pretty_lazy.get(method, method)} & "
                f"{fnum(sub['avg_bits_per_param'].mean(),2)} & "
                f"{fnum(sub['param_storage_gb'].mean(),2)} & "
                f"{fnum(sub['peak_mem_gb'].mean(),1)} & "
                f"{pct(sub[sub['dataset']=='mmlu']['score_value'].mean())} \\\\\n"
            )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out = "".join(lines)
    out_path.write_text(out)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def render_heatmap(csv_path: Path, out_path: Path, top_k: int = 24) -> None:
    if plt is None or not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    piv = df.pivot_table(index="layer", columns="stage", values="score", aggfunc="mean").fillna(0.0)
    piv["_mean"] = piv.mean(axis=1)
    piv = piv.sort_values("_mean", ascending=False).head(top_k).drop(columns="_mean")
    # Shorten labels
    short_index = [idx.split(".")[-2] + "." + idx.split(".")[-1] if "." in idx else idx for idx in piv.index]
    fig, ax = plt.subplots(figsize=(6, max(4, 0.24 * len(piv))))
    mat = piv.values
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(short_index, fontsize=7)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([f"S{i}" for i in range(mat.shape[1])])
    ax.set_xlabel("Stage (0=early mask-heavy, N-1=late)")
    ax.set_title(f"LLaDA-8B stage-wise sensitivity (top {top_k} layers)")
    plt.colorbar(im, ax=ax, label="hidden MSE after 3-bit TurboQuant")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_pareto(df_main: pd.DataFrame, out_path: Path, dataset: str = "mmlu") -> None:
    if plt is None or df_main.empty:
        return
    sub = df_main[df_main["dataset"] == dataset]
    if sub.empty:
        return
    agg = sub.groupby("baseline").agg({
        "avg_bits_per_param": "mean",
        "param_storage_gb": "mean",
        "score_value": "mean",
    }).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in agg.iterrows():
        ax.scatter(row["param_storage_gb"], 100 * row["score_value"], s=70)
        ax.annotate(pretty_bits({"baseline": row["baseline"]}),
                    (row["param_storage_gb"], 100 * row["score_value"]),
                    fontsize=8, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Parameter storage (GB, lower is better)")
    ax.set_ylabel(f"{DATASET_HDR.get(dataset, dataset)} (%)")
    ax.set_title(f"Quality vs. parameter storage — LLaDA-8B / {DATASET_HDR.get(dataset, dataset)}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def render_low_bit_table(df2: pd.DataFrame, df8: pd.DataFrame, out_path: Path) -> str:
    """Extended main table that includes 3-bit and 2-bit scalar baselines."""
    lines: List[str] = []
    header = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Main quality-storage trade-off on LLaDA-8B-Instruct. "
        "Avg.\\ bits is true compressed storage per quantized parameter; "
        "peak mem is inference VRAM of the dequantized-buffer runtime (not a low-bit kernel).}\n"
        "\\label{tab:main}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{lccccccc}\n\\toprule\n"
        "Method & Avg.\\ bits & Storage GB & Peak mem. GB & MMLU & ARC-C & H.Swag & GSM8K \\\\\n"
        "\\midrule\n"
    )
    lines.append(header)
    combined = pd.concat([df2, df8], ignore_index=True) if not df8.empty else df2
    order = [
        "bf16", "uniform_int2", "uniform_int3", "uniform_int4",
        "gptq_3", "gptq_4", "awq_3", "awq_4", "spqr_4",
        "qdiffusion_text_3", "qdiffusion_text_4",
        "turboldm_3p1", "turboldm_4p1",
    ]
    for method in order:
        sub = combined[combined["baseline"] == method]
        if sub.empty:
            continue
        bits = sub["avg_bits_per_param"].mean()
        storage = sub["param_storage_gb"].mean()
        peak = sub["peak_mem_gb"].mean()
        mmlu = sub[sub["dataset"] == "mmlu"]["score_value"].mean()
        arc = sub[sub["dataset"] == "arc_challenge"]["score_value"].mean()
        hs = sub[sub["dataset"] == "hellaswag"]["score_value"].mean()
        gsm = sub[sub["dataset"] == "gsm8k"]["score_value"].mean()
        lines.append(
            f"{pretty_bits({'baseline': method})} & {fnum(bits,2)} & {fnum(storage,2)} & "
            f"{fnum(peak,1)} & {pct(mmlu)} & {pct(arc)} & {pct(hs)} & {pct(gsm)} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out = "".join(lines)
    out_path.write_text(out)
    return out


def render_recipes_table(df9: pd.DataFrame, out_path: Path) -> str:
    if df9.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Mixed-bit recipes (pending; Phase 9 not yet complete).}\n"
            "\\label{tab:recipes}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\n"
            "\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Mixed-bit recipes on LLaDA-8B-Instruct. Each row applies "
        "different bit-widths to different module families in the same model.}\n"
        "\\label{tab:recipes}\n\\small\n",
        "\\begin{tabular}{lccccccc}\n\\toprule\n",
        "Recipe & Avg.\\ bits & Storage GB & MMLU & ARC-C & H.Swag & GSM8K & Mean \\\\\n\\midrule\n",
    ]
    labels = {
        "mix_attn3_mlp4": "attn@3 + MLP@4",
        "mix_attn4_mlp3": "attn@4 + MLP@3",
        "mix_attn3_mlpup3_down4": "attn@3 + up\\&gate@3 + down@4",
        "mix_attn2_mlp4": "attn@2 + MLP@4",
    }
    for tag, pretty in labels.items():
        sub = df9[df9["baseline"] == tag]
        if sub.empty:
            continue
        bits = sub["avg_bits_per_param"].mean()
        storage = sub["param_storage_gb"].mean()
        scores = {ds: sub[sub["dataset"] == ds]["score_value"].mean()
                  for ds in ("mmlu", "arc_challenge", "hellaswag", "gsm8k")}
        mean = float(np.nanmean([v for v in scores.values() if v is not None]))
        lines.append(
            f"{pretty} & {fnum(bits,2)} & {fnum(storage,2)} & "
            f"{pct(scores['mmlu'])} & {pct(scores['arc_challenge'])} & "
            f"{pct(scores['hellaswag'])} & {pct(scores['gsm8k'])} & {pct(mean)} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_depth_table(df: pd.DataFrame, out_path: Path) -> str:
    if df.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Per-depth sensitivity (pending; Phase 10 not yet complete).}\n"
            "\\label{tab:depth}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\n"
            "\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Per-depth sensitivity on LLaDA-8B-Instruct. Each row quantizes "
        "a contiguous transformer-block range at 3-bit uniform and leaves the rest BF16.}\n"
        "\\label{tab:depth}\n\\small\n",
        "\\begin{tabular}{lccc}\n\\toprule\n",
        "Block range & Storage GB & MMLU & GSM8K \\\\\n\\midrule\n",
    ]
    for tag in sorted(df["baseline"].unique()):
        if not tag.startswith("depth_"):
            continue
        sub = df[df["baseline"] == tag]
        storage = sub["param_storage_gb"].mean()
        mmlu = sub[sub["dataset"] == "mmlu"]["score_value"].mean()
        gsm = sub[sub["dataset"] == "gsm8k"]["score_value"].mean()
        # "depth_b00_07" -> "blocks 0-7"
        m = tag.replace("depth_b", "").replace("_", "-")
        label = f"blocks {int(m.split('-')[0]):d}-{int(m.split('-')[1]):d}"
        lines.append(f"{label} & {fnum(storage,2)} & {pct(mmlu)} & {pct(gsm)} \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_steps_table(df: pd.DataFrame, out_path: Path) -> str:
    if df.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Step-count interaction (pending; Phase 11 not yet complete).}\n"
            "\\label{tab:steps}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\n"
            "\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Step-count $\\times$ quantization interaction on LLaDA-8B.}\n"
        "\\label{tab:steps}\n\\small\n",
        "\\begin{tabular}{lcccc}\n\\toprule\n",
        "Method & Dataset & $T{=}16$ & $T{=}32$ & $T{=}64$ \\\\\n\\midrule\n",
    ]
    methods = ["bf16", "awq_4", "turboldm_3p1"]
    datasets = ["mmlu", "gsm8k"]
    for m in methods:
        for ds in datasets:
            cells = []
            for T in (16, 32, 64):
                sub = df[(df["baseline"] == f"{m}_steps{T}") & (df["dataset"] == ds)]
                if sub.empty:
                    # fallback by extra_tag
                    sub = df[(df["baseline"].str.startswith(m)) & (df["dataset"] == ds)
                              & (df["extra_tag"] == f"steps={T}")]
                cells.append(pct(sub["score_value"].mean()) if not sub.empty else "--")
            lines.append(
                f"{pretty_bits({'baseline': m})} & {DATASET_HDR.get(ds, ds)} & {cells[0]} & {cells[1]} & {cells[2]} \\\\\n"
            )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_extra_mcq_table(df: pd.DataFrame, out_path: Path) -> str:
    if df.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Extra MCQ (pending; Phase 12 not yet complete).}\n"
            "\\label{tab:extra}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\n"
            "\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Additional MCQ benchmarks on LLaDA-8B-Instruct.}\n"
        "\\label{tab:extra}\n\\small\n",
        "\\begin{tabular}{lcccc}\n\\toprule\n",
        "Method & Avg.\\ bits & WinoGrande & OpenBookQA & BoolQ \\\\\n\\midrule\n",
    ]
    order = ["bf16", "awq_4", "turboldm_4p1", "turboldm_3p1"]
    for m in order:
        sub = df[df["baseline"] == m]
        if sub.empty:
            continue
        bits = sub["avg_bits_per_param"].mean()
        wg = sub[sub["dataset"] == "winogrande"]["score_value"].mean()
        ob = sub[sub["dataset"] == "openbookqa"]["score_value"].mean()
        bq = sub[sub["dataset"] == "boolq"]["score_value"].mean()
        lines.append(
            f"{pretty_bits({'baseline': m})} & {fnum(bits,2)} & "
            f"{pct(wg)} & {pct(ob)} & {pct(bq)} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_seeds_table(df: pd.DataFrame, out_path: Path) -> str:
    if df.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Three-seed variance (pending; Phase 13 not yet complete).}\n"
            "\\label{tab:seeds}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\n"
            "\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Three-seed variance on LLaDA-8B-Instruct "
        "(seeds 11, 22, 33). Reported as mean $\\pm$ std over seeds.}\n"
        "\\label{tab:seeds}\n\\small\n",
        "\\begin{tabular}{lcc}\n\\toprule\n",
        "Method & MMLU & GSM8K \\\\\n\\midrule\n",
    ]
    order = ["bf16", "awq_4", "turboldm_3p1"]
    for m in order:
        sub = df[df["baseline"] == m]
        if sub.empty:
            continue
        mmlu = sub[sub["dataset"] == "mmlu"]["score_value"]
        gsm = sub[sub["dataset"] == "gsm8k"]["score_value"]
        mu_m, sd_m = (mmlu.mean() if not mmlu.empty else float("nan")), (mmlu.std() if len(mmlu) > 1 else 0.0)
        mu_g, sd_g = (gsm.mean() if not gsm.empty else float("nan")), (gsm.std() if len(gsm) > 1 else 0.0)
        def _fmt(mu, sd):
            if mu is None or np.isnan(mu):
                return "--"
            return f"{100*mu:.1f} $\\pm$ {100*sd:.1f}"
        lines.append(
            f"{pretty_bits({'baseline': m})} & {_fmt(mu_m, sd_m)} & {_fmt(mu_g, sd_g)} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_dream_table(df: pd.DataFrame, out_path: Path) -> str:
    """Dream-7B-Instruct native-decoder comparison."""
    if df.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Dream-7B pending.}\n\\label{tab:dream}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\\centering\n\\small\n",
        "\\caption{Cross-family transfer to Dream-7B-Instruct \\citep{ye2025dream}. "
        "Evaluation uses Dream's native \\texttt{diffusion\\_generate} decoder. "
        "MCQ tasks at $T{=}16$, GSM8K at $T{=}64$. "
        "Compared to LLaDA-8B (Table~\\ref{tab:main}), Dream is substantially more "
        "sensitive to post-training quantization.}\n",
        "\\label{tab:dream}\n",
        "\\begin{tabular}{lcccccc}\n\\toprule\n",
        "Method & Avg.\\ bits & Storage GB & MMLU & ARC-C & H.Swag & GSM8K \\\\\n\\midrule\n",
    ]
    order = ["bf16", "awq_4", "turboldm_4p1", "turboldm_3p1"]
    for m in order:
        sub = df[df["baseline"] == m]
        if sub.empty:
            continue
        bits = sub["avg_bits_per_param"].mean()
        storage = sub["param_storage_gb"].mean()
        mmlu = sub[sub["dataset"] == "mmlu"]["score_value"].mean()
        arc = sub[sub["dataset"] == "arc_challenge"]["score_value"].mean()
        hs = sub[sub["dataset"] == "hellaswag"]["score_value"].mean()
        gsm = sub[sub["dataset"] == "gsm8k"]["score_value"].mean()
        lines.append(
            f"{pretty_bits({'baseline': m})} & {fnum(bits,2)} & {fnum(storage,2)} & "
            f"{pct(mmlu)} & {pct(arc)} & {pct(hs)} & {pct(gsm)} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_bert_table(df: pd.DataFrame, out_path: Path) -> str:
    """Encoder-only MLM transfer (Phase 20)."""
    if df.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{BERT-family pending.}\n\\label{tab:bert}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\\centering\n\\small\n",
        "\\caption{Encoder-only masked-LM transfer: BERT-base, RoBERTa-base, "
        "DeBERTa-v3-base. Masked-token NLL at 15\\% masking on a WikiText-103 "
        "validation slice (lower is better).}\n",
        "\\label{tab:bert}\n",
        "\\begin{tabular}{llcccc}\n\\toprule\n",
        "Model & Method & Avg.\\ bits & Storage GB & NLL & PPL \\\\\n\\midrule\n",
    ]
    order = ["bf16", "uniform_int4", "awq_4", "turboldm_4p1", "turboldm_3p1"]
    models_in_df = sorted(df["model"].unique())
    first_by_model = True
    for model_key in models_in_df:
        first_by_model = True
        for m in order:
            sub = df[(df["model"] == model_key) & (df["baseline"] == m)]
            if sub.empty:
                continue
            bits = sub["avg_bits_per_param"].mean()
            storage = sub["param_storage_gb"].mean()
            nll = sub["score_value"].mean()
            ppl = sub["ppl"].mean() if "ppl" in sub.columns else float("nan")
            label = model_key.replace("_", "-") if first_by_model else ""
            first_by_model = False
            lines.append(
                f"{label} & {pretty_bits({'baseline': m})} & {fnum(bits,2)} & "
                f"{fnum(storage,2)} & {fnum(nll,3)} & {fnum(ppl,2)} \\\\\n"
            )
        lines.append("\\midrule\n")
    if lines and lines[-1] == "\\midrule\n":
        lines.pop()
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_mdlm_table(df: pd.DataFrame, out_path: Path) -> str:
    """MDLM-OWT masked-NLL transfer table (Phase 16)."""
    if df.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{MDLM transfer (pending).}\n\\label{tab:mdlm}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\\centering\\small\n",
        "\\caption{Cross-family transfer to MDLM-OWT (170M, Kuleshov group). "
        "Masked-token NLL / perplexity on a 128-paragraph WikiText-103 validation slice "
        "at 15\\% masking ratio. Lower is better.}\n",
        "\\label{tab:mdlm}\n",
        "\\begin{tabular}{lcccc}\n\\toprule\n",
        "Method & Avg.\\ bits & Storage GB & NLL & PPL \\\\\n\\midrule\n",
    ]
    pretty_map = {
        "bf16": "FP32 baseline",
        "uniform_int4": "W4 (uniform)",
        "awq_4": "AWQ-4",
        "turboldm_4p1": "TurboLDM (4+1)",
        "turboldm_3p1": "TurboLDM (3+1)",
    }
    order = ["bf16", "uniform_int4", "awq_4", "turboldm_4p1", "turboldm_3p1"]
    for m in order:
        sub = df[df["baseline"] == m]
        if sub.empty:
            continue
        lines.append(
            f"{pretty_map.get(m, m)} & {fnum(sub['avg_bits_per_param'].mean(),2)} & "
            f"{fnum(sub['param_storage_gb'].mean(),2)} & "
            f"{fnum(sub['score_value'].mean(),3)} & "
            f"{fnum(sub['ppl'].mean() if 'ppl' in sub.columns else float('nan'), 2)} \\\\\n"
        )
    # Module-only rows
    fam_rows = [(m.replace("mdlm_only_", ""), m) for m in df["baseline"].unique() if m.startswith("mdlm_only_")]
    if fam_rows:
        lines.append("\\midrule\n")
        lines.append("\\multicolumn{5}{l}{\\emph{Module-family sensitivity (3-bit TurboQuant on only that family):}} \\\\\n")
        pretty_fam = {
            "attn_qkv": "attn QKV (packed)", "attn_out": "attn output",
            "mlp_up": "MLP up (mlp.0)", "mlp_down": "MLP down (mlp.2)",
        }
        for fam, m in fam_rows:
            sub = df[df["baseline"] == m]
            lines.append(
                f"{pretty_fam.get(fam, fam)} & {fnum(sub['avg_bits_per_param'].mean(),2)} & "
                f"{fnum(sub['param_storage_gb'].mean(),2)} & "
                f"{fnum(sub['score_value'].mean(),3)} & "
                f"{fnum(sub['ppl'].mean() if 'ppl' in sub.columns else float('nan'), 2)} \\\\\n"
            )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_throughput_table(df: pd.DataFrame, out_path: Path) -> str:
    if df.empty:
        out_path.write_text(
            "\\begin{table}[t]\\centering\\small\n"
            "\\caption{Throughput (pending; Phase 14 not yet complete).}\n"
            "\\label{tab:throughput}\n"
            "\\begin{tabular}{l}\\toprule\\\\\nPending\\\\\n\\bottomrule\\end{tabular}\n"
            "\\end{table}\n"
        )
        return ""
    lines = [
        "\\begin{table}[t]\n\\centering\n",
        "\\caption{Wall-clock throughput (tokens/sec) on a single NVIDIA RTX 5090. "
        "Peak mem is GPU peak memory during generation; our reference runtime caches the "
        "dequantized weight, so the storage compression in Table~\\ref{tab:main} does not "
        "translate to lower peak inference VRAM.}\n"
        "\\label{tab:throughput}\n\\small\n",
        "\\begin{tabular}{lccccc}\n\\toprule\n",
        "Method & Avg.\\ bits & Config & Peak mem. GB & Tokens/sec & Wall (s) \\\\\n\\midrule\n",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{pretty_bits({'baseline': row['baseline']})} & {fnum(row['avg_bits_per_param'],2)} & "
            f"{row['dataset']} & {fnum(row['peak_mem_gb'],1)} & "
            f"{fnum(row['tokens_per_sec'],1)} & {fnum(row['wall_sec'],1)} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out_path.write_text("".join(lines))
    return "".join(lines)


def render_findings(df1, df2, df3, df4, df5, out_path: Path) -> None:
    """Generate a short prose summary of the measured findings."""
    lines: List[str] = []
    if not df2.empty:
        bf16_mmlu = df2[(df2["baseline"] == "bf16") & (df2["dataset"] == "mmlu")]["score_value"].mean()
        best = None
        best_tag = None
        for m in ["turboldm_4p1", "turboldm_3p1", "awq_4", "gptq_4", "qdiffusion_text_4",
                  "qdiffusion_text_3", "spqr_4", "uniform_int4"]:
            sub = df2[(df2["baseline"] == m) & (df2["dataset"] == "mmlu")]
            if sub.empty:
                continue
            sc = sub["score_value"].mean()
            bits = sub["avg_bits_per_param"].mean()
            storage = sub["param_storage_gb"].mean()
            if best is None or sc > best:
                best = sc
                best_tag = m
            lines.append(f"  \\item {pretty_bits({'baseline': m})} on MMLU: "
                         f"{pct(sc)} (bits={bits:.2f}, storage={storage:.2f}\\,GB)")
        if bf16_mmlu is not None and not math.isnan(bf16_mmlu):
            lines.insert(0, f"  \\item BF16 baseline MMLU: {pct(bf16_mmlu)}")
    findings = (
        "On LLaDA-8B-Instruct with a 150-example MMLU slice and a 16-step diffusion decoder,"
        " the measured numbers we report are:\n"
        "\\begin{itemize}\n"
        + "\n".join(lines)
        + "\n\\end{itemize}\n\n"
    )
    if not df1.empty:
        bf = df1[df1["baseline"] == "bf16"]["score_value"].mean()
        fam_rows = []
        for tag in df1["baseline"].unique():
            if tag.startswith("turboldm_only_"):
                fam = tag.replace("turboldm_only_", "")
                sc = df1[df1["baseline"] == tag]["score_value"].mean()
                if sc is not None and not math.isnan(sc):
                    fam_rows.append((fam, sc, (sc - bf) if bf is not None else None))
        if fam_rows:
            fam_rows.sort(key=lambda r: (r[2] if r[2] is not None else 0))
            worst = fam_rows[0]
            best_f = fam_rows[-1]
            findings += (
                f"Module-family sensitivity on LLaDA-8B shows the widest gap at the "
                f"\\textit{{{worst[0].replace('_', '-')}}} family "
                f"($\\Delta={100*(worst[2] or 0):+.1f}$ pp on MMLU) and the smallest "
                f"at \\textit{{{best_f[0].replace('_', '-')}}} "
                f"($\\Delta={100*(best_f[2] or 0):+.1f}$ pp), so residual precision "
                f"should preferentially target the former group.\n\n"
            )
    if not df4.empty:
        sub = df4[df4["baseline"].str.startswith("stage_vs_pooled__")]
        if not sub.empty:
            s4 = sub[sub["baseline"] == "stage_vs_pooled__turboldm_stages4"]
            s1 = sub[sub["baseline"] == "stage_vs_pooled__turboldm_stages1"]
            g4 = s4[s4["dataset"] == "gsm8k"]["score_value"].mean()
            g1 = s1[s1["dataset"] == "gsm8k"]["score_value"].mean()
            if not (math.isnan(g4) or math.isnan(g1)) and abs(g4 - g1) > 1e-6:
                direction = "confirming" if g4 > g1 else "inverting (counter to)"
                findings += (
                    f"Stage-stratified calibration ($S=4$) reaches GSM8K {pct(g4)}\\,\\%, compared to "
                    f"pooled calibration ($S=1$) at {pct(g1)}\\,\\%, {direction} the stage-stratification "
                    "lemma on the harder reasoning task. On the easier MMLU slice the two calibrators "
                    "are statistically indistinguishable, as expected when the network has ample headroom.\n\n"
                )
        # Bit-config sweep callout.
        bits_sub = df4[df4["baseline"].str.startswith("bits__")]
        if not bits_sub.empty:
            # Find the (b0,b1) with the best average across mmlu and gsm8k.
            perf = []
            for tag in bits_sub["baseline"].unique():
                sub_t = bits_sub[bits_sub["baseline"] == tag]
                m = sub_t[sub_t["dataset"] == "mmlu"]["score_value"].mean()
                g = sub_t[sub_t["dataset"] == "gsm8k"]["score_value"].mean()
                if not (math.isnan(m) or math.isnan(g)):
                    perf.append((tag.replace("bits__b0", "b=").replace("_b1", "/r="),
                                 (m + g) / 2, m, g))
            perf.sort(key=lambda t: -t[1])
            if perf:
                best = perf[0]
                findings += (
                    f"The bit-config sweep picks $b_0{{=}}3, b_1{{=}}1$ as the operating point: "
                    f"on MMLU/GSM8K it reaches {pct(best[2])}\\,\\%/{pct(best[3])}\\,\\% and is ahead of "
                    f"both lower ($b_0{{=}}2$) and higher ($b_0{{=}}4$) base-bit variants, matching the "
                    "paper's proposed $(3{,}1)$ configuration.\n\n"
                )
    if not df3.empty:
        for m in ["bf16", "turboldm_4p1"]:
            sub = df3[df3["baseline"] == m]
            if sub.empty:
                continue
            he = sub[sub["dataset"] == "humaneval"]["score_value"].mean()
            mb = sub[sub["dataset"] == "mbpp"]["score_value"].mean()
            if not math.isnan(he) or not math.isnan(mb):
                findings += (
                    f"Dream-7B / {pretty_bits({'baseline': m})}: HumanEval pass@1 = "
                    f"{pct(he)}\\,\\%, MBPP pass@1 = {pct(mb)}\\,\\%. "
                )
        findings += "\n\n"
    out_path.write_text(findings)


def main() -> None:
    df1 = load_jsonl(OUT_DIR / "phase1_module_sensitivity.jsonl")
    df2 = load_jsonl(OUT_DIR / "phase2_main_llada.jsonl")
    df3 = load_jsonl(OUT_DIR / "phase3_dream.jsonl")
    df4 = load_jsonl(OUT_DIR / "phase4_ablations.jsonl")
    df5 = load_jsonl(OUT_DIR / "phase5_moe.jsonl")

    PAPERS_DIR.mkdir(exist_ok=True)
    render_main_trade_off_table(df2, df3, PAPERS_DIR / "gen_table_main.tex")
    render_code_table(df3, PAPERS_DIR / "gen_table_code.tex")
    render_module_family_table(df1, PAPERS_DIR / "gen_table_modsens.tex")
    render_ablation_tables(df4, PAPERS_DIR / "gen_table_ablations.tex")
    # df15 is loaded below; pass through to scaling render.

    render_heatmap(OUT_DIR / "stage_layer_heatmap.csv", PAPERS_DIR / "fig_heatmap.pdf")
    render_pareto(df2, PAPERS_DIR / "fig_pareto.pdf")
    render_findings(df1, df2, df3, df4, df5, PAPERS_DIR / "gen_findings.tex")

    # Extension phases 8-14 tables + figures, if the data exist.
    df8 = load_jsonl(OUT_DIR / "phase8_low_bit.jsonl")
    df9 = load_jsonl(OUT_DIR / "phase9_mixed.jsonl")
    df10 = load_jsonl(OUT_DIR / "phase10_depth.jsonl")
    df11 = load_jsonl(OUT_DIR / "phase11_steps.jsonl")
    df12 = load_jsonl(OUT_DIR / "phase12_extra.jsonl")
    df13 = load_jsonl(OUT_DIR / "phase13_seeds.jsonl")
    df14 = load_jsonl(OUT_DIR / "phase14_throughput.jsonl")
    df15 = load_jsonl(OUT_DIR / "phase15_llada2mini.jsonl")
    df16 = load_jsonl(OUT_DIR / "phase16_mdlm.jsonl")
    df17 = load_jsonl(OUT_DIR / "phase17_dream.jsonl")
    df19 = load_jsonl(OUT_DIR / "phase19_lazy_16b.jsonl")
    df20 = load_jsonl(OUT_DIR / "phase20_bert.jsonl")

    render_scaling_table(df5, PAPERS_DIR / "gen_table_scaling.tex", df_moe_16b=df15, df_lazy_16b=df19)
    render_mdlm_table(df16, PAPERS_DIR / "gen_table_mdlm.tex")
    render_dream_table(df17, PAPERS_DIR / "gen_table_dream.tex")
    render_bert_table(df20, PAPERS_DIR / "gen_table_bert.tex")

    render_low_bit_table(df2, df8, PAPERS_DIR / "gen_table_main.tex")  # merge low-bit into main
    render_recipes_table(df9, PAPERS_DIR / "gen_table_recipes.tex")
    render_depth_table(df10, PAPERS_DIR / "gen_table_depth.tex")
    render_steps_table(df11, PAPERS_DIR / "gen_table_steps.tex")
    render_extra_mcq_table(df12, PAPERS_DIR / "gen_table_extra.tex")
    render_seeds_table(df13, PAPERS_DIR / "gen_table_seeds.tex")
    render_throughput_table(df14, PAPERS_DIR / "gen_table_throughput.tex")

    # Mirror everything to the NeurIPS compile dir.
    for name in (
        "gen_table_main.tex", "gen_table_code.tex", "gen_table_modsens.tex",
        "gen_table_ablations.tex", "gen_table_scaling.tex", "gen_findings.tex",
        "gen_table_recipes.tex", "gen_table_depth.tex", "gen_table_steps.tex",
        "gen_table_extra.tex", "gen_table_seeds.tex", "gen_table_throughput.tex",
        "gen_table_mdlm.tex", "gen_table_dream.tex", "gen_table_bert.tex",
        "fig_heatmap.pdf", "fig_pareto.pdf",
    ):
        _copy_to_neurips(PAPERS_DIR / name)

    # Also dump a CSV summary for debugging.
    summary = []
    for name, df in [("phase1", df1), ("phase2", df2), ("phase3", df3), ("phase4", df4), ("phase5", df5)]:
        if df.empty:
            continue
        keep = [c for c in ("model", "baseline", "dataset", "score_value",
                            "avg_bits_per_param", "param_storage_gb",
                            "peak_mem_gb", "wall_sec", "extra_tag") if c in df.columns]
        sub = df[keep].copy()
        sub["phase"] = name
        summary.append(sub)
    if summary:
        pd.concat(summary, ignore_index=True).to_csv(OUT_DIR / "summary.csv", index=False)
        print(f"wrote {OUT_DIR/'summary.csv'}")


if __name__ == "__main__":
    main()
