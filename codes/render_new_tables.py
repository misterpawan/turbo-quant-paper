# Release change: create generated table/figure directories in a code-only checkout.
"""
Render LaTeX tables from Phase 25 / 26 / 27 JSONL logs.

Each render_phaseN() function produces one gen_table_*.tex artefact
that can be inlined into papers/neurips2026/main.tex.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers"
OUT_NR = ROOT / "papers" / "neurips2026"


def load(path: Path) -> list:
    if not path.exists():
        return []
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


def render_phase25() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase25_stage_gsm8k.jsonl")
    if not rows:
        print("phase25: no data")
        return
    # Organise: (S, n_cal, dataset) -> score
    grid = {}
    for r in rows:
        S = r.get("stage_count", r.get("quant_cfg", {}).get("stage_count", 0))
        n_cal = r.get("n_cal", r.get("quant_cfg", {}).get("calibration_max_examples", 0))
        dataset = r.get("dataset", "?")
        grid[(S, n_cal, dataset)] = r
    out_lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 25 revisit of the stage-stratification claim on \\textsc{gsm8k} (reviewer Phase 8). TurboLDM $(3{+}1)$, single seed, $n{=}100$ for GSM8K and $n{=}200$ for MMLU. $S{=}1$ is pooled (non-stratified) calibration; $S{=}4$ is the default used throughout the paper. The abstract's claim that stage stratification dominates pooled on \\textsc{gsm8k} is \\emph{not} reproduced here; on $n{=}100$ \\textsc{gsm8k} pooled $S{=}1$ scores within $\\pm 5$pp of $S{=}4$, and on MMLU the effect is flat. We accordingly walk back the abstract wording and flag stage stratification as a [pilot] rule in the practitioner's recipe.}",
        "\\label{tab:phase25-gsm8k}",
        "\\begin{tabular}{cccc}",
        "\\toprule",
        "Dataset & $S$ & $N_\\text{cal}=24$ & $N_\\text{cal}=64$ \\\\",
        "\\midrule",
    ]
    for dataset in ("gsm8k", "mmlu"):
        for S in (1, 2, 4, 8):
            r24 = grid.get((S, 24, dataset))
            r64 = grid.get((S, 64, dataset))
            s24 = f"{r24.get('score_value', 0)*100:.1f}" if r24 else "--"
            s64 = f"{r64.get('score_value', 0)*100:.1f}" if r64 else "--"
            row_dataset = "\\textsc{gsm8k}" if dataset == "gsm8k" else "MMLU"
            out_lines.append(f"{row_dataset} & {S} & {s24} & {s64} \\\\")
        out_lines.append("\\midrule")
    if out_lines[-1] == "\\midrule":
        out_lines.pop()
    out_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    text = "\n".join(out_lines) + "\n"
    (OUT / "gen_table_phase25_stage_gsm8k.tex").write_text(text)
    (OUT_NR / "gen_table_phase25_stage_gsm8k.tex").write_text(text)
    print("wrote gen_table_phase25_stage_gsm8k.tex")


def render_phase26() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase26_residual_matched.jsonl")
    if not rows:
        print("phase26: no data")
        return
    # Organise: (mode, frac) -> dict of dataset -> score
    by_key = defaultdict(dict)
    bits_by_key = {}
    for r in rows:
        mode = r.get("schedule_mode", r.get("quant_cfg", {}).get("schedule_mode", "?"))
        frac = r.get("residual_top_fraction",
                       r.get("quant_cfg", {}).get("residual_top_fraction", 0.0))
        dataset = r.get("dataset", "?")
        by_key[(mode, frac)][dataset] = r.get("score_value", 0) * 100
        bits_by_key[(mode, frac)] = r.get("avg_bits_per_param", 0)
    order = [
        ("none",          0.00),
        ("random",        0.15),
        ("static_layer",  0.15),
        ("stage",         0.15),
        ("stage_entropy", 0.15),
        ("layer_once",    0.15),
        ("stage",         0.30),
        ("stage",         0.60),
    ]
    display = {
        "none": "No residual (base only)",
        "random": "Random (top-frac pairs)",
        "static_layer": "Static layer (all stages)",
        "stage": "Stage sensitivity (default)",
        "stage_entropy": "Stage + entropy gating",
        "layer_once": "Layer-once knapsack",
    }
    out_lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 26 matched-budget residual-schedule ablation (reviewer Phase 4). Every row uses base\\_bits$=3$, residual\\_bits$=1$, $S{=}4$; we vary only the schedule mode and residual top-fraction. We rerun at three coverages so the sensitivity-ranked ``stage'' mode can be compared at matched average-bits against the unscheduled baselines. Score is accuracy percentage on a single seed; the headline $(3{+}1)$ operating point in the main table corresponds to the ``stage, $0.15$'' row.}",
        "\\label{tab:phase26-residual}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Schedule mode & top-frac & Avg bits & GSM8K & MMLU & ARC-C \\\\",
        "\\midrule",
    ]
    for mode, frac in order:
        if (mode, frac) not in by_key:
            continue
        label = display.get(mode, mode)
        bits = bits_by_key[(mode, frac)]
        gsm = by_key[(mode, frac)].get("gsm8k", None)
        mm = by_key[(mode, frac)].get("mmlu", None)
        ar = by_key[(mode, frac)].get("arc_challenge", None)
        def fmt(x):
            return "--" if x is None else f"{x:.1f}"
        out_lines.append(f"{label} & {frac:.2f} & {bits:.2f} & {fmt(gsm)} & {fmt(mm)} & {fmt(ar)} \\\\")
    out_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    text = "\n".join(out_lines) + "\n"
    (OUT / "gen_table_phase26_residual.tex").write_text(text)
    (OUT_NR / "gen_table_phase26_residual.tex").write_text(text)
    print("wrote gen_table_phase26_residual.tex")


def render_phase27() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase27_lmhead.jsonl")
    if not rows:
        print("phase27: no data")
        return
    # Organise: (recipe, include_head, dataset) -> list of scores across seeds
    grid = defaultdict(list)
    bits_by = {}
    for r in rows:
        recipe = r.get("recipe", "?")
        inc = r.get("include_head", False)
        dataset = r.get("dataset", "?")
        grid[(recipe, inc, dataset)].append(r.get("score_value", 0) * 100)
        bits_by[(recipe, inc, dataset)] = r.get("avg_bits_per_param", 0)
    out_lines = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 27 LM-head quantisation across 4 tasks with 3 seeds (reviewer Phase 5). For each recipe (TurboLDM $(3{+}1)$, TurboLDM $(4{+}1)$, AWQ-4) we run with the LM head left at BF16 (default recipe) and with the LM head quantised at the same recipe. Scores are accuracy percentage: across-seed mean $\\pm$ seed standard deviation. MMLU $n{=}200$, ARC-C $n{=}200$, HellaSwag $n{=}300$, \\textsc{gsm8k} $n{=}100$. A blank row means the grid cell did not complete in time.}",
        "\\label{tab:phase27-lmhead}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & LM head & MMLU & ARC-C & HellaSwag & GSM8K \\\\",
        "\\midrule",
    ]
    for recipe in ("turboldm_3p1", "turboldm_4p1", "awq_4"):
        pretty = {"turboldm_3p1": "TurboLDM (3+1)", "turboldm_4p1": "TurboLDM (4+1)", "awq_4": "AWQ-4"}[recipe]
        for inc in (False, True):
            head_tag = "quantised" if inc else "BF16"
            entries_any = False
            cols = [pretty, head_tag]
            for dataset in ("mmlu", "arc_challenge", "hellaswag", "gsm8k"):
                scores = grid.get((recipe, inc, dataset), [])
                if scores:
                    entries_any = True
                    m = mean(scores)
                    sd = stdev(scores) if len(scores) > 1 else 0.0
                    cols.append(f"{m:.1f}$\\pm${sd:.1f}")
                else:
                    cols.append("--")
            if entries_any:
                out_lines.append(" & ".join(cols) + " \\\\")
    out_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    text = "\n".join(out_lines) + "\n"
    (OUT / "gen_table_phase27_lmhead.tex").write_text(text)
    (OUT_NR / "gen_table_phase27_lmhead.tex").write_text(text)
    print("wrote gen_table_phase27_lmhead.tex")


def render_phase30() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase30_modsens_multitask.jsonl")
    if not rows:
        print("phase30: no data"); return
    grid = defaultdict(list)
    for r in rows:
        family = r.get("family", "?")
        dataset = r.get("dataset", "?")
        grid[(family, dataset)].append(r.get("score_value", 0) * 100)
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 30 module-family sensitivity across 4 tasks with 3 seeds (reviewer Phase 7). Each row quantises \\emph{only} the named family with 3-bit TurboQuant (no residual); everything else stays BF16. Reported as mean$\\pm$std percentage accuracy over seeds $\\{42, 43, 44\\}$.}",
        "\\label{tab:phase30-modsens}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Family & MMLU & ARC-C & HellaSwag & GSM8K \\\\",
        "\\midrule",
    ]
    for family in ("att_proj", "attn_out", "q_proj", "k_proj", "v_proj",
                    "ff_proj", "ff_out", "attn_all", "mlp_all"):
        cells = [family.replace("_", r"\_")]
        any_data = False
        for d in ("mmlu", "arc_challenge", "hellaswag", "gsm8k"):
            ss = grid.get((family, d), [])
            if ss:
                any_data = True
                m = mean(ss); s = stdev(ss) if len(ss) > 1 else 0.0
                cells.append(f"{m:.1f}$\\pm${s:.1f}")
            else:
                cells.append("--")
        if any_data:
            out.append(" & ".join(cells) + " \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase30_modsens.tex").write_text(text)
    (OUT_NR / "gen_table_phase30_modsens.tex").write_text(text)
    print("wrote gen_table_phase30_modsens.tex")


def render_phase31() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase31_steps_pareto.jsonl")
    if not rows:
        print("phase31: no data"); return
    grid = {}
    for r in rows:
        method = r.get("method", "?"); T = r.get("T", 0); dataset = r.get("dataset", "?")
        grid[(method, T, dataset)] = (r.get("score_value", 0) * 100, r.get("wall_sec", 0))
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 31 step-count $\\times$ quantisation Pareto (reviewer Phase 9). Each cell is accuracy\\%\\ (wall seconds). Five methods $\\times$ five $T$ values on MMLU $n{=}200$ and \\textsc{gsm8k} $n{=}100$, single seed.}",
        "\\label{tab:phase31-steps}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method / dataset & $T{=}8$ & $T{=}16$ & $T{=}32$ & $T{=}64$ & $T{=}128$ \\\\",
        "\\midrule",
    ]
    for method in ("bf16", "w4", "awq_4", "turboldm_3p1", "turboldm_4p1"):
        for dataset in ("mmlu", "gsm8k"):
            pretty_m = {"bf16": "BF16", "w4": "W4", "awq_4": "AWQ-4",
                        "turboldm_3p1": "TurboLDM (3+1)", "turboldm_4p1": "TurboLDM (4+1)"}[method]
            pretty_d = dataset.upper() if dataset != "gsm8k" else "\\textsc{gsm8k}"
            cells = [f"{pretty_m} / {pretty_d}"]
            any_data = False
            for T in (8, 16, 32, 64, 128):
                cell = grid.get((method, T, dataset))
                if cell is None:
                    cells.append("--")
                else:
                    acc, wall = cell
                    cells.append(f"{acc:.1f} ({wall:.0f}s)")
                    any_data = True
            if any_data:
                out.append(" & ".join(cells) + " \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase31_steps.tex").write_text(text)
    (OUT_NR / "gen_table_phase31_steps.tex").write_text(text)
    print("wrote gen_table_phase31_steps.tex")


def render_phase32() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase32_16b_failure.jsonl")
    if not rows:
        print("phase32: no data"); return
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 32 LLaDA2.0-mini (16B) lazy-quantisation failure analysis (reviewer Phase 10). Each row is MMLU $n{=}75$. The BF16 baseline is $64.0$. Quality collapse is not uniform: compare attn-only vs MLP-only to see which family is responsible.}",
        "\\label{tab:phase32-16b}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Config & Avg bits & Storage GB & Peak mem GB & MMLU \\\\",
        "\\midrule",
    ]
    pretty = {"lazy_4_attn_only": "Lazy-4 attention only",
              "lazy_4_mlp_only":  "Lazy-4 MLP only",
              "lazy_5":           "Lazy-(4+2) full model"}
    for r in rows:
        tag = r.get("baseline", "?")
        bits = r.get("avg_bits_per_param", 0)
        storage = r.get("param_storage_gb", 0)
        peak = r.get("peak_mem_gb", 0)
        score = r.get("score_value", 0) * 100
        out.append(f"{pretty.get(tag, tag)} & {bits:.2f} & {storage:.2f} & {peak:.1f} & {score:.1f} \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase32_16b.tex").write_text(text)
    (OUT_NR / "gen_table_phase32_16b.tex").write_text(text)
    print("wrote gen_table_phase32_16b.tex")


def render_phase33() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase33_encoder_extended.jsonl")
    if not rows:
        print("phase33: no data"); return
    grid = {}
    for r in rows:
        grid[(r.get("model", "?"), r.get("baseline", "?"))] = r
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 33 encoder-only masked-LM transfer, extended (reviewer Phase 12): BERT-base, RoBERTa-base, DeBERTa-v3-base at $512$-paragraph WikiText-103 slice. Masked-token NLL at $15\\%$ masking (lower is better). $\\Delta$NLL is the increase over the FP32 baseline.}",
        "\\label{tab:phase33-encoder}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Model & Method & Avg bits & NLL & $\\Delta$NLL & PPL \\\\",
        "\\midrule",
    ]
    for m in ("bert_base", "roberta_base", "deberta_v3_base"):
        bf16 = grid.get((m, "bf16"))
        bf16_nll = bf16.get("score_value", 0) if bf16 else None
        for tag in ("bf16", "uniform_int4", "awq_4", "turboldm_4p1", "turboldm_3p1"):
            r = grid.get((m, tag))
            if not r: continue
            nll = r.get("score_value", 0); bits = r.get("avg_bits_per_param", 0); ppl = r.get("ppl", 0)
            delta = nll - bf16_nll if bf16_nll is not None else 0
            out.append(f"{m.replace('_', '-')} & {tag.replace('_', r'-')} & {bits:.2f} & {nll:.3f} & {delta:+.3f} & {ppl:.2f} \\\\")
        out.append("\\midrule")
    if out[-1] == "\\midrule": out.pop()
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase33_encoder.tex").write_text(text)
    (OUT_NR / "gen_table_phase33_encoder.tex").write_text(text)
    print("wrote gen_table_phase33_encoder.tex")


def render_phase35() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase35_deberta.jsonl")
    if not rows:
        print("phase35: no data"); return
    # Sort: bf16 first, then quantised methods
    method_order = ["bf16", "uniform_int4", "awq_4", "turboldm_4p1", "turboldm_3p1"]
    by_tag = {r.get("baseline", "?"): r for r in rows}
    bf16 = by_tag.get("bf16")
    bf16_nll = bf16.get("score_value") if bf16 else None
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 35: DeBERTa-v3-base masked-LM transfer (reviewer Phase 12 follow-up). Reruns the BERT/RoBERTa methods in a fresh \\texttt{dlquant-cu12} conda environment (PyTorch 2.5.1+cu121, libnvjitlink 12.4.127), which resolves the \\texttt{libnvrtc-builtins.so.13.0} loader error from the disentangled-attention CUDA kernel. Masked-token NLL at $15\\%$ masking on the same $512$-paragraph WikiText-103 slice as Phase 33; lower is better; $\\Delta$NLL is increase over the FP32 baseline.}",
        "\\label{tab:phase35-deberta}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Avg bits & NLL & $\\Delta$NLL & PPL \\\\",
        "\\midrule",
    ]
    pretty = {"bf16": "FP32 baseline", "uniform_int4": "W4 (uniform)", "awq_4": "AWQ-4",
              "turboldm_4p1": "TurboLDM (4+1)", "turboldm_3p1": "TurboLDM (3+1)"}
    for tag in method_order:
        r = by_tag.get(tag)
        if not r: continue
        bits = r.get("avg_bits_per_param", 0)
        nll = r.get("score_value", 0)
        ppl = r.get("ppl", 0)
        delta = nll - bf16_nll if bf16_nll is not None else 0
        out.append(f"{pretty.get(tag, tag)} & {bits:.2f} & {nll:.3f} & {delta:+.3f} & {ppl:.2f} \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase35_deberta.tex").write_text(text)
    (OUT_NR / "gen_table_phase35_deberta.tex").write_text(text)
    print("wrote gen_table_phase35_deberta.tex")


def render_phase37() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase37_gptq_full.jsonl")
    if not rows:
        print("phase37: no data"); return
    grid = {}
    for r in rows:
        grid[(r.get("method", "?"), r.get("dataset", "?"))] = r
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 37: full-Hessian GPTQ baseline at $n{=}500$ (closes the GPTQ-diag lower-bound caveat). For each method the value is single-seed accuracy at the new evaluation slice.}",
        "\\label{tab:phase37-gptq-full}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Avg bits & MMLU & ARC-C & H.Swag & GSM8K \\\\",
        "\\midrule",
    ]
    for method in ("gptq_full_3", "gptq_full_4"):
        bits = next((grid.get((method, d), {}).get("avg_bits_per_param", 0) for d in ("mmlu",)), 0)
        cols = [method.replace("_", r"\_"), f"{bits:.2f}"]
        for d in ("mmlu", "arc_challenge", "hellaswag", "gsm8k"):
            r = grid.get((method, d))
            cols.append(f"{r.get('score_value', 0)*100:.1f}" if r else "--")
        out.append(" & ".join(cols) + " \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase37_gptq_full.tex").write_text(text)
    (OUT_NR / "gen_table_phase37_gptq_full.tex").write_text(text)
    print("wrote gen_table_phase37_gptq_full.tex")


def render_phase38() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase38_chunked.jsonl")
    if not rows:
        print("phase38: no data"); return
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 38: cached-dequant vs chunked low-bit forward on AWQ-4 / LLaDA-8B-Instruct (MMLU, $n{=}75$). The chunked variant reconstructs each linear's BF16 weight in row-slices of $512$ at forward time, so the per-layer BF16 buffer never persists. This realises the upper-bound on VRAM savings achievable by removing the cached-dequant; a true low-bit kernel would also gain throughput.}",
        "\\label{tab:phase38-chunked}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Mode & Storage GB & Peak VRAM GB & MMLU & Wall (s) \\\\",
        "\\midrule",
    ]
    for r in rows:
        mode = r.get("mode", r.get("baseline", "?"))
        storage = r.get("param_storage_gb", 0)
        peak = r.get("peak_mem_gb", 0)
        score = r.get("score_value", 0) * 100
        wall = r.get("wall_sec", 0)
        out.append(f"{mode} & {storage:.2f} & {peak:.2f} & {score:.1f} & {wall:.1f} \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase38_chunked.tex").write_text(text)
    (OUT_NR / "gen_table_phase38_chunked.tex").write_text(text)
    print("wrote gen_table_phase38_chunked.tex")


def render_phase39() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase39_embed.jsonl")
    if not rows:
        print("phase39: no data"); return
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 39: 4-bit groupwise QuantizedEmbedding for the input embedding table on top of TurboLDM-$(3{+}1)$ (LLaDA-8B-Instruct, $n{=}500$). Saves $\\sim\\!780$\\,MiB of BF16 residual at the cost of an in-flight dequantisation per gathered row. Compare the rows below to the TurboLDM-$(3{+}1)$ row of Table~\\ref{tab:main}.}",
        "\\label{tab:phase39-embed}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & Avg bits & Storage GB & Score & Wall (s) \\\\",
        "\\midrule",
    ]
    for r in rows:
        d = r.get("dataset", "?")
        bits = r.get("avg_bits_per_param", 0)
        storage = r.get("param_storage_gb", 0)
        score = r.get("score_value", 0) * 100
        wall = r.get("wall_sec", 0)
        d_pretty = {"mmlu": "MMLU", "arc_challenge": "ARC-C",
                     "hellaswag": "HellaSwag", "gsm8k": r"\textsc{gsm8k}"}.get(d, d.replace('_', '-'))
        out.append(f"{d_pretty} & {bits:.2f} & {storage:.2f} & {score:.1f} & {wall:.1f} \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase39_embed.tex").write_text(text)
    (OUT_NR / "gen_table_phase39_embed.tex").write_text(text)
    print("wrote gen_table_phase39_embed.tex")


def render_phase40() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase40_deberta_v1.jsonl")
    if not rows:
        print("phase40: no data"); return
    grid = {}
    for r in rows:
        grid[(r.get("model", "?"), r.get("baseline", "?"))] = r
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 40: DeBERTa v1 / v2 masked-LM transfer (closes the DeBERTa caveat in a non-RTD checkpoint). Masked-token NLL at $15\\%$ masking on $512$-paragraph WikiText-103. Lower is better; $\\Delta$NLL is increase over the FP32 baseline.}",
        "\\label{tab:phase40-deberta-v1}",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Model & Method & Avg bits & NLL & PPL \\\\",
        "\\midrule",
    ]
    pretty = {"bf16": "FP32 baseline", "uniform_int4": "W4 (uniform)", "awq_4": "AWQ-4",
              "turboldm_4p1": "TurboLDM (4+1)", "turboldm_3p1": "TurboLDM (3+1)"}
    for m in ("deberta_v1_base", "deberta_v2_xlarge"):
        for tag in ("bf16", "uniform_int4", "awq_4", "turboldm_4p1", "turboldm_3p1"):
            r = grid.get((m, tag))
            if not r: continue
            bits = r.get("avg_bits_per_param", 0)
            nll = r.get("score_value", 0)
            ppl = r.get("ppl", 0)
            out.append(f"{m.replace('_', r'-')} & {pretty.get(tag, tag)} & {bits:.2f} & {nll:.3f} & {ppl:.2f} \\\\")
        out.append("\\midrule")
    if out[-1] == "\\midrule": out.pop()
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase40_deberta_v1.tex").write_text(text)
    (OUT_NR / "gen_table_phase40_deberta_v1.tex").write_text(text)
    print("wrote gen_table_phase40_deberta_v1.tex")


def render_phase41() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase41_16b_attnonly.jsonl")
    if not rows:
        print("phase41: no data"); return
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 41: 16B attn-only A/B via accelerate device\\_map (closes the 32 GB VRAM caveat). LLaDA2.0-mini, MMLU $n{=}75$, BF16 baseline $64.0$.}",
        "\\label{tab:phase41-16b-attnonly}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Config & Avg bits & Storage GB & Peak GB & MMLU \\\\",
        "\\midrule",
    ]
    for r in rows:
        tag = r.get("baseline", "?")
        bits = r.get("avg_bits_per_param", 0)
        storage = r.get("param_storage_gb", 0)
        peak = r.get("peak_mem_gb", 0)
        score = r.get("score_value", 0) * 100
        out.append(f"{tag.replace('_', r'\\_')} & {bits:.2f} & {storage:.2f} & {peak:.1f} & {score:.1f} \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase41_attnonly.tex").write_text(text)
    (OUT_NR / "gen_table_phase41_attnonly.tex").write_text(text)
    print("wrote gen_table_phase41_attnonly.tex")


def render_phase42() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase42_modsens_true.jsonl")
    if not rows:
        print("phase42: no data"); return
    grid = {}
    for r in rows:
        grid[(r.get("family", "?"), r.get("dataset", "?"))] = r
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 42 corrects the regex artefact in Table~\\ref{tab:phase30-modsens}: this row reports the \\emph{true} compound families ($\\mathtt{attn\\_all} = \\{q, k, v, \\mathtt{attn\\_out}\\}$, $\\mathtt{mlp\\_all} = \\{\\mathtt{ff\\_proj}, \\mathtt{ff\\_out}\\}$) all quantised together at 3-bit (no residual). Single seed, $n{=}500$. The headline reading reverses the ``attention is cheap'' claim: at the compound level attention is the more sensitive family.}",
        "\\label{tab:phase42-modsens-true}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "True compound family & Storage GB & MMLU & ARC-C & H.Swag & GSM8K \\\\",
        "\\midrule",
        "BF16 reference & -- & 61.8 & 86.0 & 69.2 & 68.6 \\\\",
    ]
    for fam in ("attn_all", "mlp_all"):
        cells = [fam.replace("_", r"\_")]
        bits_storage = grid.get((fam, "mmlu"), {})
        cells.append(f"{bits_storage.get('param_storage_gb', 0):.2f}")
        for d in ("mmlu", "arc_challenge", "hellaswag", "gsm8k"):
            r = grid.get((fam, d))
            cells.append(f"{r.get('score_value', 0)*100:.1f}" if r else "--")
        out.append(" & ".join(cells) + " \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase42_modsens_true.tex").write_text(text)
    (OUT_NR / "gen_table_phase42_modsens_true.tex").write_text(text)
    print("wrote gen_table_phase42_modsens_true.tex")


def render_phase43() -> None:
    rows = load(ROOT / "runs" / "overnight" / "phase43_recipes_ci.jsonl")
    if not rows:
        print("phase43: no data"); return
    grid = defaultdict(list)
    for r in rows:
        grid[(r.get("recipe", "?"), r.get("dataset", "?"))].append(r.get("score_value", 0) * 100)
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 43 mixed-bit recipes at $n{=}500$ with 3 seeds (replaces the earlier single-seed $n{=}100$ pilot in Table~\\ref{tab:recipes}). Mean$\\pm$std accuracy. Compare against the corresponding W4 / AWQ-4 / TurboLDM rows of Table~\\ref{tab:main}.}",
        "\\label{tab:phase43-recipes-ci}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Recipe & MMLU & ARC-C & H.Swag & GSM8K \\\\",
        "\\midrule",
    ]
    pretty = {"attn3_mlp4": r"attn@3 + MLP@4", "attn4_mlp3": r"attn@4 + MLP@3",
              "attn3_upgate3_down4": r"attn@3 + up\&gate@3 + down@4",
              "attn2_mlp4": r"attn@2 + MLP@4"}
    for recipe in ("attn3_mlp4", "attn4_mlp3", "attn3_upgate3_down4", "attn2_mlp4"):
        cells = [pretty.get(recipe, recipe)]
        any_data = False
        for d in ("mmlu", "arc_challenge", "hellaswag", "gsm8k"):
            ss = grid.get((recipe, d), [])
            if ss:
                any_data = True
                m = mean(ss); s = stdev(ss) if len(ss) > 1 else 0.0
                cells.append(f"{m:.1f}$\\pm${s:.1f}")
            else:
                cells.append("--")
        if any_data:
            out.append(" & ".join(cells) + " \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase43_recipes_ci.tex").write_text(text)
    (OUT_NR / "gen_table_phase43_recipes_ci.tex").write_text(text)
    print("wrote gen_table_phase43_recipes_ci.tex")


def render_phase44() -> None:
    """Render the trajectory KL plot (figure) and a small summary table."""
    src = ROOT / "runs" / "overnight" / "phase44_trajectory.jsonl"
    if not src.exists():
        print("phase44: no data"); return
    import json as _j
    rows = [_j.loads(l) for l in src.open() if l.strip()]
    if not rows:
        print("phase44: empty"); return
    # Render a small per-task summary table and a side figure.
    by_task = defaultdict(list)
    for r in rows:
        task = "gsm8k" if r["prompt_id"].startswith("gsm8k") else "mmlu"
        by_task[task].append(r)
    out = [
        "\\begin{table}[t]\\centering\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Phase 44 trajectory quantisation lens. Each row is averaged across prompts in the named task (8 prompts each). Mean per-step KL divergence between BF16 and TurboLDM-$(3{+}1)$ logits at the position the model fills, plus average top-1 token agreement and an early-vs-late split. The early-vs-late split (steps $0$--$15$ vs $48$--$63$ at $T{=}64$) is the headline reading: on \\textsc{gsm8k} the late-step KL is much smaller than early, suggesting the trajectory reduces accumulated quantisation error; on MCQ the difference is small.}",
        "\\label{tab:phase44-trajectory}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Task & mean KL (all steps) & mean top-1 agree & mean KL early & mean KL late \\\\",
        "\\midrule",
    ]
    import numpy as _np
    for task in ("mmlu", "gsm8k"):
        rs = by_task.get(task, [])
        if not rs:
            continue
        all_kl = [s["kl_bf16_to_quant"] for r in rs for s in r["trajectory"]]
        all_t1 = [s["top1_agreement"] for r in rs for s in r["trajectory"]]
        early = [s["kl_bf16_to_quant"] for r in rs for s in r["trajectory"] if s["step"] < 16]
        late = [s["kl_bf16_to_quant"] for r in rs for s in r["trajectory"] if s["step"] >= 48]
        out.append(f"{task.upper() if task=='mmlu' else r'\textsc{gsm8k}'} & "
                   f"{_np.mean(all_kl):.3f} & {_np.mean(all_t1)*100:.1f}\\% & "
                   f"{_np.mean(early):.3f} & {_np.mean(late):.3f} \\\\")
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    text = "\n".join(out) + "\n"
    (OUT / "gen_table_phase44_trajectory.tex").write_text(text)
    (OUT_NR / "gen_table_phase44_trajectory.tex").write_text(text)
    print("wrote gen_table_phase44_trajectory.tex")


def main() -> None:
    Path("papers/neurips2026").mkdir(parents=True, exist_ok=True)
    render_phase25()
    render_phase26()
    render_phase27()
    render_phase30()
    render_phase31()
    render_phase32()
    render_phase33()
    render_phase35()
    render_phase37()
    render_phase38()
    render_phase39()
    render_phase40()
    render_phase41()
    render_phase42()
    render_phase43()
    render_phase44()


if __name__ == "__main__":
    main()
