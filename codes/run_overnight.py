"""
TurboLDM overnight experiment driver.

Runs Phases 1–5 sequentially. Each completed row is appended to JSONL so
the run is resumable. Checkpoints are shared across datasets for a given
(model, method) pair to avoid re-loading / re-quantizing the model.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import TurboLDM_experiments_helper as H  # noqa: E402
from TurboLDM_experiments_helper import (  # noqa: E402
    GenerateConfig,
    MODEL_SPECS,
    QuantConfig,
    RunConfig,
    ResultRecord,
    avg_bits_per_param,
    default_generate_config_for_model,
    evaluate_dataset,
    iter_named_linears,
    load_model_and_tokenizer,
    maybe_reset_cuda_peak,
    model_parameter_storage_bytes,
    prepare_stage_batches_for_model,
    quantize_model_inplace,
    seed_everything,
    set_matmul_precision,
    compute_stage_layer_scores,
)


# ---------- Slice sizes chosen to fit 12 h wall clock ----------
SLICE_SIZES = {
    "mmlu": 150,
    "arc_challenge": 200,
    "hellaswag": 200,
    "piqa": 200,
    "gsm8k": 75,
    "humaneval": 40,
    "mbpp": 80,
}

# Smaller slice for Phase 1/4 where we do many methods and only care
# about relative deltas.
SMALL_SLICE_SIZES = {**SLICE_SIZES, "mmlu": 100, "gsm8k": 40, "humaneval": 30, "mbpp": 40}

OUT_DIR = Path("runs/overnight")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "logs").mkdir(exist_ok=True)
MASTER_LOG = OUT_DIR / "overnight.log"


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with MASTER_LOG.open("a") as f:
        f.write(line + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_done_keys(jsonl_path: Path) -> set:
    """Resume helper: return set of (model, baseline, dataset, tag) already logged."""
    done = set()
    if not jsonl_path.exists():
        return done
    with jsonl_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                done.add((row.get("model"), row.get("baseline"), row.get("dataset"), row.get("extra_tag", "")))
            except Exception:
                continue
    return done


def mcq_fast_gen_cfg(spec, dataset: str) -> GenerateConfig:
    """A faster gen config than the helper default: keep gen_length short for MCQ and
    drop diffusion steps for MCQ / QA because the answer is tiny."""
    cfg = default_generate_config_for_model(spec, dataset)
    if dataset in ("mmlu", "arc_challenge", "hellaswag", "piqa"):
        cfg.steps = 16
        cfg.gen_length = 8
        cfg.block_length = 8
        cfg.batch_size = 2
        # No stop strings for MCQ — `parse_choice_answer` is robust to whitespace
        # and Dream-7B's chat-template output starts with a newline that would
        # otherwise be cut.
        cfg.stop_strings = tuple()
    elif dataset == "gsm8k":
        cfg.steps = 64
        cfg.gen_length = 128
        cfg.block_length = 32
        cfg.batch_size = 1
    elif dataset in ("humaneval", "mbpp"):
        cfg.steps = 64
        cfg.gen_length = 128  # 7-8B model on 32 GB can't do 256 safely
        cfg.block_length = 32
        cfg.batch_size = 1
    return cfg


def run_and_log(
    model,
    tokenizer,
    spec,
    method_tag: str,
    quant_cfg: QuantConfig,
    datasets: Sequence[str],
    jsonl_path: Path,
    slice_sizes: Dict[str, int],
    done_keys: set,
    extra_tag: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    storage_gb = model_parameter_storage_bytes(model) / (1024**3)
    bits = avg_bits_per_param(model)
    for ds in datasets:
        key = (spec.key, method_tag, ds, extra_tag)
        if key in done_keys:
            log(f"SKIP (resume) {spec.key} / {method_tag} / {ds} / {extra_tag}")
            continue
        n = slice_sizes.get(ds)
        gen_cfg = mcq_fast_gen_cfg(spec, ds)
        log(f"eval {spec.key} / {method_tag} / {ds} (n={n}, steps={gen_cfg.steps}, gen_len={gen_cfg.gen_length})")
        t0 = time.perf_counter()
        maybe_reset_cuda_peak()
        try:
            rec = evaluate_dataset(model=model, tokenizer=tokenizer, spec=spec, dataset_name=ds,
                                   gen_cfg=gen_cfg, max_examples=n, device="cuda")
            rec.baseline = method_tag
            rec.param_storage_gb = storage_gb
            rec.avg_bits_per_param = bits
            row = rec.to_dict()
            row["extra_tag"] = extra_tag
            row["wall_sec"] = time.perf_counter() - t0
            row["quant_cfg"] = asdict(quant_cfg)
            row["gen_cfg"] = asdict(gen_cfg)
            append_jsonl(jsonl_path, row)
            rows.append(row)
            log(f"  -> score={rec.score_value:.3f} mem={rec.peak_mem_gb:.1f}GB bits={bits:.2f} "
                f"storage={storage_gb:.2f}GB elapsed={row['wall_sec']:.1f}s")
        except Exception as exc:
            log(f"  !! FAILED {spec.key}/{method_tag}/{ds}: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {
                "model": spec.key, "baseline": method_tag, "dataset": ds,
                "extra_tag": extra_tag, "error": str(exc),
                "trace": traceback.format_exc(),
            })
    return rows


def build_method_model(spec, base_quant_cfg: QuantConfig) -> tuple:
    """Load a fresh base model and apply the given quantization."""
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    if base_quant_cfg.method in ("bf16", "baseline", "none"):
        return model, tok
    stage_batches = prepare_stage_batches_for_model(model, tok, spec, base_quant_cfg,
                                                    run_cfg=RunConfig(cache_dir="cache"))
    model = quantize_model_inplace(model, tok, spec, base_quant_cfg,
                                   run_cfg=RunConfig(cache_dir="cache"),
                                   stage_batches=stage_batches)
    del stage_batches
    clear_cuda()
    # Force-release any unreferenced layer weights left over from the old
    # Linear modules so inference memory stays flat.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tok


# ------------------------------------------------------------------
# Phase 1: module-family sensitivity on LLaDA-8B
# ------------------------------------------------------------------

MODULE_FAMILIES = {
    "q_proj":   r"q_proj$",
    "k_proj":   r"k_proj$",
    "v_proj":   r"v_proj$",
    "attn_out": r"(attn_out|o_proj|out_proj)$",
    "attn_all": r"(q_proj|k_proj|v_proj|attn_out|o_proj|out_proj)$",
    "mlp_gate": r"(ff_proj|gate_proj)$",
    "mlp_up":   r"up_proj$",
    "mlp_down": r"(ff_out|down_proj)$",
    "mlp_all":  r"(ff_proj|up_proj|ff_out|gate_proj|down_proj|fc1|fc2)$",
}


def phase1_module_sensitivity(spec_key: str = "llada_8b_instruct") -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase1_module_sensitivity.jsonl"
    done = load_done_keys(jsonl)
    log(f"=== Phase 1: module-family sensitivity on {spec_key} ===")

    # bf16 baseline
    if (spec_key, "bf16", "mmlu", "") not in done:
        clear_cuda()
        model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
        run_and_log(model, tok, spec, "bf16", QuantConfig(method="bf16"),
                    ["mmlu"], jsonl, SMALL_SLICE_SIZES, done)
        del model, tok; clear_cuda()

    # Each module family quantized alone with TurboLDM base 3-bit. Signal
    # strength matters: 3-bit provides the largest per-family quality deltas
    # because the base codec is intentionally aggressive.
    for fam, regex in MODULE_FAMILIES.items():
        qcfg = QuantConfig(
            method="turboldm", base_bits=3, residual_bits=1, block_size=128,
            stage_count=4, sensitivity_metric="hidden_mse",
            include_regex=regex, exclude_regex=r"(lm_head|embed|embedding|router)",
            residual_top_fraction=0.0,
            calibration_max_examples=24, probe_max_examples=3,
            calibration_max_tokens=192,
        )
        key = (spec_key, f"turboldm_only_{fam}", "mmlu", "")
        if key in done:
            log(f"SKIP phase1 fam={fam}")
            continue
        try:
            model, tok = build_method_model(spec, qcfg)
        except Exception as exc:
            log(f"  !! build failed for fam={fam}: {exc}")
            continue
        run_and_log(model, tok, spec, f"turboldm_only_{fam}", qcfg,
                    ["mmlu"], jsonl, SMALL_SLICE_SIZES, done)
        del model, tok; clear_cuda()

    # Stage x Layer heatmap using compute_stage_layer_scores (hidden_mse).
    heatmap_csv = OUT_DIR / "stage_layer_heatmap.csv"
    if not heatmap_csv.exists():
        clear_cuda()
        model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
        qcfg = QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                           stage_count=4, sensitivity_metric="hidden_mse",
                           calibration_max_examples=24, probe_max_examples=3,
                           calibration_max_tokens=192)
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        scores = compute_stage_layer_scores(model, stage_batches, qcfg)
        scores.to_csv(heatmap_csv, index=False)
        log(f"wrote heatmap csv with {len(scores)} rows -> {heatmap_csv}")
        del model, tok, stage_batches, scores; clear_cuda()


# ------------------------------------------------------------------
# Phase 2: main trade-off on LLaDA-8B
# ------------------------------------------------------------------
PHASE2_METHODS = [
    ("bf16", QuantConfig(method="bf16", w_bits=16)),
    ("uniform_int4", QuantConfig(method="uniform", w_bits=4, group_size=128)),
    ("gptq_4", QuantConfig(method="gptq", w_bits=4, group_size=128,
                            calibration_max_examples=32, probe_max_examples=4,
                            calibration_max_tokens=256)),
    ("awq_4", QuantConfig(method="awq", w_bits=4, group_size=128,
                           calibration_max_examples=32, probe_max_examples=4,
                           calibration_max_tokens=256)),
    ("spqr_4", QuantConfig(method="spqr", w_bits=4, group_size=128, outlier_fraction=0.001)),
    ("qdiffusion_text_4", QuantConfig(method="qdiffusion_text", w_bits=4, group_size=128,
                                       stage_count=4, sensitivity_metric="hidden_mse",
                                       calibration_max_examples=32, probe_max_examples=4,
                                       calibration_max_tokens=256)),
    ("turboldm_4p1", QuantConfig(method="turboldm", base_bits=4, residual_bits=1,
                                 block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                 residual_top_fraction=0.15, entropy_gating=False,
                                 calibration_max_examples=32, probe_max_examples=4,
                                 calibration_max_tokens=256)),
    ("turboldm_3p1", QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                 block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                 residual_top_fraction=0.15, entropy_gating=False,
                                 calibration_max_examples=32, probe_max_examples=4,
                                 calibration_max_tokens=256)),
]
PHASE2_DATASETS = ["mmlu", "arc_challenge", "hellaswag", "piqa", "gsm8k"]


def phase2_main_llada() -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    jsonl = OUT_DIR / "phase2_main_llada.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 2: main trade-off on LLaDA-8B ===")
    for tag, qcfg in PHASE2_METHODS:
        if all((spec.key, tag, ds, "") in done for ds in PHASE2_DATASETS):
            log(f"SKIP phase2 method={tag}")
            continue
        try:
            t0 = time.perf_counter()
            model, tok = build_method_model(spec, qcfg)
            log(f"  built {tag} in {time.perf_counter()-t0:.1f}s "
                f"bits={avg_bits_per_param(model):.2f} storage={model_parameter_storage_bytes(model)/1024**3:.2f}GB")
        except Exception as exc:
            log(f"  !! build failed {tag}: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl",
                         {"phase": 2, "model": spec.key, "tag": tag, "err": str(exc),
                          "trace": traceback.format_exc()})
            continue
        run_and_log(model, tok, spec, tag, qcfg, PHASE2_DATASETS, jsonl, SLICE_SIZES, done)
        del model, tok; clear_cuda()


# ------------------------------------------------------------------
# Phase 3: Dream-7B transfer + code
# ------------------------------------------------------------------
PHASE3_METHODS = [
    ("bf16", QuantConfig(method="bf16", w_bits=16)),
    ("awq_4", QuantConfig(method="awq", w_bits=4, group_size=128,
                           calibration_max_examples=32, probe_max_examples=4,
                           calibration_max_tokens=256)),
    ("qdiffusion_text_4", QuantConfig(method="qdiffusion_text", w_bits=4, group_size=128,
                                       stage_count=4, sensitivity_metric="hidden_mse",
                                       calibration_max_examples=32, probe_max_examples=4,
                                       calibration_max_tokens=256)),
    ("turboldm_4p1", QuantConfig(method="turboldm", base_bits=4, residual_bits=1,
                                 block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                 residual_top_fraction=0.15, entropy_gating=False,
                                 calibration_max_examples=32, probe_max_examples=4,
                                 calibration_max_tokens=256)),
]
# HumanEval/MBPP dataset scripts are broken on datasets 4.7; MBPP uses a script
# path entirely. HumanEval can be loaded but 7-8B inference with gen_length=256
# exceeds our 32 GB VRAM. We therefore restrict the Dream transfer study to
# MCQ + GSM8K, which is still an instructive cross-model check.
PHASE3_DATASETS = ["mmlu", "arc_challenge", "hellaswag", "gsm8k"]


def phase3_dream() -> None:
    spec = MODEL_SPECS["dream_7b_instruct"]
    jsonl = OUT_DIR / "phase3_dream.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 3: Dream-7B transfer + code ===")
    for tag, qcfg in PHASE3_METHODS:
        if all((spec.key, tag, ds, "") in done for ds in PHASE3_DATASETS):
            log(f"SKIP phase3 method={tag}")
            continue
        try:
            model, tok = build_method_model(spec, qcfg)
        except Exception as exc:
            log(f"  !! build failed Dream/{tag}: {exc}")
            continue
        run_and_log(model, tok, spec, tag, qcfg, PHASE3_DATASETS, jsonl, SLICE_SIZES, done)
        del model, tok; clear_cuda()


# ------------------------------------------------------------------
# Phase 4: ablations on LLaDA-8B
# ------------------------------------------------------------------
def phase4_ablations() -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    jsonl = OUT_DIR / "phase4_ablations.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 4: ablations on LLaDA-8B ===")
    abls: List[tuple] = []

    # a) stage vs pooled
    abls.append(("stage_vs_pooled__turboldm_stages4",
                 QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                             stage_count=4, sensitivity_metric="hidden_mse",
                             residual_top_fraction=0.15, calibration_max_examples=64, probe_max_examples=8)))
    abls.append(("stage_vs_pooled__turboldm_stages1",
                 QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                             stage_count=1, stage_ratios=(0.50,), sensitivity_metric="hidden_mse",
                             residual_top_fraction=0.15, calibration_max_examples=64, probe_max_examples=8)))

    # b) residual schedule
    abls.append(("resched__no_residual",
                 QuantConfig(method="turboldm", base_bits=3, residual_bits=0, residual_top_fraction=0.0,
                             stage_count=4, sensitivity_metric="hidden_mse",
                             calibration_max_examples=64, probe_max_examples=8)))
    abls.append(("resched__static_mixed",
                 QuantConfig(method="qdiffusion_text", w_bits=4, static_high_precision_fraction=0.20,
                             stage_count=4, sensitivity_metric="hidden_mse",
                             calibration_max_examples=64, probe_max_examples=8)))
    abls.append(("resched__stage_residual",
                 QuantConfig(method="turboldm", base_bits=3, residual_bits=1, residual_top_fraction=0.15,
                             stage_count=4, sensitivity_metric="hidden_mse", entropy_gating=False,
                             calibration_max_examples=64, probe_max_examples=8)))
    abls.append(("resched__stage_plus_entropy",
                 QuantConfig(method="turboldm", base_bits=3, residual_bits=1, residual_top_fraction=0.15,
                             stage_count=4, sensitivity_metric="hidden_mse", entropy_gating=True,
                             entropy_quantile=0.50, calibration_max_examples=64, probe_max_examples=8)))

    # c) sensitivity proxy
    for metric in ("hidden_mse", "attn_mse", "act_max"):
        abls.append((f"sens__{metric}",
                     QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                 stage_count=4, sensitivity_metric=metric,
                                 residual_top_fraction=0.15, calibration_max_examples=64, probe_max_examples=8)))

    # d) block / rotation
    for block in (64, 128):
        abls.append((f"codec__block{block}",
                     QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=block,
                                 stage_count=4, sensitivity_metric="hidden_mse",
                                 residual_top_fraction=0.15, calibration_max_examples=64, probe_max_examples=8)))

    # e) bit-config sweep
    for b0, b1 in [(2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1)]:
        abls.append((f"bits__b0{b0}_b1{b1}",
                     QuantConfig(method="turboldm", base_bits=b0, residual_bits=max(b1, 1), block_size=128,
                                 stage_count=4, sensitivity_metric="hidden_mse",
                                 residual_top_fraction=0.15 if b1 > 0 else 0.0,
                                 calibration_max_examples=64, probe_max_examples=8)))

    datasets = ["mmlu", "gsm8k"]
    for tag, qcfg in abls:
        if all((spec.key, tag, ds, "") in done for ds in datasets):
            log(f"SKIP phase4 ablation={tag}")
            continue
        try:
            model, tok = build_method_model(spec, qcfg)
        except Exception as exc:
            log(f"  !! build failed ablation {tag}: {exc}")
            continue
        run_and_log(model, tok, spec, tag, qcfg, datasets, jsonl, SMALL_SLICE_SIZES, done)
        del model, tok; clear_cuda()


# ------------------------------------------------------------------
# Phase 5: MoE scaling appendix
# ------------------------------------------------------------------
def phase5_moe() -> None:
    if "llada_moe_7b_instruct" not in MODEL_SPECS:
        return
    spec = MODEL_SPECS["llada_moe_7b_instruct"]
    jsonl = OUT_DIR / "phase5_moe.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 5: MoE scaling on LLaDA-MoE-7B ===")
    for tag, qcfg in [
        ("bf16", QuantConfig(method="bf16")),
        ("turboldm_3p1", QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                      block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=48, probe_max_examples=6)),
    ]:
        if all((spec.key, tag, ds, "") in done for ds in ["mmlu"]):
            continue
        try:
            model, tok = build_method_model(spec, qcfg)
        except Exception as exc:
            log(f"  !! build failed MoE/{tag}: {exc}")
            continue
        run_and_log(model, tok, spec, tag, qcfg, ["mmlu"], jsonl, SMALL_SLICE_SIZES, done)
        del model, tok; clear_cuda()


# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=str, default="1,2,3,4,5")
    args = parser.parse_args()
    phases = {int(p) for p in args.phases.split(",") if p}
    seed_everything(42)
    set_matmul_precision()
    log(f"driver start, phases={sorted(phases)}")
    dispatch = {
        1: phase1_module_sensitivity,
        2: phase2_main_llada,
        3: phase3_dream,
        4: phase4_ablations,
        5: phase5_moe,
    }
    for p in sorted(phases):
        fn = dispatch.get(p)
        if fn is None:
            continue
        try:
            fn()
        except Exception as exc:
            log(f"!! phase {p} crashed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl",
                         {"phase": p, "err": str(exc), "trace": traceback.format_exc()})
    log("driver done")


if __name__ == "__main__":
    main()
