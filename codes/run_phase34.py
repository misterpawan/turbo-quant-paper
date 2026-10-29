# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 34: Main quality-storage table at n=500 with 3 seeds and paired CIs
(reviewer Phase 2 at the requested sample size).

Phase 28 used n=200/200/300/100 for MMLU/ARC-C/HellaSwag/GSM8K because of
the time budget. Phase 34 reruns the same 8-method x 3-seed grid at
n=500 across MMLU/ARC-C/HellaSwag/GSM8K so the 95% paired bootstrap CIs
tighten and the weak gaps from Phase 28 become decisive.

Wall-clock budget on a single RTX 5090:
  MCQ n=500       ~3 min per cell
  GSM8K n=500     ~12 min per cell
  per seed cycle  8 methods x (3 MCQ + 1 GSM8K) = 8 x 21 min = 168 min
  3 seeds         ~8.4 h total

Resume: rows skip-by (model, baseline, dataset, seed). The driver
appends to runs/overnight/phase34_main_ci_n500.jsonl.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    evaluate_dataset, prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

DATASETS = [
    ("mmlu",         500),
    ("arc_challenge", 500),
    ("hellaswag",    500),
    ("gsm8k",        500),
]

METHODS = {
    "bf16":          QuantConfig(method="bf16"),
    "w3":            QuantConfig(method="uniform", w_bits=3, group_size=128, exclude_regex=EXCL),
    "w4":            QuantConfig(method="uniform", w_bits=4, group_size=128, exclude_regex=EXCL),
    "gptq_diag_4":   QuantConfig(method="gptq", w_bits=4, group_size=128, exclude_regex=EXCL,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192),
    "awq_4":         QuantConfig(method="awq", w_bits=4, group_size=128, exclude_regex=EXCL,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192),
    "qdiff4":        QuantConfig(method="qdiffusion_text", w_bits=4, group_size=128, exclude_regex=EXCL,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192),
    "turboldm_3p1":  QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                   stage_count=4, sensitivity_metric="hidden_mse",
                                   residual_top_fraction=0.15,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192, exclude_regex=EXCL),
    "turboldm_4p1":  QuantConfig(method="turboldm", base_bits=4, residual_bits=1, block_size=128,
                                   stage_count=4, sensitivity_metric="hidden_mse",
                                   residual_top_fraction=0.15,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192, exclude_regex=EXCL),
}

SEEDS = [42, 43, 44]


def run_one(method: str, dataset: str, n_examples: int, seed: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{method}_s{seed}"
    jsonl = OUT_DIR / "phase34_main_ci_n500.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset and r.get("seed") == seed):
                        log(f"Phase 34 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = METHODS[method]
    seed_everything(seed); set_matmul_precision()
    log(f"Phase 34 build {tag}/{dataset} seed={seed} n={n_examples}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        if qcfg.method != "bf16":
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
            model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
            del stage_batches
            gc.collect(); torch.cuda.empty_cache()
        storage_gb = model_parameter_storage_bytes(model) / 1024**3
        bits = avg_bits_per_param(model)
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        log(f"  eval {tag}/{dataset} n={n_examples} bits={bits:.2f}")
        t0 = time.perf_counter()
        maybe_reset_cuda_peak()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=n_examples)
        rec.baseline = tag
        rec.param_storage_gb = storage_gb
        rec.avg_bits_per_param = bits
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["quant_cfg"] = asdict(qcfg)
        row["gen_cfg"] = asdict(gen_cfg)
        row["method"] = method
        row["seed"] = seed
        row["n_examples"] = n_examples
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} bits={bits:.2f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 34 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 34, "tag": tag, "seed": seed, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    # Same structure as Phase 28: seed outermost so we have at least one
    # complete seed if we get interrupted.
    for seed in SEEDS:
        for method in METHODS:
            for dataset, n in DATASETS:
                run_one(method, dataset, n, seed)
    log("Phase 34 done")


if __name__ == "__main__":
    main()
