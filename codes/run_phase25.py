# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 25: Stage-stratification ablation on GSM8K + MMLU contrast.

Addresses reviewer Phase 8. The original stage-bin sweep (Phase 24) used
MMLU at n=75 which is inside the noise envelope; the abstract claim that
stage stratification dominates pooled calibration is specifically about
GSM8K. This phase reruns S in {1, 2, 4, 8} x N_cal in {24, 64} on
GSM8K-n=100 and adds MMLU-n=200 as a non-reasoning contrast.
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


def run_one(S: int, n_cal: int, dataset: str, max_examples: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"turboldm_3p1_S{S}_ncal{n_cal}"
    jsonl = OUT_DIR / "phase25_stage_gsm8k.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset):
                        log(f"Phase 25 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=1, block_size=128,
        stage_count=S, sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
        calibration_max_examples=n_cal, probe_max_examples=max(3, n_cal // 8),
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    seed_everything(42); set_matmul_precision()
    log(f"Phase 25 build S={S} n_cal={n_cal}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches
        gc.collect(); torch.cuda.empty_cache()
        storage_gb = model_parameter_storage_bytes(model) / 1024**3
        bits = avg_bits_per_param(model)
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        log(f"  eval {tag}/{dataset} n={max_examples} bits={bits:.2f}")
        t0 = time.perf_counter()
        maybe_reset_cuda_peak()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=max_examples)
        rec.baseline = tag
        rec.param_storage_gb = storage_gb
        rec.avg_bits_per_param = bits
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["quant_cfg"] = asdict(qcfg)
        row["gen_cfg"] = asdict(gen_cfg)
        row["stage_count"] = S
        row["n_cal"] = n_cal
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 25 S={S} n_cal={n_cal} {dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 25, "S": S, "n_cal": n_cal, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    # GSM8K is the reasoning task where the stage claim lives.
    for n_cal in (24, 64):
        for S in (1, 2, 4, 8):
            run_one(S, n_cal, "gsm8k", max_examples=100)
    # MMLU as non-reasoning contrast (larger slice than Phase 24's n=75).
    for n_cal in (24, 64):
        for S in (1, 2, 4, 8):
            run_one(S, n_cal, "mmlu", max_examples=200)
    log("Phase 25 done")


if __name__ == "__main__":
    main()
