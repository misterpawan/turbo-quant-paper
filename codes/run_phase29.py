# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 29: per-(layer, stage) hidden-state MSE dump
(reviewer Phase 6 analogue).

The reviewer's original Phase 6 was for a different paper and asked for
per-step critic-score tracking. The analogue for a PTQ paper on an MDLM
is: log the per-(layer, stage) hidden-state MSE induced by quantisation
and correlate it with the downstream task delta measured in Phase 30.
That *is* a mechanism test: if the sensitivity metric the scheduler
uses correlates with real task sensitivity, then the theory-motivated
scoring is doing useful work; if it does not, the method is effectively
random weight placement.

This driver loads LLaDA-8B-Instruct, runs
``compute_stage_layer_scores`` on a 24-example, 192-token WikiText
probe at S=4 stages, and dumps the raw (layer, stage, score) table to
``runs/overnight/phase29_layer_stage_mse.csv``. A separate
post-processing step (after Phase 30 finishes) will correlate this
with the measured per-module MMLU/ARC-C/HellaSwag/GSM8K deltas.
"""
from __future__ import annotations

import os, sys, gc, traceback, csv, time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    prepare_stage_batches_for_model, compute_stage_layer_scores,
    load_model_and_tokenizer,
)
from run_overnight import OUT_DIR, log, clear_cuda

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


def main() -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    out_csv = OUT_DIR / "phase29_layer_stage_mse.csv"
    if out_csv.exists():
        log(f"Phase 29 SKIP (output exists): {out_csv}")
        return
    seed_everything(42); set_matmul_precision()
    log("Phase 29 build LLaDA-8B-Instruct")
    t0 = time.perf_counter()
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=1, block_size=128,
        stage_count=4, sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        scores = compute_stage_layer_scores(model, stage_batches, qcfg)
        log(f"  scores shape: {scores.shape}, columns: {list(scores.columns)}")
        scores.to_csv(out_csv, index=False)
        log(f"  wrote {out_csv} ({len(scores)} rows) in {time.perf_counter()-t0:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 29 failed: {exc}\n{traceback.format_exc()}")
    del model, tok; clear_cuda()
    log("Phase 29 done")


if __name__ == "__main__":
    main()
