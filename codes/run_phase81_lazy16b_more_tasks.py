# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 81: LLaDA2.0-mini lazy reconstruction on ARC-C, HellaSwag, GSM8K.

The 16B story currently rests on a single MMLU n=75 cell. This phase
extends the lazy-3 / lazy-4 / uniform-int4 sweep from the lazy
reconstruction 16B run to the other three main MCQ tasks at the same
n=75 budget, so the 16B failure-analysis claim has cross-task
support.

We reuse the lazy-reconstruction loader and quantisation pipeline
from the existing 16B driver verbatim; only the dataset list
changes.
"""
from __future__ import annotations

import os, sys
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import QuantConfig
from run_overnight import OUT_DIR, log
from run_phase19 import run_one, EXCL

DATASETS = [("arc_challenge", 75), ("hellaswag", 75), ("gsm8k", 75)]


def main() -> None:
    cfg_common = dict(
        block_size=128, stage_count=2, stage_ratios=(0.6, 0.2),
        sensitivity_metric="hidden_mse",
        residual_top_fraction=0.0,
        calibration_max_examples=2, probe_max_examples=1,
        calibration_max_tokens=64, calibration_batch_size=1,
        compute_stage_scores=False,
        exclude_regex=EXCL,
    )
    for dataset, n in DATASETS:
        # bf16 first (uses device_map=auto, slow due to CPU offload)
        run_one("llada2_mini",
                QuantConfig(method="bf16", **cfg_common),
                dataset, "bf16", max_examples=n)
        # lazy-4
        run_one("llada2_mini",
                QuantConfig(method="turboldm_lazy", base_bits=4, residual_bits=1, **cfg_common),
                dataset, "turboldm_lazy_4", max_examples=n)
        # lazy-3
        run_one("llada2_mini",
                QuantConfig(method="turboldm_lazy", base_bits=3, residual_bits=1, **cfg_common),
                dataset, "turboldm_lazy_3", max_examples=n)
    log("Phase 81 done")


if __name__ == "__main__":
    main()
