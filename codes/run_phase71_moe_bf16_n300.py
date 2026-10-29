# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 71: LLaDA-MoE-7B BF16 at n=300 — fair reference for the Phase 59 NF4
and Phase 66 AWQ-4/TurboLDM-(3+1) rows.

Phase 5's BF16 MoE numbers are at n=100. This phase produces a matched
n=300 BF16 reference so the MoE comparison in tab:nf4-main is on equal
footing.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, seed_everything, set_matmul_precision,
    maybe_reset_cuda_peak, evaluate_dataset,
    load_model_and_tokenizer,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

DATASETS = [("mmlu", 300), ("arc_challenge", 300), ("hellaswag", 300), ("gsm8k", 300)]


def run_one(dataset: str, n_examples: int) -> None:
    spec = MODEL_SPECS["llada_moe_7b_instruct"]
    tag = "bf16_n300_s42"
    jsonl = OUT_DIR / "phase71_moe_bf16_n300.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 71 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 71 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        maybe_reset_cuda_peak()
        t0 = time.perf_counter()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=n_examples)
        rec.baseline = tag
        rec.avg_bits_per_param = 16.0
        rec.param_storage_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["method"] = "bf16"
        row["seed"] = 42
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} peak={rec.peak_mem_gb:.1f}GB wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 71 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 71, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for dataset, n in DATASETS:
        run_one(dataset, n)
    log("Phase 71 done")


if __name__ == "__main__":
    main()
