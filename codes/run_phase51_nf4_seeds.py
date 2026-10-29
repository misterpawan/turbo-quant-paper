# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 51: NF4 multi-seed CI extension to Phase 47.

Phase 47 produced one row per dataset at seed=42. To put a credible
error bar on the NF4 numbers in the main table we add seeds 17 and
101, mirroring the 3-seed convention used in Phase 43. Aggregator can
then merge phase47 + phase51 to compute mean and 95% CI.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import bitsandbytes as bnb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, seed_everything, set_matmul_precision,
    maybe_reset_cuda_peak, evaluate_dataset,
    load_model_and_tokenizer, iter_named_linears, get_parent_module,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]
SEEDS = [17, 101]


def replace_with_nf4(model: nn.Module) -> int:
    n = 0
    layer_names = [name for name, _ in iter_named_linears(model, exclude_regex=EXCL)]
    for name in layer_names:
        parent, leaf = get_parent_module(model, name)
        linear = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        bnb_layer = bnb.nn.Linear4bit(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
        )
        bnb_layer.load_state_dict(linear.state_dict())
        bnb_layer = bnb_layer.to(linear.weight.device)
        setattr(parent, leaf, bnb_layer)
        n += 1
    return n


def run_one(dataset: str, n_examples: int, seed: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"nf4_s{seed}"
    jsonl = OUT_DIR / "phase51_nf4_seeds.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 51 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(seed); set_matmul_precision()
    log(f"Phase 51 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        n_swapped = replace_with_nf4(model)
        log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
        gc.collect(); torch.cuda.empty_cache()
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        maybe_reset_cuda_peak()
        t0 = time.perf_counter()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=n_examples)
        rec.baseline = tag
        rec.param_storage_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        rec.avg_bits_per_param = 4.0
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["method"] = "nf4"
        row["seed"] = seed
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} peak={rec.peak_mem_gb:.1f}GB wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 51 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 51, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for seed in SEEDS:
        for dataset, n in DATASETS:
            run_one(dataset, n, seed)
    log("Phase 51 done")


if __name__ == "__main__":
    main()
