# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 52: NF4 step-budget ablation.

Phase 49 documented that, on MMLU, the BF16 model's answer letter
already settles within the first 1-3 denoising steps. Phase 31
showed accuracy as a function of T for BF16 / TurboLDM-3+1. Phase 52
adds the NF4 row: how does NF4 hold up at very few steps?

This is the natural NF4 counterpart of Phase 31's step-Pareto and
shows whether a real 4-bit kernel can be paired with an aggressive
step budget without collapsing accuracy.

T values for MCQ: {2, 4, 8, 16, 32, 64}. T values for GSM8K:
{16, 32, 64, 128}. n=200 MCQ, n=100 GSM8K.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import replace
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
    maybe_reset_cuda_peak, load_model_and_tokenizer,
    iter_named_linears, get_parent_module,
    generate_text_batch, parse_choice_answer, normalize_number_text,
    format_mcq_example, format_gsm8k_example, load_hf_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

GRID = [
    ("mmlu",  200, [2, 4, 8, 16, 32, 64]),
    ("gsm8k", 100, [16, 32, 64, 128]),
]


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


def evaluate_at_T(model, tok, spec, dataset: str, n: int, T_value: int) -> dict:
    base_gen = mcq_fast_gen_cfg(spec, dataset)
    cfg = replace(base_gen, steps=T_value)
    ds = load_hf_dataset(dataset, max_examples=n)
    is_mcq = dataset != "gsm8k"
    correct = 0; total = 0
    t0 = time.perf_counter()
    for ex in ds:
        if is_mcq:
            p, lab = format_mcq_example(dataset, ex)
        else:
            p, lab = format_gsm8k_example(ex)
            lab = normalize_number_text(lab)
        out = generate_text_batch(model, tok, spec, [p], cfg, device="cuda")[0]
        pred = parse_choice_answer(out, num_choices=4) if is_mcq else normalize_number_text(out)
        if pred == lab:
            correct += 1
        total += 1
    return {"score": correct / max(total, 1), "n": total,
            "wall": time.perf_counter() - t0}


def run_one(dataset: str, n: int, T_value: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"nf4_T{T_value}"
    jsonl = OUT_DIR / "phase52_nf4_steps.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 52 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 52 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        n_swapped = replace_with_nf4(model)
        log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
        gc.collect(); torch.cuda.empty_cache()
        maybe_reset_cuda_peak()
        res = evaluate_at_T(model, tok, spec, dataset, n, T_value)
        row = {
            "model": spec.key, "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "T": T_value,
            "method": "nf4", "seed": 42,
            "avg_bits_per_param": 4.0,
            "param_storage_gb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3,
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} T={T_value} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 52 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 52, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for dataset, n, Ts in GRID:
        for T_value in Ts:
            run_one(dataset, n, T_value)
    log("Phase 52 done")


if __name__ == "__main__":
    main()
