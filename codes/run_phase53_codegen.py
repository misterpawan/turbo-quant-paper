# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 53: Code generation under quantization.

The original 12-hour plan listed HumanEval / MBPP on Dream-7B as a
stretch goal that was never executed. This phase fills that gap on
LLaDA-8B-Instruct with the three quant configurations the paper
actually compares: BF16 reference, NF4 (real 4-bit kernel), and
TurboLDM-3+1 (the paper's flagship recipe). Six rows total.
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
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    iter_named_linears, get_parent_module,
    load_model_and_tokenizer,
    prepare_stage_batches_for_model, quantize_model_inplace,
    evaluate_humaneval, evaluate_mbpp,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

DATASETS = [("humaneval", 80), ("mbpp", 150)]


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


METHODS = {
    "bf16": None,
    "nf4": "nf4",
    "turboldm_3p1": QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                  stage_count=4, sensitivity_metric="hidden_mse",
                                  residual_top_fraction=0.15,
                                  calibration_max_examples=24, probe_max_examples=3,
                                  calibration_max_tokens=192, exclude_regex=EXCL),
}


def run_one(method: str, dataset: str, n_examples: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{method}_s42"
    jsonl = OUT_DIR / "phase53_codegen.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 53 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 53 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        config = METHODS[method]
        if method == "nf4":
            n_swapped = replace_with_nf4(model)
            log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
            gc.collect(); torch.cuda.empty_cache()
            bits = 4.0
            storage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        elif config is not None:
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, config, RunConfig())
            model = quantize_model_inplace(model, tok, spec, config, RunConfig(), stage_batches=stage_batches)
            del stage_batches; gc.collect(); torch.cuda.empty_cache()
            bits = avg_bits_per_param(model)
            storage = model_parameter_storage_bytes(model) / 1024**3
        else:
            bits = 16.0
            storage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        maybe_reset_cuda_peak()
        t0 = time.perf_counter()
        if dataset == "humaneval":
            rec = evaluate_humaneval(model=model, tokenizer=tok, spec=spec,
                                     gen_cfg=gen_cfg, max_examples=n_examples)
        else:
            rec = evaluate_mbpp(model=model, tokenizer=tok, spec=spec,
                                gen_cfg=gen_cfg, max_examples=n_examples)
        rec.baseline = tag
        rec.avg_bits_per_param = bits
        rec.param_storage_gb = storage
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["method"] = method
        row["seed"] = 42
        append_jsonl(jsonl, row)
        log(f"    -> pass@1={rec.score_value:.3f} peak={rec.peak_mem_gb:.1f}GB wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 53 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 53, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for method in METHODS:
        for dataset, n in DATASETS:
            run_one(method, dataset, n)
    log("Phase 53 done")


if __name__ == "__main__":
    main()
