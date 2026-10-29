# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 54: Throughput (tokens/sec) under real-kernel quantization.

Phase 14 measured throughput at BF16 only. NF4 has a real 4-bit
kernel and is the row that should show genuine inference speed-up
relative to BF16; TurboLDM-3+1 is reported for completeness even
though the reference path stores a dequantized buffer (see paper
Limitations). This is the first VRAM-realistic latency claim in the
paper.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import bitsandbytes as bnb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, GenerateConfig, QuantConfig, RunConfig,
    seed_everything, set_matmul_precision, avg_bits_per_param,
    model_parameter_storage_bytes, maybe_reset_cuda_peak,
    iter_named_linears, get_parent_module,
    load_model_and_tokenizer, generate_text_batch,
    prepare_stage_batches_for_model, quantize_model_inplace,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
PROMPT = ("Explain the difference between supervised and unsupervised machine learning "
          "in a few sentences.")
GEN_CFGS = {
    "gen64":  GenerateConfig(steps=32, gen_length=64,  block_length=32, batch_size=1, temperature=0.0),
    "gen128": GenerateConfig(steps=64, gen_length=128, block_length=32, batch_size=1, temperature=0.0),
    "gen256": GenerateConfig(steps=128, gen_length=256, block_length=32, batch_size=1, temperature=0.0),
}

METHODS = {
    "bf16": None,
    "nf4": "nf4",
    "turboldm_3p1": QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                  stage_count=4, sensitivity_metric="hidden_mse",
                                  residual_top_fraction=0.15,
                                  calibration_max_examples=24, probe_max_examples=3,
                                  calibration_max_tokens=192, exclude_regex=EXCL),
}


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


def run_method(method: str) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{method}_s42"
    jsonl = OUT_DIR / "phase54_throughput.jsonl"
    done = set()
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag:
                        done.add(r.get("dataset"))
                except Exception:
                    continue
    if all(cfg in done for cfg in GEN_CFGS):
        log(f"Phase 54 SKIP {tag} (all gen-cfgs done)")
        return
    seed_everything(42); set_matmul_precision()
    log(f"Phase 54 build {tag}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        config = METHODS[method]
        if method == "nf4":
            n_swapped = replace_with_nf4(model)
            log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
            gc.collect(); torch.cuda.empty_cache()
            bits, storage = 4.0, sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        elif config is not None:
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, config, RunConfig())
            model = quantize_model_inplace(model, tok, spec, config, RunConfig(), stage_batches=stage_batches)
            del stage_batches; gc.collect(); torch.cuda.empty_cache()
            bits = avg_bits_per_param(model)
            storage = model_parameter_storage_bytes(model) / 1024**3
        else:
            bits = 16.0
            storage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

        for cfg_tag, gen_cfg in GEN_CFGS.items():
            if cfg_tag in done:
                log(f"  Phase 54 SKIP {tag}/{cfg_tag}"); continue
            try:
                _ = generate_text_batch(model, tok, spec, [PROMPT], gen_cfg, device="cuda")
            except Exception as exc:
                log(f"  !! warm-up failed {tag}/{cfg_tag}: {exc}")
                continue
            torch.cuda.synchronize()
            maybe_reset_cuda_peak()
            t0 = time.perf_counter()
            try:
                _ = generate_text_batch(model, tok, spec, [PROMPT] * 2, gen_cfg, device="cuda")
            except Exception as exc:
                log(f"  !! timed pass failed {tag}/{cfg_tag}: {exc}")
                continue
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tokens = 2 * gen_cfg.gen_length
            row = {
                "model": spec.key, "baseline": tag, "dataset": cfg_tag,
                "method": method, "seed": 42,
                "avg_bits_per_param": bits, "param_storage_gb": storage,
                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
                "wall_sec": elapsed, "tokens": tokens,
                "tokens_per_sec": tokens / elapsed,
                "gen_cfg": asdict(gen_cfg),
            }
            append_jsonl(jsonl, row)
            log(f"    {tag}/{cfg_tag}: {tokens/elapsed:.1f} tok/s peak {row['peak_mem_gb']:.1f} GB")
    except Exception as exc:
        log(f"  !! Phase 54 {tag} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 54, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for method in METHODS:
        run_method(method)
    log("Phase 54 done")


if __name__ == "__main__":
    main()
