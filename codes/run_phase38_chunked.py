# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 38: chunked low-bit forward (kernel-style memory savings).

Replaces the cached-dequant ``UniformLinear`` with
``UniformLinearChunked`` which never materialises the full BF16 weight.
We then time forward throughput (tokens/sec) and peak VRAM at gen64
and gen128 on LLaDA-8B-Instruct, comparing AWQ-4 in cached-dequant
mode vs.\ chunked mode.

This is *not* a true CUDA kernel; it's a memory-side proxy that shows
the upper bound on VRAM savings achievable by removing the per-layer
BF16 buffer. A fused matmul kernel would also gain throughput.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, GenerateConfig,
    seed_everything, set_matmul_precision, avg_bits_per_param,
    model_parameter_storage_bytes, maybe_reset_cuda_peak,
    evaluate_dataset, prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer, generate_text_batch,
    UniformLinear, UniformLinearChunked, current_peak_mem_gb,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


def swap_to_chunked(model: nn.Module, chunk_size: int = 512) -> int:
    """In-place swap every UniformLinear to UniformLinearChunked. Returns count."""
    swapped = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, UniformLinear):
            # Reconstruct a quant_pack from mod's metadata.
            md = {k: v for k, v in mod.quant_metadata.items()}
            md["q"] = md["q"].to(mod._weight.device) if hasattr(mod, "_weight") else md["q"]
            md["deq"] = mod._weight.detach()  # placeholder, not used
            new = UniformLinearChunked(
                linear=type("X", (), {"bias": mod.bias, "weight": mod._weight, "in_features": mod._weight.shape[1], "out_features": mod._weight.shape[0]})(),
                quant_pack=md,
                input_scale=mod.input_scale if mod.input_scale is not None else None,
                chunk_size=chunk_size,
            ).to(mod._weight.device, dtype=mod._weight.dtype)
            # Replace in parent.
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            setattr(parent, parts[-1], new)
            del mod
            swapped += 1
    return swapped


def benchmark_one(method: str, mode: str, dataset: str = "mmlu", n: int = 50,
                  chunk_size: int = 512) -> None:
    """Build the model, optionally swap to chunked, run a small eval and log VRAM."""
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{method}_{mode}"
    jsonl = OUT_DIR / "phase38_chunked.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("model") == spec.key and r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 38 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    if method == "awq_4":
        qcfg = QuantConfig(method="awq", w_bits=4, group_size=128, exclude_regex=EXCL,
                            calibration_max_examples=24, probe_max_examples=3,
                            calibration_max_tokens=192)
    elif method == "w4_uniform":
        qcfg = QuantConfig(method="uniform", w_bits=4, group_size=128, exclude_regex=EXCL)
    else:
        raise KeyError(method)
    seed_everything(42); set_matmul_precision()
    log(f"Phase 38 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        if mode == "chunked":
            n_swapped = swap_to_chunked(model, chunk_size=chunk_size)
            log(f"  swapped {n_swapped} linears to chunked (chunk_size={chunk_size})")
            gc.collect(); torch.cuda.empty_cache()
        storage_gb = model_parameter_storage_bytes(model) / 1024**3
        log(f"  ready: storage={storage_gb:.2f}GB pre-eval VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB")
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        maybe_reset_cuda_peak()
        t0 = time.perf_counter()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=n)
        rec.baseline = tag
        rec.param_storage_gb = storage_gb
        rec.avg_bits_per_param = avg_bits_per_param(model)
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["mode"] = mode
        row["chunk_size"] = chunk_size
        row["peak_mem_gb"] = current_peak_mem_gb()
        append_jsonl(jsonl, row)
        log(f"    -> {tag}/{dataset} score={rec.score_value:.3f} peak={row['peak_mem_gb']:.2f}GB wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 38 {tag} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 38, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    # Compare cached vs chunked on the strongest 4-bit baseline (AWQ-4)
    # AND on W4 uniform (no AWQ correction, so accuracy survives chunked path).
    for mode in ("cached", "chunked"):
        benchmark_one("awq_4", mode, dataset="mmlu", n=75, chunk_size=512)
    for mode in ("cached", "chunked"):
        benchmark_one("w4_uniform", mode, dataset="mmlu", n=75, chunk_size=512)
    log("Phase 38 done")


if __name__ == "__main__":
    main()
