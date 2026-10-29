# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 32: LLaDA2.0-mini 16B failure analysis (reviewer Phase 10).

Phase 19 showed that LazyTurboLDMLinear realises the storage/VRAM win
at 16B but quality collapses (lazy-4-bit MMLU 22.7, lazy-3-bit 17.3 vs
BF16 64.0). This phase tries to diagnose which family is responsible:

  (i)  attn-only 4-bit (leaves MLP BF16) via lazy kernel
  (ii) MLP-only 4-bit (leaves attn BF16) via lazy kernel
  (iii) recipe transfer: use the best LLaDA-8B mixed-bit recipe
       (attn@3 + MLP@4) on LLaDA2.0-mini
  (iv) per-layer reconstruction MSE: dump lazy-reconstructed weight
       vs.\ original BF16 weight for a random subset of layers so we
       can see whether the failure concentrates in particular layers.

All runs use the CPU-first -> GPU-compact flow from Phase 19. Eval is
75-example MMLU only (short) because each 16B build is expensive.
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
    resolve_torch_dtype,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL_FULL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
# Isolate attn. LLaDA2.0-mini uses Qwen2.5-style fused QKV named
# ``query_key_value`` plus an output projection named ``dense``; the
# generic LLaDA-8B (Llama-style) naming uses q_proj/k_proj/v_proj/attn_out.
# The regex below covers both.
INC_ATTN = r"(query_key_value|attention\.dense|att_proj|attn_out|q_proj|k_proj|v_proj|o_proj)"
# Isolate MLP. LLaDA2.0-mini uses gate_proj/up_proj/down_proj inside per-expert
# MLPs; the generic LLaDA-8B uses ff_proj / ff_out.
INC_MLP = r"(gate_proj|up_proj|down_proj|ff_proj|\.ff_out$)"


def _load_16b_cpu(spec):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
    torch_dtype = resolve_torch_dtype(spec.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code,
        torch_dtype=torch_dtype, device_map="cpu", low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


CONFIGS = {
    "lazy_4_attn_only":  QuantConfig(method="turboldm_lazy", base_bits=4, residual_bits=1, block_size=128,
                                        stage_count=2, stage_ratios=(0.6, 0.2),
                                        sensitivity_metric="hidden_mse", residual_top_fraction=0.0,
                                        calibration_max_examples=2, probe_max_examples=1,
                                        calibration_max_tokens=64, calibration_batch_size=1,
                                        compute_stage_scores=False,
                                        include_regex=INC_ATTN, exclude_regex=EXCL_FULL),
    "lazy_4_mlp_only":   QuantConfig(method="turboldm_lazy", base_bits=4, residual_bits=1, block_size=128,
                                        stage_count=2, stage_ratios=(0.6, 0.2),
                                        sensitivity_metric="hidden_mse", residual_top_fraction=0.0,
                                        calibration_max_examples=2, probe_max_examples=1,
                                        calibration_max_tokens=64, calibration_batch_size=1,
                                        compute_stage_scores=False,
                                        include_regex=INC_MLP, exclude_regex=EXCL_FULL),
    "lazy_5":            QuantConfig(method="turboldm_lazy", base_bits=4, residual_bits=2, block_size=128,
                                        stage_count=2, stage_ratios=(0.6, 0.2),
                                        sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
                                        calibration_max_examples=2, probe_max_examples=1,
                                        calibration_max_tokens=64, calibration_batch_size=1,
                                        compute_stage_scores=False, exclude_regex=EXCL_FULL),
}


def run_one(tag: str, qcfg: QuantConfig) -> None:
    spec = MODEL_SPECS["llada2_mini"]
    jsonl = OUT_DIR / "phase32_16b_failure.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("model") == spec.key and r.get("baseline") == tag:
                        log(f"Phase 32 SKIP {tag}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 32 build {tag} (16B CPU-first)")
    t0 = time.perf_counter()
    clear_cuda()
    try:
        model, tok = _load_16b_cpu(spec)
    except Exception as exc:
        log(f"  !! load failed: {exc}")
        return
    log(f"  loaded {time.perf_counter()-t0:.1f}s")
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        log(f"  quantised: bits={avg_bits_per_param(model):.2f} "
            f"storage={model_parameter_storage_bytes(model)/1024**3:.2f}GB")
        model = model.to("cuda")
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log(f"  moved to GPU: VRAM {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        gen_cfg = mcq_fast_gen_cfg(spec, "mmlu")
        maybe_reset_cuda_peak()
        t1 = time.perf_counter()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name="mmlu", gen_cfg=gen_cfg, max_examples=75)
        rec.baseline = tag
        rec.param_storage_gb = model_parameter_storage_bytes(model) / 1024**3
        rec.avg_bits_per_param = avg_bits_per_param(model)
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t1
        row["quant_cfg"] = asdict(qcfg)
        row["gen_cfg"] = asdict(gen_cfg)
        append_jsonl(jsonl, row)
        log(f"    -> mmlu={rec.score_value:.3f} peak={rec.peak_mem_gb:.1f}GB "
            f"wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 32 {tag} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 32, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for tag, qcfg in CONFIGS.items():
        run_one(tag, qcfg)
    log("Phase 32 done")


if __name__ == "__main__":
    main()
