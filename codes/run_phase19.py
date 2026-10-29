# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 19: 16B-scale quantization via the lazy reconstruction kernel.

Our normal ``apply_turboldm`` caches a dequantized BF16 buffer in each
``TurboLDMLinear`` for convenience; at 16B parameters that means the
quantized model has roughly the same inference VRAM as the BF16 baseline,
which is incompatible with fitting everything on a 32 GB GPU.

``apply_lazy_turboldm`` uses :class:`LazyTurboLDMLinear` which stores only
int8 indices, per-block RMS, random-sign bitmap, and a tiny codebook per
linear. The BF16 weight is reconstructed in ``forward()`` and freed
immediately. This gives true sub-BF16 runtime memory and enables
quantization for 16B-class models within a single 32 GB budget.
"""
from __future__ import annotations

import os, sys, time, torch, gc, traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    quantize_model_inplace, prepare_stage_batches_for_model,
    evaluate_dataset, iter_named_linears, resolve_torch_dtype,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


def _load_16b(spec, to_cpu_first: bool = False) -> tuple:
    """Load a 16B model.

    If ``to_cpu_first`` is True we put the entire model on CPU (no
    accelerate offloading hooks). This is required for in-place
    quantization to actually reduce GPU footprint: with ``device_map='auto'``
    accelerate's ``AlignDevicesHook`` keeps references to the original
    parameters, so replacing a ``nn.Linear`` with a compact
    ``LazyTurboLDMLinear`` does not free the BF16 weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
    torch_dtype = resolve_torch_dtype(spec.dtype)
    if to_cpu_first:
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code,
            torch_dtype=torch_dtype, device_map="cpu", low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code,
            torch_dtype=torch_dtype, device_map="auto", low_cpu_mem_usage=True,
        )
    model.eval()
    return model, tok


def run_one(spec_key: str, qcfg: QuantConfig, dataset_name: str, tag: str,
            max_examples: int) -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase19_lazy_16b.jsonl"
    # Resume: skip if already logged.
    if jsonl.exists():
        import json as _json
        with jsonl.open() as f:
            for line in f:
                try:
                    r = _json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset_name):
                        log(f"Phase 19 SKIP (resume) {tag}/{dataset_name}")
                        return
                except Exception:
                    continue
    seed_everything(42)
    set_matmul_precision()
    log(f"Phase 19 build {tag} on {spec_key}")
    t0 = time.perf_counter()
    # For quant methods, load on CPU first so we can truly compact the model
    # before moving any weights to GPU. For bf16 use device_map=auto.
    to_cpu_first = qcfg.method != "bf16"
    try:
        model, tok = _load_16b(spec, to_cpu_first=to_cpu_first)
    except Exception as exc:
        log(f"  !! load failed: {exc}")
        return
    log(f"  loaded {time.perf_counter()-t0:.1f}s "
        f"VRAM {torch.cuda.memory_allocated()/1024**3:.2f}GB cpu_first={to_cpu_first}")
    if qcfg.method != "bf16":
        try:
            # Build stage batches on CPU too.
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
            model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
            del stage_batches
            gc.collect()
            log(f"  quantized {tag} on CPU: bits={avg_bits_per_param(model):.2f} "
                f"storage={model_parameter_storage_bytes(model)/1024**3:.2f}GB")
            # Move compact model to GPU.
            model = model.to("cuda")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"  moved to GPU: VRAM {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        except Exception as exc:
            log(f"  !! quantize failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 19, "tag": tag,
                                                      "err": str(exc), "trace": traceback.format_exc()})
            del model, tok; gc.collect(); torch.cuda.empty_cache()
            return
    gen_cfg = mcq_fast_gen_cfg(spec, dataset_name)
    log(f"  eval {tag}/{dataset_name} n={max_examples}")
    t0 = time.perf_counter()
    maybe_reset_cuda_peak()
    try:
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                                dataset_name=dataset_name, gen_cfg=gen_cfg,
                                max_examples=max_examples)
        rec.baseline = tag
        rec.param_storage_gb = model_parameter_storage_bytes(model) / 1024**3
        rec.avg_bits_per_param = avg_bits_per_param(model)
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["quant_cfg"] = asdict(qcfg)
        row["gen_cfg"] = asdict(gen_cfg)
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} bits={rec.avg_bits_per_param:.2f} "
            f"storage={rec.param_storage_gb:.2f}GB peak={rec.peak_mem_gb:.1f}GB "
            f"wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"    !! eval failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 19, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    # BF16 baseline is already in Phase 15. We focus on quant methods here.
    cfg_common = dict(
        block_size=128, stage_count=2, stage_ratios=(0.6, 0.2),
        sensitivity_metric="hidden_mse",
        residual_top_fraction=0.0,
        calibration_max_examples=2, probe_max_examples=1,
        calibration_max_tokens=64, calibration_batch_size=1,
        compute_stage_scores=False,
        exclude_regex=EXCL,
    )
    run_one("llada2_mini",
            QuantConfig(method="turboldm_lazy", base_bits=4, residual_bits=1, **cfg_common),
            "mmlu", "turboldm_lazy_4", max_examples=75)
    run_one("llada2_mini",
            QuantConfig(method="turboldm_lazy", base_bits=3, residual_bits=1, **cfg_common),
            "mmlu", "turboldm_lazy_3", max_examples=75)
    # Uniform 4-bit baseline at 16B — our UniformLinear still caches BF16, so
    # this will OOM, but we log that honestly.
    run_one("llada2_mini",
            QuantConfig(method="uniform", w_bits=4, group_size=128, exclude_regex=EXCL),
            "mmlu", "uniform_int4", max_examples=75)
    log("Phase 19 done")


if __name__ == "__main__":
    main()
