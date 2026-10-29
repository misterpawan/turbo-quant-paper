# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 41: 16B attn-only A/B via accelerate device_map (closes the
"can't fit attn-only on 32 GB" caveat).

Phase 32 found that quantising only attention on LLaDA2.0-mini
(query_key_value/dense, fused QKV) succeeds at the quant step
(storage 0.10 GB) but the remaining BF16 MLP+embeds keeps the model at
~30.1 GB peak on a 32 GB GPU and the next forward OOMs.

Phase 41 dodges the cap by leaving MLP on CPU via accelerate's
``dispatch_model`` with a balanced device map: attention stays on GPU
(where the small quantised tensors and per-block dequant happen), MLP
modules are CPU-resident and copied-to-GPU per call. The eval is
roughly 5-10x slower than Phase 19's full-quant on-GPU run, but it
gets us the missing attn-only data point.
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

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
INC_ATTN = r"(query_key_value|attention\.dense|att_proj|attn_out|q_proj|k_proj|v_proj|o_proj)"


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


def main() -> None:
    spec = MODEL_SPECS["llada2_mini"]
    tag = "lazy_4_attn_only_offload"
    jsonl = OUT_DIR / "phase41_16b_attnonly.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("model") == spec.key and r.get("baseline") == tag:
                        log(f"Phase 41 SKIP {tag}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 41 build {tag} (16B CPU-first attn-only + accelerate offload)")
    t0 = time.perf_counter()
    clear_cuda()
    try:
        model, tok = _load_16b_cpu(spec)
    except Exception as exc:
        log(f"  !! load failed: {exc}")
        return
    log(f"  loaded {time.perf_counter()-t0:.1f}s")
    qcfg = QuantConfig(method="turboldm_lazy", base_bits=4, residual_bits=1, block_size=128,
                        stage_count=2, stage_ratios=(0.6, 0.2),
                        sensitivity_metric="hidden_mse", residual_top_fraction=0.0,
                        calibration_max_examples=2, probe_max_examples=1,
                        calibration_max_tokens=64, calibration_batch_size=1,
                        compute_stage_scores=False,
                        include_regex=INC_ATTN, exclude_regex=EXCL)
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        log(f"  quantised: bits={avg_bits_per_param(model):.2f} "
            f"storage={model_parameter_storage_bytes(model)/1024**3:.2f}GB")
        # Now place: attention modules on GPU, MLP/router/expert MLPs on CPU.
        # We can build a device_map by walking named modules.
        device_map = {}
        for name, mod in model.named_modules():
            # Leaf-ish placement: any linear / norm / embedding / lm_head.
            n = name.lower()
            if "attention" in n or "att_proj" in n or "attn_out" in n:
                device_map[name] = "cuda:0"
            elif "embed" in n or "lm_head" in n or n.endswith("ff_out") or "norm" in n:
                device_map[name] = "cuda:0"
            else:
                device_map[name] = "cpu"
        log(f"  built device_map with {sum(v=='cuda:0' for v in device_map.values())} GPU and "
            f"{sum(v=='cpu' for v in device_map.values())} CPU placements")
        # Use accelerate to dispatch.
        try:
            from accelerate import dispatch_model
            model = dispatch_model(model, device_map=device_map, main_device="cuda:0")
            log(f"  dispatched: VRAM {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        except Exception as exc:
            log(f"  !! dispatch_model failed: {exc}; falling back to model.to('cuda')")
            model = model.to("cuda")
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
        log(f"    -> mmlu={rec.score_value:.3f} peak={rec.peak_mem_gb:.1f}GB wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 41 {tag} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 41, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()
    log("Phase 41 done")


if __name__ == "__main__":
    main()
