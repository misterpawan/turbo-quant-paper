# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 66: AWQ-4 and TurboLDM-(3+1) on LLaDA-MoE-7B-A1B-Instruct.

Phase 5 covered MoE BF16; Phase 59 added MoE NF4. The remaining
"does the recipe transfer to MoE?" question needs AWQ-4 (the
strongest 4-bit baseline) and TurboLDM-(3+1) (the paper's flagship)
on the same model. Single seed, n=300 to keep runtime bounded; if
both methods complete, additional seeds can be added in a follow-up.
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
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    evaluate_dataset, load_model_and_tokenizer,
    prepare_stage_batches_for_model, quantize_model_inplace,
    apply_awq_fallback, load_calibration_texts,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
DATASETS = [("mmlu", 300), ("arc_challenge", 300), ("hellaswag", 300), ("gsm8k", 300)]

METHODS = {
    "awq_4": QuantConfig(method="awq", w_bits=4, group_size=128,
                          calibration_max_examples=24, probe_max_examples=3,
                          calibration_max_tokens=256, exclude_regex=EXCL),
    "turboldm_3p1": QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                  stage_count=4, sensitivity_metric="hidden_mse",
                                  residual_top_fraction=0.15,
                                  calibration_max_examples=24, probe_max_examples=3,
                                  calibration_max_tokens=192, exclude_regex=EXCL),
}


def run_one(method: str, dataset: str, n: int) -> None:
    spec = MODEL_SPECS["llada_moe_7b_instruct"]
    tag = f"{method}_s42"
    jsonl = OUT_DIR / "phase66_moe_methods.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 66 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 66 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    qcfg = METHODS[method]
    try:
        # Both AWQ and TurboLDM are dispatched by quantize_model_inplace;
        # the helper builds activation stats from the same stage_batches
        # for AWQ when method=="awq".
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches
        gc.collect(); torch.cuda.empty_cache()
        bits = avg_bits_per_param(model)
        storage = model_parameter_storage_bytes(model) / 1024**3
        log(f"  bits={bits:.2f} storage={storage:.2f}GB")
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        maybe_reset_cuda_peak()
        t0 = time.perf_counter()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=n)
        rec.baseline = tag
        rec.avg_bits_per_param = bits
        rec.param_storage_gb = storage
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["method"] = method
        row["seed"] = 42
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} peak={rec.peak_mem_gb:.1f}GB wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 66 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 66, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for method in METHODS:
        for dataset, n in DATASETS:
            run_one(method, dataset, n)
    log("Phase 66 done")


if __name__ == "__main__":
    main()
