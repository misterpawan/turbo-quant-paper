# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 31: full step-count x quantisation Pareto
(reviewer Phase 9).

Goal: show the accuracy-vs-wallclock frontier at T in {8, 16, 32, 64,
128} for every major method. The current Appendix step-count table
only covers T in {16, 32, 64} at BF16 / AWQ-4 / TurboLDM (3+1).
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict, replace

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, GenerateConfig,
    seed_everything, set_matmul_precision, avg_bits_per_param,
    model_parameter_storage_bytes, maybe_reset_cuda_peak,
    evaluate_dataset, prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

METHODS = {
    "bf16":          QuantConfig(method="bf16"),
    "w4":            QuantConfig(method="uniform", w_bits=4, group_size=128, exclude_regex=EXCL),
    "awq_4":         QuantConfig(method="awq", w_bits=4, group_size=128, exclude_regex=EXCL,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192),
    "turboldm_3p1":  QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                   stage_count=4, sensitivity_metric="hidden_mse",
                                   residual_top_fraction=0.15,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192, exclude_regex=EXCL),
    "turboldm_4p1":  QuantConfig(method="turboldm", base_bits=4, residual_bits=1, block_size=128,
                                   stage_count=4, sensitivity_metric="hidden_mse",
                                   residual_top_fraction=0.15,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192, exclude_regex=EXCL),
}

T_VALUES = [8, 16, 32, 64, 128]
DATASETS = [
    ("mmlu", 200),
    ("gsm8k", 100),
]


def run_one(method: str, T: int, dataset: str, n_examples: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{method}_T{T}"
    jsonl = OUT_DIR / "phase31_steps_pareto.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset):
                        log(f"Phase 31 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = METHODS[method]
    seed_everything(42); set_matmul_precision()
    log(f"Phase 31 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        if qcfg.method != "bf16":
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
            model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
            del stage_batches; gc.collect(); torch.cuda.empty_cache()
        storage_gb = model_parameter_storage_bytes(model) / 1024**3
        bits = avg_bits_per_param(model)
        # Build a generate config with varying T but default gen_length/block.
        base_gen = mcq_fast_gen_cfg(spec, dataset)
        gen_cfg = replace(base_gen, steps=T)
        log(f"  eval {tag}/{dataset} n={n_examples} bits={bits:.2f} T={T}")
        t0 = time.perf_counter()
        maybe_reset_cuda_peak()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=n_examples)
        rec.baseline = tag
        rec.param_storage_gb = storage_gb
        rec.avg_bits_per_param = bits
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["quant_cfg"] = asdict(qcfg)
        row["gen_cfg"] = asdict(gen_cfg)
        row["method"] = method
        row["T"] = T
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} T={T} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 31 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 31, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for method in METHODS:
        for T in T_VALUES:
            for dataset, n in DATASETS:
                run_one(method, T, dataset, n)
    log("Phase 31 done")


if __name__ == "__main__":
    main()
