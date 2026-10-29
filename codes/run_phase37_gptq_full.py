# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 37: Full-Hessian GPTQ baseline (closes the GPTQ-diag lower-bound caveat).

Implements original-style GPTQ with column-by-column iterative
quantisation against the Cholesky-damped Hessian H = X^T X over a
calibration probe. Replaces the "GPTQ-diag is a lower bound on GPTQ"
caveat in the paper with a real measurement.

Run matrix: 4 datasets at n=500, single seed. Two configs: 3-bit and
4-bit.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    evaluate_dataset, prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]

METHODS = {
    "gptq_full_3": QuantConfig(method="gptq_full", w_bits=3, group_size=128, exclude_regex=EXCL,
                                  damping=0.01,
                                  calibration_max_examples=24, probe_max_examples=3,
                                  calibration_max_tokens=192),
    "gptq_full_4": QuantConfig(method="gptq_full", w_bits=4, group_size=128, exclude_regex=EXCL,
                                  damping=0.01,
                                  calibration_max_examples=24, probe_max_examples=3,
                                  calibration_max_tokens=192),
}


def run_one(method: str, dataset: str, n_examples: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{method}_s42"
    jsonl = OUT_DIR / "phase37_gptq_full.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset):
                        log(f"Phase 37 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = METHODS[method]
    seed_everything(42); set_matmul_precision()
    log(f"Phase 37 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        storage_gb = model_parameter_storage_bytes(model) / 1024**3
        bits = avg_bits_per_param(model)
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        log(f"  eval {tag}/{dataset} n={n_examples} bits={bits:.2f}")
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
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} bits={bits:.2f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 37 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 37, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for method in METHODS:
        for dataset, n in DATASETS:
            run_one(method, dataset, n)
    log("Phase 37 done")


if __name__ == "__main__":
    main()
