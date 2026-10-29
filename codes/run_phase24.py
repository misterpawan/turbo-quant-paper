# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 24: Stage-bin sweep S ∈ {1, 2, 4, 8}.

Reviewer (P1): "the stage scheduler is only validated at one bin count;
please show whether S=1 (no stages) and S=8 change anything." TurboLDM
partitions calibration batches into ``stage_count`` bins by masking ratio
and accumulates separate sensitivity statistics. S=1 collapses to a
non-staged sensitivity baseline; S=8 tests whether finer-grained
stratification pays off. We run at 3+1 because that is where the
scheduler should matter most.
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
    load_model_and_tokenizer,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


def run_one(S: int, dataset: str = "mmlu", max_examples: int = 75) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"turboldm_3p1_S{S}"
    jsonl = OUT_DIR / "phase24_stages.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset):
                        log(f"Phase 24 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=1, block_size=128,
        stage_count=S, sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    seed_everything(42); set_matmul_precision()
    log(f"Phase 24 build S={S}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches
        gc.collect(); torch.cuda.empty_cache()
        storage_gb = model_parameter_storage_bytes(model) / 1024**3
        bits = avg_bits_per_param(model)
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        log(f"  eval {tag}/{dataset} n={max_examples} bits={bits:.2f}")
        t0 = time.perf_counter()
        maybe_reset_cuda_peak()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=max_examples)
        rec.baseline = tag
        rec.param_storage_gb = storage_gb
        rec.avg_bits_per_param = bits
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["quant_cfg"] = asdict(qcfg)
        row["gen_cfg"] = asdict(gen_cfg)
        row["stage_count"] = S
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 24 S={S} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 24, "S": S,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for S in (1, 2, 4, 8):
        run_one(S)
    log("Phase 24 done")


if __name__ == "__main__":
    main()
