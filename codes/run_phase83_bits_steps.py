# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 83: bits x denoising-steps Pareto on LLaDA-8B.

The novel finding the paper rests on: in a diffusion LM, quantisation
noise can be partially compensated by spending more denoising steps.
We sweep three precision settings and three step budgets on GSM8K
(generative, where compounding matters) and MMLU (MCQ, where it
shouldn't), yielding a 2-D quality surface in (bits, steps).

Grid:
  precision : bf16 | turboldm_4_1 | turboldm_3_1
  steps     : 16 | 32 | 64
  datasets  : gsm8k (n=150) | mmlu (n=200)

Each (model, dataset, steps) cell appends a row to phase83_bits_steps.jsonl.
The model is built once per precision and reused across the steps grid.
"""
from __future__ import annotations

import os, sys, time, gc, json, traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer, evaluate_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

DATASETS = [("gsm8k", 150), ("mmlu", 200)]
STEPS = [16, 32, 64]

OUT_PATH = OUT_DIR / "phase83_bits_steps.jsonl"


def already_done(model_tag: str, dataset: str, steps: int) -> bool:
    if not OUT_PATH.exists():
        return False
    with OUT_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if (r.get("baseline") == model_tag
                        and r.get("dataset") == dataset
                        and r.get("gen_cfg", {}).get("steps") == steps):
                    return True
            except Exception:
                continue
    return False


def eval_at_steps(model, tok, spec, model_tag: str, dataset: str,
                   max_examples: int, steps: int, qcfg) -> None:
    if already_done(model_tag, dataset, steps):
        log(f"Phase 83 SKIP {model_tag}/{dataset}/T{steps}")
        return
    gen_cfg = mcq_fast_gen_cfg(spec, dataset)
    gen_cfg.steps = steps
    log(f"  eval {model_tag}/{dataset}/T{steps} n={max_examples}")
    t0 = time.perf_counter()
    maybe_reset_cuda_peak()
    try:
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                                dataset_name=dataset, gen_cfg=gen_cfg,
                                max_examples=max_examples)
        rec.baseline = model_tag
        rec.param_storage_gb = model_parameter_storage_bytes(model) / 1024**3
        rec.avg_bits_per_param = avg_bits_per_param(model)
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["quant_cfg"] = asdict(qcfg) if qcfg is not None else {"method": "bf16"}
        row["gen_cfg"] = asdict(gen_cfg)
        append_jsonl(OUT_PATH, row)
        log(f"    -> score={rec.score_value:.3f} bits={rec.avg_bits_per_param:.2f} "
            f"wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"    !! eval failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl",
                     {"phase": 83, "model_tag": model_tag, "dataset": dataset,
                      "steps": steps, "err": str(exc), "trace": traceback.format_exc()})


def run_variant(spec, model_tag: str, qcfg) -> None:
    cells = [(ds, n, s) for (ds, n) in DATASETS for s in STEPS
             if not already_done(model_tag, ds, s)]
    if not cells:
        log(f"Phase 83 SKIP all cells for {model_tag}")
        return

    log(f"Phase 83 build {model_tag}")
    seed_everything(42); set_matmul_precision()
    clear_cuda()
    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    if qcfg is not None:
        try:
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
            model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(),
                                            stage_batches=stage_batches)
            del stage_batches; gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            log(f"  !! quantize failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl",
                         {"phase": 83, "model_tag": model_tag, "err": str(exc),
                          "trace": traceback.format_exc()})
            del model, tok; clear_cuda()
            return
    log(f"  built {model_tag} in {time.perf_counter()-t0:.1f}s | "
        f"bits={avg_bits_per_param(model):.2f} "
        f"storage={model_parameter_storage_bytes(model)/1024**3:.2f}GB")

    for ds, n, s in cells:
        eval_at_steps(model, tok, spec, model_tag, ds, n, s, qcfg)

    del model, tok; clear_cuda()


def main() -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    cfg_common = dict(
        method="turboldm", block_size=128, stage_count=4,
        stage_ratios=(0.85, 0.60, 0.35, 0.15),
        sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    variants = [
        ("bf16",          None),
        ("turboldm_4_1",  QuantConfig(base_bits=4, residual_bits=1, **cfg_common)),
        ("turboldm_3_1",  QuantConfig(base_bits=3, residual_bits=1, **cfg_common)),
    ]
    for tag, qcfg in variants:
        run_variant(spec, tag, qcfg)
    log("Phase 83 done")


if __name__ == "__main__":
    main()
