# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 27: LM-head quantisation on 4 tasks with multiple seeds
(reviewer Phase 5).

Reviewer concern: the earlier LM-head pilot in Phase 22 used MMLU n=75
only and showed quantised head MMLU > default MMLU, which is
surprising. The ask is to re-run with proper task coverage and seeds
and update the practitioner recipe accordingly.

Run matrix:

  recipe      in {turboldm_3p1, turboldm_4p1, awq_4}
  head        in {bf16, quantised}
  dataset     in {mmlu, arc_challenge, hellaswag, gsm8k}
  seed        in {42, 43, 44}

Per-example correctness is stored on top of the aggregate score so we
can compute paired bootstrap CI against BF16 later.

Storage reporting per row:
  * quantised linears GiB
  * BF16 residual GiB (embeddings + norms + biases + head if bf16)
  * full checkpoint GiB

We replace the Phase 22 monkey-patch with an explicit exclude_regex
switch that either includes or excludes the LM head by name.
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

EXCL_BASE_HEAD_BF16 = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
EXCL_BASE_HEAD_QUANT = r"(embed|embedding|router|gate\.weight$)"

# LLaDA's LM head is transformer.ff_out with out_features=126464, which is
# above the default iter_named_linears cap of 5e4. We temporarily raise
# the cap whenever include_lm_head is True.
MAX_OUT_HEAD_CAP = 200000

DATASETS = [
    ("mmlu",         200),
    ("arc_challenge", 200),
    ("hellaswag",    300),
    ("gsm8k",        100),
]

RECIPES = {
    "turboldm_3p1": dict(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                          stage_count=4, sensitivity_metric="hidden_mse",
                          residual_top_fraction=0.15,
                          calibration_max_examples=24, probe_max_examples=3,
                          calibration_max_tokens=192),
    "turboldm_4p1": dict(method="turboldm", base_bits=4, residual_bits=1, block_size=128,
                          stage_count=4, sensitivity_metric="hidden_mse",
                          residual_top_fraction=0.15,
                          calibration_max_examples=24, probe_max_examples=3,
                          calibration_max_tokens=192),
    "awq_4": dict(method="awq", w_bits=4, group_size=128,
                   calibration_max_examples=24, probe_max_examples=3,
                   calibration_max_tokens=192),
}

SEEDS = [42, 43, 44]


def run_one(recipe: str, include_head: bool, dataset: str, n_examples: int,
            seed: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{recipe}{'_head' if include_head else ''}_s{seed}"
    jsonl = OUT_DIR / "phase27_lmhead.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset and r.get("seed") == seed):
                        log(f"Phase 27 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    kwargs = dict(RECIPES[recipe])
    kwargs["exclude_regex"] = EXCL_BASE_HEAD_QUANT if include_head else EXCL_BASE_HEAD_BF16
    qcfg = QuantConfig(**kwargs, include_lm_head=include_head)
    seed_everything(seed); set_matmul_precision()
    log(f"Phase 27 build {tag}/{dataset} seed={seed}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    # Monkey-patch iter_named_linears to use a larger out-features cap when the
    # LM head is in scope. This is the same trick Phase 22 used.
    orig_iter = None
    if include_head:
        orig_iter = H.iter_named_linears
        def _patched(model, include_regex=None, exclude_regex=None, max_out_features=MAX_OUT_HEAD_CAP):
            return orig_iter(model, include_regex, exclude_regex, max_out_features)
        H.iter_named_linears = _patched
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches
        gc.collect(); torch.cuda.empty_cache()
        storage_gb = model_parameter_storage_bytes(model) / 1024**3
        bits = avg_bits_per_param(model)
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        log(f"  eval {tag}/{dataset} n={n_examples} bits={bits:.2f} storage={storage_gb:.2f}GB")
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
        row["recipe"] = recipe
        row["include_head"] = include_head
        row["seed"] = seed
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 27 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 27, "tag": tag, "seed": seed, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        if orig_iter is not None:
            H.iter_named_linears = orig_iter
    del model, tok; clear_cuda()


def main() -> None:
    # Start with 1 seed across the grid, then add more seeds if time permits.
    for seed in SEEDS:
        for recipe in RECIPES:
            for include_head in (False, True):
                for dataset, n in DATASETS:
                    run_one(recipe, include_head, dataset, n, seed)
    log("Phase 27 done")


if __name__ == "__main__":
    main()
