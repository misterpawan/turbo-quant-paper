# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 84: 3-seed CIs for the GSM8K row on LLaDA-8B.

The reviewer-required experiment. The main results table compares
TurboLDM, GPTQ-4, AWQ-4, uniform-int4, and BF16 on GSM8K but with a
single seed. With small benchmark slices the cell-to-cell noise is
non-negligible, so this phase reruns the GSM8K column with three
seeds (17, 42, 101) per quant method at n=200. Offline aggregation
turns the 3-seed cells into mean +/- 95 percent bootstrap CIs that
get plumbed into gen_table_main_ci.tex.

Methods rerun: bf16 (1 seed only, for reference), uniform_int4,
gptq_4, awq_4, turboldm_4_1, turboldm_3_1. The 3-bit baselines
(gptq_3, awq_3) are not included here because they sit far below
the BF16 line and CIs at that loss level don't change the picture;
they can be added later if reviewers ask.
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

DATASET = "gsm8k"
N_EXAMPLES = 200
SEEDS = [17, 42, 101]

OUT_PATH = OUT_DIR / "phase84_seed_ci_gsm8k.jsonl"


def already_done(model_tag: str, seed: int) -> bool:
    if not OUT_PATH.exists():
        return False
    with OUT_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("baseline") == model_tag and r.get("seed") == seed:
                    return True
            except Exception:
                continue
    return False


def build_model(spec, qcfg):
    """Load + (optionally) quantise. Returns (model, tok) or raises."""
    set_matmul_precision()
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    if qcfg is not None:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(),
                                        stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
    return model, tok


def run_seed(spec, model_tag: str, qcfg, seed: int) -> None:
    if already_done(model_tag, seed):
        log(f"Phase 84 SKIP {model_tag}/seed={seed}")
        return
    log(f"Phase 84 build {model_tag} seed={seed}")
    seed_everything(seed)
    t0 = time.perf_counter()
    try:
        model, tok = build_model(spec, qcfg)
    except Exception as exc:
        log(f"  !! build failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl",
                     {"phase": 84, "model_tag": model_tag, "seed": seed,
                      "err": str(exc), "trace": traceback.format_exc()})
        return
    log(f"  built {model_tag} in {time.perf_counter()-t0:.1f}s | "
        f"bits={avg_bits_per_param(model):.2f} "
        f"storage={model_parameter_storage_bytes(model)/1024**3:.2f}GB")

    gen_cfg = mcq_fast_gen_cfg(spec, DATASET)
    log(f"  eval {model_tag}/{DATASET}/seed={seed} n={N_EXAMPLES}")
    t0 = time.perf_counter()
    maybe_reset_cuda_peak()
    try:
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                                dataset_name=DATASET, gen_cfg=gen_cfg,
                                max_examples=N_EXAMPLES)
        rec.baseline = model_tag
        rec.param_storage_gb = model_parameter_storage_bytes(model) / 1024**3
        rec.avg_bits_per_param = avg_bits_per_param(model)
        row = rec.to_dict()
        row["seed"] = seed
        row["wall_sec"] = time.perf_counter() - t0
        row["quant_cfg"] = asdict(qcfg) if qcfg is not None else {"method": "bf16"}
        row["gen_cfg"] = asdict(gen_cfg)
        append_jsonl(OUT_PATH, row)
        log(f"    -> score={rec.score_value:.3f} bits={rec.avg_bits_per_param:.2f} "
            f"wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"    !! eval failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl",
                     {"phase": 84, "model_tag": model_tag, "seed": seed,
                      "err": str(exc), "trace": traceback.format_exc()})

    del model, tok; clear_cuda()


def main() -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    turbo_common = dict(
        method="turboldm", block_size=128, stage_count=4,
        stage_ratios=(0.85, 0.60, 0.35, 0.15),
        sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    quant_methods = [
        ("uniform_int4",  QuantConfig(method="uniform", w_bits=4, group_size=128, exclude_regex=EXCL)),
        ("gptq_4",        QuantConfig(method="gptq", w_bits=4, group_size=128, exclude_regex=EXCL,
                                       calibration_max_examples=64, calibration_max_tokens=512)),
        ("awq_4",         QuantConfig(method="awq", w_bits=4, group_size=128, exclude_regex=EXCL,
                                       calibration_max_examples=64, calibration_max_tokens=512)),
        ("turboldm_4_1",  QuantConfig(base_bits=4, residual_bits=1, **turbo_common)),
        ("turboldm_3_1",  QuantConfig(base_bits=3, residual_bits=1, **turbo_common)),
    ]

    # BF16 reference: only seed=42 (deterministic in our pipeline; one cell suffices).
    run_seed(spec, "bf16", None, seed=42)

    # Quant methods: all three seeds.
    for tag, qcfg in quant_methods:
        for seed in SEEDS:
            run_seed(spec, tag, qcfg, seed)
    log("Phase 84 done")


if __name__ == "__main__":
    main()
