# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 30: module-family sensitivity map across 4 tasks with 3 seeds
(reviewer Phase 7).

Replaces the MMLU-100 single-seed family sweep in Section 5.3 with a
proper multi-task, multi-seed version. Each row quantises \emph{only}
the named family at 3-bit TurboQuant (no residual) and leaves
everything else BF16.
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

# LLaDA uses packed QKV ("att_proj"). Single-family isolation: only
# quantise the named family. Everything else must be excluded (stays
# BF16). For a family "F" the exclude regex is "NOT matching F".
# We use include_regex to pick the family and a broad exclude for safety.
FAMILIES = {
    "att_proj":  r"att_proj",
    "attn_out":  r"attn_out",
    "q_proj":    r"q_proj",
    "k_proj":    r"k_proj",
    "v_proj":    r"v_proj",
    "ff_proj":   r"ff_proj",      # fused gate+up on LLaDA/OLMo
    "ff_out":    r"\.ff_out$",    # block-level down
    "attn_all":  r"(att_proj|attn_out)",
    "mlp_all":   r"(ff_proj|\.ff_out$)",
}

DATASETS = [
    ("mmlu",         200),
    ("arc_challenge", 200),
    ("hellaswag",    300),
    ("gsm8k",        100),
]

SEEDS = [42, 43, 44]


def run_one(family: str, dataset: str, n_examples: int, seed: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"mod_{family}_s{seed}"
    jsonl = OUT_DIR / "phase30_modsens_multitask.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset and r.get("seed") == seed):
                        log(f"Phase 30 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=0, block_size=128,
        stage_count=4, sensitivity_metric="hidden_mse", residual_top_fraction=0.0,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192,
        include_regex=FAMILIES[family],
        exclude_regex=r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)",
    )
    seed_everything(seed); set_matmul_precision()
    log(f"Phase 30 build {tag}/{dataset} seed={seed}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
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
        row["family"] = family
        row["seed"] = seed
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 30 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 30, "tag": tag, "seed": seed, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    # Seed outermost so we always have at least one seed of every family.
    for seed in SEEDS:
        for family in FAMILIES:
            for dataset, n in DATASETS:
                run_one(family, dataset, n, seed)
    log("Phase 30 done")


if __name__ == "__main__":
    main()
