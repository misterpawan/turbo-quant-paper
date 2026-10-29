# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 72: Vocab-side quantisation combos on LLaDA-8B.

Reviewer asked: what happens when both LM head and embedding are
quantised on top of the body? The earlier per-component pilots
(Phase 27 lm_head, Phase 39 embedding) treat each side independently.
This phase tests the full deployment combo:

    body only (TurboLDM-(3+1))
    body + LM head quantised
    body + embedding quantised
    body + LM head + embedding quantised

Reports total checkpoint GiB and accuracy on the 4-dataset main grid
at n=500. The expected payoff is 3.8-4.0 GiB total checkpoint if both
sides survive.
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
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

# Body excludes lm_head and embedding by default
EXCL_BODY        = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
# Body + lm_head quantised: drop lm_head from exclude
EXCL_BODY_HEAD   = r"(embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
# Body + embedding quantised: drop embedding from exclude
EXCL_BODY_EMBED  = r"(lm_head|router|gate\.weight$|transformer\.ff_out$)"
# Body + head + embedding: drop both
EXCL_BODY_BOTH   = r"(router|gate\.weight$|transformer\.ff_out$)"

VARIANTS = {
    "body_only":   EXCL_BODY,
    "body_head":   EXCL_BODY_HEAD,
    "body_embed":  EXCL_BODY_EMBED,
    "body_both":   EXCL_BODY_BOTH,
}

DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]


def make_qcfg(exclude_regex: str) -> QuantConfig:
    return QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                        stage_count=4, sensitivity_metric="hidden_mse",
                        residual_top_fraction=0.15,
                        calibration_max_examples=24, probe_max_examples=3,
                        calibration_max_tokens=192, exclude_regex=exclude_regex)


def run_one(variant: str, dataset: str, n: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"turboldm_3p1_{variant}_s42"
    jsonl = OUT_DIR / "phase72_vocab_side_quant.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 72 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 72 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    qcfg = make_qcfg(VARIANTS[variant])
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        bits = avg_bits_per_param(model)
        storage_quant = model_parameter_storage_bytes(model) / 1024**3
        # full param storage (already in compressed form for the modules touched)
        storage_total = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        log(f"  variant={variant} bits={bits:.2f} quant_storage={storage_quant:.2f}GB "
            f"total_storage={storage_total:.2f}GB")
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        maybe_reset_cuda_peak()
        t0 = time.perf_counter()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=n)
        rec.baseline = tag
        rec.avg_bits_per_param = bits
        rec.param_storage_gb = storage_quant
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["method"] = "turboldm_3p1"
        row["variant"] = variant
        row["seed"] = 42
        row["total_storage_gb"] = storage_total
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 72 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 72, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    # body_only is already done in Phase 28/34 main table; we still rerun
    # at this n=500 single-seed to make the comparison strictly internal
    # to this phase. SKIP-by-(tag, dataset) keeps it cheap if interrupted.
    for variant in VARIANTS:
        for dataset, n in DATASETS:
            run_one(variant, dataset, n)
    log("Phase 72 done")


if __name__ == "__main__":
    main()
