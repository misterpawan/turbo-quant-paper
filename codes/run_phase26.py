# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 26: Matched-budget residual-schedule ablation (reviewer Phase 4).

The existing residual-schedule ablation in Section "Ablations" has
``No residual`` at 3.26 bits scoring higher GSM8K than ``Stage residual``
at 3.53 bits, which directly undermines the method story. The reviewer
asked: if we rerun every schedule at matched average-bits, does the
stage schedule actually help?

Methods (all at base_bits=3, residual_bits=1, block_size=128, S=4):

  * none           - base only (no residual)
  * random         - random top_fraction (layer, stage) pairs
  * static_layer   - pick top_fraction/S fraction of *layers* and
                     activate all stages on them (no stage gating)
  * stage          - sensitivity-ranked (layer, stage) pairs [default]
  * stage_entropy  - stage + entropy_gating=True
  * layer_once     - knapsack in the layer-once cost model

Target avg_bits ~ 3.55 so we can compare at matched storage. We sweep
``residual_top_fraction in {0.00, 0.15, 0.30, 0.60}`` so the matched
point (~3.55) falls in the middle of the sweep.

Datasets: GSM8K (reasoning, where the claim lives) at n=100,
MMLU at n=200, ARC-C at n=200.
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

DATASETS = [
    ("gsm8k", 100),
    ("mmlu", 200),
    ("arc_challenge", 200),
]

MODES = [
    ("none",          0.00),
    ("random",        0.15),
    ("static_layer",  0.15),
    ("stage",         0.15),
    ("stage_entropy", 0.15),
    ("layer_once",    0.15),
    ("stage",         0.30),   # +residual coverage
    ("stage",         0.60),   # +residual coverage
]


def run_one(mode: str, frac: float, dataset: str, max_examples: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"turboldm_3p1_{mode}_frac{int(frac*100):02d}"
    jsonl = OUT_DIR / "phase26_residual_matched.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset):
                        log(f"Phase 26 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=1, block_size=128,
        stage_count=4, sensitivity_metric="hidden_mse",
        residual_top_fraction=frac, schedule_mode=mode,
        entropy_gating=(mode == "stage_entropy"),
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    seed_everything(42); set_matmul_precision()
    log(f"Phase 26 build mode={mode} frac={frac}")
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
        row["schedule_mode"] = mode
        row["residual_top_fraction"] = frac
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} bits={bits:.2f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 26 {mode}/{frac} {dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 26, "mode": mode, "frac": frac, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for mode, frac in MODES:
        for dataset, n in DATASETS:
            run_one(mode, frac, dataset, n)
    log("Phase 26 done")


if __name__ == "__main__":
    main()
