# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 79: LLaDA-MoE-7B per-family module sensitivity at n=200.

Currently the paper has no per-family map for LLaDA-MoE; the
abstract's "compound-attention reproduces across model families"
claim is supported by Dream but not by the MoE family. This phase
adds the MoE counterpart of the LLaDA-8B map, with the same
per-family TurboLDM-(3+0) recipe and the same 4-task evaluation grid
at n=200/200/300/100 single seed.
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

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

# Match the Phase 30 family set, adapted for LLaDA-MoE's naming
FAMILIES = {
    "q_proj":   r"\.q_proj$",
    "k_proj":   r"\.k_proj$",
    "v_proj":   r"\.v_proj$",
    "o_proj":   r"\.o_proj$",
    "attn_all": r"(\.q_proj|\.k_proj|\.v_proj|\.o_proj)$",
    "gate_proj": r"\.gate_proj$",
    "up_proj":   r"\.up_proj$",
    "down_proj": r"\.down_proj$",
    "mlp_all":   r"(\.gate_proj|\.up_proj|\.down_proj)$",
}

DATASETS = [("mmlu", 200), ("arc_challenge", 200), ("hellaswag", 300), ("gsm8k", 100)]


def make_qcfg(include_regex: str) -> QuantConfig:
    return QuantConfig(method="turboldm", base_bits=3, residual_bits=0, block_size=128,
                        stage_count=4, sensitivity_metric="hidden_mse",
                        residual_top_fraction=0.0,
                        calibration_max_examples=24, probe_max_examples=3,
                        calibration_max_tokens=192,
                        include_regex=include_regex, exclude_regex=EXCL)


def run_one(family: str, dataset: str, n: int) -> None:
    spec = MODEL_SPECS["llada_moe_7b_instruct"]
    tag = f"moe_mod_{family}_s42"
    jsonl = OUT_DIR / "phase79_moe_modsens.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 79 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 79 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    qcfg = make_qcfg(include_regex=FAMILIES[family])
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        bits = avg_bits_per_param(model)
        storage = model_parameter_storage_bytes(model) / 1024**3
        log(f"  family={family} bits={bits:.2f} storage={storage:.2f}GiB")
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
        row["method"] = "turboldm_3"
        row["family"] = family
        row["seed"] = 42
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 79 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 79, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for family in FAMILIES:
        for dataset, n in DATASETS:
            run_one(family, dataset, n)
    log("Phase 79 done")


if __name__ == "__main__":
    main()
