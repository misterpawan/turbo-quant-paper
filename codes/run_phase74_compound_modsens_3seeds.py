# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 74: Three-seed corrected compound module sensitivity (true
attn_all and mlp_all at 3-bit on LLaDA-8B).

The corrected compound-attention vs.\ compound-MLP finding now
appears in the abstract and conclusion. The current corrected-compound
table is single-seed; the reviewer asked for matched seed coverage so
the central claim is CI-backed. This script reruns just two
quantisation cells (true attn_all, true mlp_all at 3-bit base, no
residual) at seeds {17, 42, 101} on the four-task n=500/300 grid.
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
DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 300)]
SEEDS = [17, 42, 101]

# True compound regexes (match every linear in the named block, not
# just one projection name)
FAMILIES = {
    "attn_all": r"(\.q_proj|\.k_proj|\.v_proj|\.o_proj|attn_out|att_proj|query_key_value|attention\.dense)$",
    "mlp_all":  r"(\.gate_proj|\.up_proj|\.down_proj|ff_proj|\.ff_out)$",
}


def make_qcfg(include_regex: str) -> QuantConfig:
    return QuantConfig(method="turboldm", base_bits=3, residual_bits=0, block_size=128,
                        stage_count=4, sensitivity_metric="hidden_mse",
                        residual_top_fraction=0.0,
                        calibration_max_examples=24, probe_max_examples=3,
                        calibration_max_tokens=192,
                        include_regex=include_regex, exclude_regex=EXCL)


def run_one(family: str, dataset: str, n: int, seed: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"true_{family}_s{seed}"
    jsonl = OUT_DIR / "phase74_compound_modsens_3seeds.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 74 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(seed); set_matmul_precision()
    log(f"Phase 74 build {tag}/{dataset}")
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
        row["method"] = f"turboldm_3_{family}"
        row["family"] = family
        row["seed"] = seed
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 74 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 74, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for seed in SEEDS:
        for family in FAMILIES:
            for dataset, n in DATASETS:
                run_one(family, dataset, n, seed)
    log("Phase 74 done")


if __name__ == "__main__":
    main()
