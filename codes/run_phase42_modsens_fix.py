# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 42: rerun true attn_all and true mlp_all module-family sensitivity.

Reviewer round 4 noted that Phase 30's attn_all row collapses to
attn_out (regex artefact for the LLaDA-8B Llama-style block which uses
separate q/k/v/attn_out, not a fused att_proj). This phase reruns the
attn_all row using the correct compound regex, and also reruns mlp_all
to confirm. Single seed, n=500.
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

# True multi-projection regexes for LLaDA-8B (block_type=llama).
INC_ATTN_ALL = r"(q_proj|k_proj|v_proj|attn_out)"
INC_MLP_ALL  = r"(ff_proj|\.ff_out$)"

DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]


def run_one(family: str, regex: str, dataset: str, n_examples: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"mod_{family}_true_s42"
    jsonl = OUT_DIR / "phase42_modsens_true.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset):
                        log(f"Phase 42 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=0, block_size=128,
        stage_count=4, sensitivity_metric="hidden_mse", residual_top_fraction=0.0,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192,
        include_regex=regex,
        exclude_regex=r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)",
    )
    seed_everything(42); set_matmul_precision()
    log(f"Phase 42 build {tag}/{dataset} regex={regex}")
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
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 42 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 42, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for family, regex in [("attn_all", INC_ATTN_ALL), ("mlp_all", INC_MLP_ALL)]:
        for dataset, n in DATASETS:
            run_one(family, regex, dataset, n)
    log("Phase 42 done")


if __name__ == "__main__":
    main()
