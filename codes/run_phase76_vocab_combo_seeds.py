# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 76: Multi-seed CI for the body+LM-head+QuantizedEmbedding combo.

The vocab-side combo run produced single-seed numbers for four
variants. The deployment headline ("body + head + embedding total
checkpoint X.X GiB at near-BF16 quality") needs paired CIs to be
defensible. This script reruns the body_head_embed variant at seeds
{17, 101} on the four-task n=500/300 grid; combined with the
seed-42 single-seed run, the aggregator can compute mean and
paired-bootstrap CIs.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    evaluate_dataset, load_model_and_tokenizer,
    prepare_stage_batches_for_model, quantize_model_inplace,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg
from run_phase39_embed import QuantizedEmbedding, replace_embedding

EXCL_BODY = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 300)]
SEEDS = [17, 101]


def make_qcfg() -> QuantConfig:
    return QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                        stage_count=4, sensitivity_metric="hidden_mse",
                        residual_top_fraction=0.15,
                        calibration_max_examples=24, probe_max_examples=3,
                        calibration_max_tokens=192, exclude_regex=EXCL_BODY,
                        include_lm_head=True)


def total_checkpoint_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def run_one(dataset: str, n: int, seed: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"turboldm_3p1_body_head_embed_s{seed}"
    jsonl = OUT_DIR / "phase76_vocab_combo_seeds.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 76 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(seed); set_matmul_precision()
    log(f"Phase 76 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    qcfg = make_qcfg()
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        n_emb = replace_embedding(model)
        log(f"  swapped {n_emb} embedding -> QuantizedEmbedding")
        bits = avg_bits_per_param(model)
        storage_quant = model_parameter_storage_bytes(model) / 1024**3
        storage_total = total_checkpoint_bytes(model) / 1024**3
        log(f"  bits={bits:.2f} quant={storage_quant:.2f}GiB total={storage_total:.2f}GiB")
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
        row["variant"] = "body_head_embed"
        row["seed"] = seed
        row["total_storage_gb"] = storage_total
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 76 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 76, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for seed in SEEDS:
        for dataset, n in DATASETS:
            run_one(dataset, n, seed)
    log("Phase 76 done")


if __name__ == "__main__":
    main()
