# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 73: Combined LM-head + 4-bit QuantizedEmbedding on top of
TurboLDM-(3+1) body.

The earlier vocab-side combo attempt (Phase 72) failed because the
helper's iter_named_linears does not enumerate nn.Embedding modules,
so the include/exclude regex switch was a no-op. This script does
it properly: it sets include_lm_head=True on the QuantConfig (the
helper's documented LM-head flag) and uses the QuantizedEmbedding
class from the QuantizedEmbedding pilot to swap nn.Embedding.

Variants:
  body_only          baseline TurboLDM-(3+1)
  body_head          + LM head quantised at the same recipe
  body_embed         + 4-bit QuantizedEmbedding
  body_head_embed    + both
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

# QuantizedEmbedding lives in the embedding-pilot driver
from run_phase39_embed import QuantizedEmbedding, replace_embedding

EXCL_BODY = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 300)]

VARIANTS = {
    "body_only":       {"include_lm_head": False, "swap_embedding": False},
    "body_head":       {"include_lm_head": True,  "swap_embedding": False},
    "body_embed":      {"include_lm_head": False, "swap_embedding": True},
    "body_head_embed": {"include_lm_head": True,  "swap_embedding": True},
}


def make_qcfg(include_lm_head: bool) -> QuantConfig:
    return QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                        stage_count=4, sensitivity_metric="hidden_mse",
                        residual_top_fraction=0.15,
                        calibration_max_examples=24, probe_max_examples=3,
                        calibration_max_tokens=192, exclude_regex=EXCL_BODY,
                        include_lm_head=include_lm_head)


def total_checkpoint_bytes(model: nn.Module) -> int:
    """Bytes summed over every parameter and registered buffer; covers
    quantised codes, scales, indices, and BF16 residual."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total


def run_one(variant: str, dataset: str, n: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"turboldm_3p1_{variant}_s42"
    jsonl = OUT_DIR / "phase73_vocab_combined.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 73 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 73 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    cfg = VARIANTS[variant]
    qcfg = make_qcfg(include_lm_head=cfg["include_lm_head"])
    try:
        # Body (and optionally LM head) via TurboLDM
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        # Embedding via QuantizedEmbedding (independent path)
        if cfg["swap_embedding"]:
            n_emb = replace_embedding(model)
            log(f"  swapped {n_emb} embedding module -> QuantizedEmbedding")
        bits = avg_bits_per_param(model)
        storage_quant = model_parameter_storage_bytes(model) / 1024**3
        storage_total = total_checkpoint_bytes(model) / 1024**3
        log(f"  variant={variant} bits={bits:.2f} quant_storage={storage_quant:.2f}GiB "
            f"total_storage={storage_total:.2f}GiB")
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
        log(f"  !! Phase 73 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 73, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for variant in VARIANTS:
        for dataset, n in DATASETS:
            run_one(variant, dataset, n)
    log("Phase 73 done")


if __name__ == "__main__":
    main()
