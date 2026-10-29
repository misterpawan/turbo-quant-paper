# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 45: architecture sensitivity fingerprint.

For each masked-diffusion model we have weights for, walk every linear
layer and compute weight-side statistics that correlate with PTQ
fragility:

  * weight max / RMS                (outlier ratio)
  * weight 99.9th percentile / RMS  (heavy-tailed-ness)
  * stable rank ||W||_F^2 / ||W||_2^2
  * mean abs / RMS                  (skewness of |w|)
  * fraction of |w| above 5*RMS     (large-outlier fraction)

These per-layer numbers are averaged within module families, giving a
fingerprint per (model, family). Phase 45 then ships a CSV that
downstream analysis can correlate with the per-family quantisation
loss measured in Phases 30, 42, 48.

Models: LLaDA-8B-Instruct, Dream-7B-Instruct, MDLM-OWT, LLaDA-MoE-7B.
LLaDA2.0-mini is skipped here to fit the GPU/disk budget; its
fingerprint can be added in a follow-up by reusing the lazy load path.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, csv
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, ModelSpec, seed_everything, set_matmul_precision,
    load_model_and_tokenizer, iter_named_linears, resolve_torch_dtype,
)
from run_overnight import OUT_DIR, log, clear_cuda

# Family-name regex per architecture (Llama-style and OLMo-style covered).
FAMILY_PATTERNS = [
    ("q_proj",   "q_proj"),
    ("k_proj",   "k_proj"),
    ("v_proj",   "v_proj"),
    ("o_proj",   "o_proj"),
    ("attn_out", "attn_out"),
    ("att_proj", "att_proj"),
    ("query_key_value", "query_key_value"),
    ("attention_dense", "attention.dense"),
    ("gate_proj", "gate_proj"),
    ("up_proj",   "up_proj"),
    ("down_proj", "down_proj"),
    ("ff_proj",   "ff_proj"),
    ("ff_out",    ".ff_out"),
]


def family_of(name: str) -> str:
    n = name.lower()
    for fam, key in FAMILY_PATTERNS:
        if key in n:
            return fam
    return "other"


def stats_of(w: torch.Tensor) -> dict:
    """Compute per-layer fingerprint stats from a 2D weight matrix."""
    w = w.detach().to(torch.float32)
    flat = w.flatten().abs()
    rms = float((w * w).mean().sqrt().item())
    if rms < 1e-10:
        return {"max_over_rms": 0, "p999_over_rms": 0, "stable_rank": 0,
                "mean_abs_over_rms": 0, "frac_above_5rms": 0,
                "out_features": w.shape[0], "in_features": w.shape[1]}
    max_abs = float(flat.max().item())
    # torch.quantile inputs are capped at ~16M elements; subsample for big layers.
    if flat.numel() > 10_000_000:
        idx = torch.randint(0, flat.numel(), (10_000_000,), device=flat.device)
        p999 = float(torch.quantile(flat[idx], 0.999).item())
    else:
        p999 = float(torch.quantile(flat, 0.999).item())
    fro_sq = float((w * w).sum().item())
    # Stable rank uses spectral norm; SVD is expensive on big matrices,
    # use power iteration-ish via a single iteration of randomised SVD.
    try:
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        spec = float(s.max().item())
    except Exception:
        spec = float(w.abs().max(dim=1).values.max().item())
    stable_rank = fro_sq / (spec * spec + 1e-12)
    mean_abs = float(flat.mean().item())
    frac_above_5rms = float((flat > 5 * rms).float().mean().item())
    return {
        "max_over_rms": max_abs / rms,
        "p999_over_rms": p999 / rms,
        "stable_rank": stable_rank,
        "mean_abs_over_rms": mean_abs / rms,
        "frac_above_5rms": frac_above_5rms,
        "out_features": w.shape[0],
        "in_features": w.shape[1],
    }


def fingerprint_one(model_key: str, spec: ModelSpec) -> list:
    log(f"Phase 45 fingerprint {model_key}")
    seed_everything(42); set_matmul_precision()
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    rows = []
    try:
        for name, lin in iter_named_linears(model):
            if not isinstance(lin.weight, torch.Tensor): continue
            try:
                s = stats_of(lin.weight)
            except torch.cuda.OutOfMemoryError:
                # Fall back to CPU for the SVD on huge layers.
                s = stats_of(lin.weight.detach().to("cpu"))
            s["model"] = model_key
            s["layer_name"] = name
            s["family"] = family_of(name)
            rows.append(s)
    finally:
        del model, tok; clear_cuda()
    log(f"  collected {len(rows)} layers from {model_key}")
    return rows


def main() -> None:
    out_csv = OUT_DIR / "phase45_arch_fingerprint.csv"
    if out_csv.exists():
        log(f"Phase 45 SKIP (output exists): {out_csv}")
        return
    candidates = [
        ("llada_8b_instruct", MODEL_SPECS["llada_8b_instruct"]),
        ("llada2_mini",       MODEL_SPECS["llada2_mini"]),
    ]
    # Also do Dream-7B and MDLM-OWT via inline ModelSpec so we don't depend
    # on MODEL_SPECS having entries for every family.
    from transformers import AutoModel
    candidates.append((
        "dream_7b_instruct",
        ModelSpec(key="dream_7b_instruct",
                   model_id="Dream-org/Dream-v0-Instruct-7B",
                   model_cls="AutoModel", trust_remote_code=True, dtype="bfloat16",
                   default_steps=16, default_block_length=32, default_gen_length=32,
                   uses_chat_template=True, mask_token_hint=151666,
                   requires_left_padding=False),
    ))
    if "llada_moe_7b_instruct" in MODEL_SPECS:
        candidates.append(("llada_moe_7b_instruct", MODEL_SPECS["llada_moe_7b_instruct"]))
    all_rows = []
    for k, s in candidates:
        try:
            all_rows.extend(fingerprint_one(k, s))
        except Exception as exc:
            log(f"  !! {k} fingerprint failed: {exc}")
    if not all_rows: return
    fields = ["model", "family", "layer_name", "out_features", "in_features",
              "max_over_rms", "p999_over_rms", "stable_rank",
              "mean_abs_over_rms", "frac_above_5rms"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fields})
    log(f"Phase 45 done; wrote {out_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
