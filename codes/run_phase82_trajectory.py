# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 82: per-step trajectory and entropy diff under quantisation (LLaDA-8B).

For 32 fixed prompts (16 MMLU + 16 GSM8K), run a deterministic
low-confidence remasking decode for T=64 steps under three precisions:
BF16, TurboLDM (4+1), TurboLDM (3+1). At every step record:

  * filled_pos_rel, filled_token, filled_confidence (the position the
    sampler greedily commits to)
  * n_masked (remaining masked positions)
  * mean_entropy_at_masked (mean per-position predicted entropy across
    every still-masked position -- not just the one we fill)
  * top1_at_masked (top-1 token id at every masked position; lets us
    compute per-step Jaccard / position-agreement against BF16 offline)
  * logits_at_filled (full vocabulary logits at the filled position;
    for KL diff against BF16)

Output one JSON line per (prompt_id, model_tag) into
runs/overnight/phase82_trajectory.jsonl. Offline analysis computes
KL(BF16||quant) at filled positions, unmask-position Jaccard, and
mean-entropy curves vs stage.

Models are built once each (not per prompt), so the bulk of wall time
is the 32x64 forward passes per model rather than rebuild overhead.
"""
from __future__ import annotations

import os, sys, time, gc, json, math, traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer, model_forward_logits, format_mcq_example,
    format_gsm8k_example, load_hf_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
T = 64
GEN_LEN = 64
N_MMLU = 16
N_GSM8K = 16
MASK_TOKEN_ID = 126336  # LLaDA-8B mask token

OUT_PATH = OUT_DIR / "phase82_trajectory.jsonl"


def collect_trajectory(model, tok, prompt: str, T: int = T,
                        gen_length: int = GEN_LEN, mask_token_id: int = MASK_TOKEN_ID,
                        device: str = "cuda") -> list:
    """Deterministic low-confidence remasking decode with rich per-step logs."""
    tok_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = tok_ids.shape[1]
    masked_region = torch.full((1, gen_length), mask_token_id, dtype=tok_ids.dtype, device=device)
    x = torch.cat([tok_ids, masked_region], dim=1)
    out = []
    with torch.no_grad():
        for step in range(T):
            logits = model_forward_logits(model, x)
            mask_idx = (x[0] == mask_token_id).nonzero(as_tuple=True)[0]
            if mask_idx.numel() == 0:
                break
            ml = logits[0, mask_idx].float()
            probs = F.softmax(ml, dim=-1)
            log_probs = F.log_softmax(ml, dim=-1)
            entropies = -(probs * log_probs).sum(dim=-1)
            mean_entropy = float(entropies.mean().item())
            confidences, top_ids = probs.max(dim=-1)
            best = int(confidences.argmax().item())
            pos = int(mask_idx[best].item())
            tok_id = int(top_ids[best].item())
            pos_rel_list = (mask_idx - prefix_len).tolist()
            top1_list = top_ids.tolist()
            out.append({
                "step": step,
                "n_masked": int(mask_idx.numel()),
                "filled_pos_rel": pos - prefix_len,
                "filled_token": tok_id,
                "filled_confidence": float(confidences[best].item()),
                "mean_entropy_at_masked": mean_entropy,
                "masked_positions_rel": pos_rel_list,
                "top1_at_masked": top1_list,
                "logits_at_filled": ml[best].cpu().to(torch.float32).numpy().tolist(),
            })
            x[0, pos] = tok_id
    return out


def run_model(spec, model_tag: str, qcfg, prompts: list) -> None:
    """Build one model variant, run all prompts, write trajectories, free."""
    seen = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add((r.get("model_tag"), r.get("prompt_id")))
                except Exception:
                    continue
    todo = [(pid, p) for (pid, p) in prompts if (model_tag, pid) not in seen]
    if not todo:
        log(f"Phase 82 SKIP all prompts for {model_tag}")
        return

    log(f"Phase 82 build {model_tag}")
    seed_everything(42); set_matmul_precision()
    clear_cuda()
    t0 = time.perf_counter()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    if qcfg is not None:
        try:
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
            model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(),
                                            stage_batches=stage_batches)
            del stage_batches; gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            log(f"  !! quantize failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl",
                         {"phase": 82, "model_tag": model_tag, "err": str(exc),
                          "trace": traceback.format_exc()})
            del model, tok; clear_cuda()
            return
    log(f"  built {model_tag} in {time.perf_counter()-t0:.1f}s")

    for pid, prompt in todo:
        try:
            traj = collect_trajectory(model, tok, prompt, T=T, gen_length=GEN_LEN)
            payload = {"model_tag": model_tag, "prompt_id": pid,
                       "T": T, "gen_length": GEN_LEN, "trajectory": traj}
            append_jsonl(OUT_PATH, payload)
            mean_h = sum(s["mean_entropy_at_masked"] for s in traj) / max(len(traj), 1)
            mean_c = sum(s["filled_confidence"] for s in traj) / max(len(traj), 1)
            log(f"    {model_tag}/{pid}: T={len(traj)} mean_H={mean_h:.3f} mean_conf={mean_c:.3f}")
        except Exception as exc:
            log(f"    !! {model_tag}/{pid} failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl",
                         {"phase": 82, "model_tag": model_tag, "prompt_id": pid,
                          "err": str(exc), "trace": traceback.format_exc()})

    del model, tok; clear_cuda()


def main() -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    seed_everything(42)
    mm = load_hf_dataset("mmlu", max_examples=N_MMLU)
    gs = load_hf_dataset("gsm8k", max_examples=N_GSM8K)
    prompts = []
    for i, ex in enumerate(mm):
        p, _ = format_mcq_example("mmlu", ex)
        prompts.append((f"mmlu_{i}", p))
    for i, ex in enumerate(gs):
        p, _ = format_gsm8k_example(ex)
        prompts.append((f"gsm8k_{i}", p))

    cfg_common = dict(
        method="turboldm", block_size=128, stage_count=4,
        stage_ratios=(0.85, 0.60, 0.35, 0.15),
        sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    variants = [
        ("bf16",          None),
        ("turboldm_4_1",  QuantConfig(base_bits=4, residual_bits=1, **cfg_common)),
        ("turboldm_3_1",  QuantConfig(base_bits=3, residual_bits=1, **cfg_common)),
    ]
    for tag, qcfg in variants:
        run_model(spec, tag, qcfg, prompts)
    log("Phase 82 done")


if __name__ == "__main__":
    main()
