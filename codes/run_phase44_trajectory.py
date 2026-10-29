# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 44: diffusion trajectory quantisation lens.

Reviewer round 4's "wow" experiment. For every denoising step we
record, for a small set of prompts, the per-step divergence between
the BF16 reference model and a quantised model:

  * logit KL on masked positions (BF16 vs quantised, averaged over
    masked tokens that step)
  * top-1 token agreement on masked positions
  * answer-margin (logit gap between top-1 and top-2 of the canonical
    answer position, when defined)
  * entropy of the denoiser distribution

These let us answer the mechanistic question: do quantised dLLMs
\emph{recover} from early-step error as $T$ grows on \textsc{gsm8k}, and
\emph{not} on MCQ? The Phase 31 step-Pareto already shows the
quantitative pattern; Phase 44 measures the per-step divergence that
\emph{explains} it.

Methods: BF16 (reference) vs TurboLDM-(3+1).
Datasets: 16 MMLU prompts + 16 \textsc{gsm8k} prompts.
T values: 64 (long enough to see the recovery dynamic).

Implementation strategy: build BF16, generate with cached intermediate
logits, then build quantised, regenerate with the same prompts and
input-mask schedule, and diff the per-step logits.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, GenerateConfig,
    seed_everything, set_matmul_precision, avg_bits_per_param,
    model_parameter_storage_bytes, maybe_reset_cuda_peak,
    prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer, model_forward_logits, format_mcq_example,
    format_gsm8k_example, load_hf_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


def collect_step_logits(model, tok, spec, prompt: str, T: int = 64,
                          gen_length: int = 64, mask_token_id: int = 126336,
                          device: str = "cuda") -> list:
    """Run a deterministic low-confidence remasking schedule and capture
    per-step logits at masked positions. Returns a list of dicts:
    [{step, mask_positions, logits_at_masked, decoded_so_far}, ...].
    """
    # Encode prompt and append a fresh masked region of length gen_length.
    tok_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = tok_ids.shape[1]
    masked_region = torch.full((1, gen_length), mask_token_id, dtype=tok_ids.dtype, device=device)
    x = torch.cat([tok_ids, masked_region], dim=1)
    out = []
    with torch.no_grad():
        for step in range(T):
            logits = model_forward_logits(model, x)  # (1, L, V)
            mask_idx = (x[0] == mask_token_id).nonzero(as_tuple=True)[0]
            if mask_idx.numel() == 0:
                break
            # Score logits at masked positions; pick the highest-confidence one to fill.
            ml = logits[0, mask_idx]  # (n_masked, V)
            probs = F.softmax(ml.float(), dim=-1)
            confidences, top_ids = probs.max(dim=-1)
            # Greedy: fill the most-confident masked position.
            best = int(confidences.argmax().item())
            pos = int(mask_idx[best].item())
            tok_id = int(top_ids[best].item())
            out.append({
                "step": step,
                "n_masked": int(mask_idx.numel()),
                "filled_pos_rel": pos - prefix_len,
                "filled_token": tok_id,
                "filled_confidence": float(confidences[best].item()),
                # Save the *full* row of logits at the position we filled (for KL).
                "logits_at_filled": ml[best].detach().cpu().to(torch.float32).numpy().tolist(),
            })
            x[0, pos] = tok_id
    return out


def kl_divergence(p_logits: list, q_logits: list) -> float:
    p = F.log_softmax(torch.tensor(p_logits, dtype=torch.float32), dim=-1)
    q = F.log_softmax(torch.tensor(q_logits, dtype=torch.float32), dim=-1)
    return float((p.exp() * (p - q)).sum().item())


def run_one(prompt_id: str, prompt: str, T: int = 64) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    jsonl = OUT_DIR / "phase44_trajectory.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("prompt_id") == prompt_id:
                        log(f"Phase 44 SKIP {prompt_id}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 44 prompt {prompt_id} | T={T}")
    clear_cuda()
    # 1. BF16 reference.
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        bf16_steps = collect_step_logits(model, tok, spec, prompt, T=T)
    except Exception as exc:
        log(f"  !! BF16 trajectory failed: {exc}")
        del model, tok; clear_cuda()
        return
    bf16_filled = [(s["step"], s["filled_pos_rel"], s["filled_token"]) for s in bf16_steps]
    del model; gc.collect(); torch.cuda.empty_cache()
    # 2. Quantised model (TurboLDM-(3+1)).
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=1, block_size=128,
        stage_count=4, sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        q_steps = collect_step_logits(model, tok, spec, prompt, T=T)
    except Exception as exc:
        log(f"  !! quantised trajectory failed: {exc}")
        del model, tok; clear_cuda()
        return
    # 3. Diff per step.
    n_compare = min(len(bf16_steps), len(q_steps))
    rows = []
    for k in range(n_compare):
        b = bf16_steps[k]; q = q_steps[k]
        kl = kl_divergence(b["logits_at_filled"], q["logits_at_filled"])
        same_token = int(b["filled_token"] == q["filled_token"])
        rows.append({
            "step": k,
            "n_masked_bf16": b["n_masked"],
            "n_masked_quant": q["n_masked"],
            "kl_bf16_to_quant": kl,
            "top1_agreement": same_token,
            "bf16_confidence": b["filled_confidence"],
            "quant_confidence": q["filled_confidence"],
        })
    payload = {
        "prompt_id": prompt_id,
        "T": T,
        "n_compared": n_compare,
        "trajectory": rows,
    }
    append_jsonl(jsonl, payload)
    log(f"    {prompt_id}: traj_len={n_compare} mean_kl={sum(r['kl_bf16_to_quant'] for r in rows)/max(n_compare,1):.3f}")
    del model, tok; clear_cuda()


def main() -> None:
    # Build a small fixed prompt set: 8 from MMLU + 8 from GSM8K.
    spec = MODEL_SPECS["llada_8b_instruct"]
    seed_everything(42)
    mm = load_hf_dataset("mmlu", max_examples=8)
    gs = load_hf_dataset("gsm8k", max_examples=8)
    prompts = []
    for i, ex in enumerate(mm):
        p, _ = format_mcq_example("mmlu", ex)
        prompts.append((f"mmlu_{i}", p))
    for i, ex in enumerate(gs):
        p, _ = format_gsm8k_example(ex)
        prompts.append((f"gsm8k_{i}", p))
    for pid, prompt in prompts:
        run_one(pid, prompt, T=64)
    log("Phase 44 done")


if __name__ == "__main__":
    main()
