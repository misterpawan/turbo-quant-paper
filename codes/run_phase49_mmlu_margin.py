# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 49: per-step logit margin at the MMLU answer-letter position.

Phase 44 measured per-step KL at the position the model fills, but for
MCQ the answer is a single letter at the end of the prompt. This phase
tracks, at every denoising step, the BF16 vs TurboLDM-(3+1) logits at
the canonical answer-letter position (ABCD). Should show that the
answer is decided very early, explaining the step-count flatness on
MMLU.
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
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer, model_forward_logits, format_mcq_example,
    load_hf_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
MASK_ID = 126336


def collect_answer_margin(model, tok, prompt: str, T: int = 64,
                          gen_length: int = 8, device: str = "cuda") -> list:
    """Run the model and at each step record the logit margin between the
    top answer letter (ABCD) and the runner-up at the *first* masked
    position (i.e.\ the answer letter slot)."""
    enc = tok(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = enc.shape[1]
    masked = torch.full((1, gen_length), MASK_ID, dtype=enc.dtype, device=device)
    x = torch.cat([enc, masked], dim=1)
    abcd_ids = [tok.encode(c, add_special_tokens=False)[-1] for c in ("A", "B", "C", "D")]
    answer_pos = prefix_len  # the first masked position is the answer letter slot
    margins = []
    with torch.no_grad():
        for step in range(T):
            logits = model_forward_logits(model, x)
            row = logits[0, answer_pos].float()
            probs = F.softmax(row, dim=-1)
            abcd_probs = [float(probs[i].item()) for i in abcd_ids]
            top_letter = max(range(4), key=lambda i: abcd_probs[i])
            top_prob = abcd_probs[top_letter]
            second = sorted(abcd_probs, reverse=True)[1]
            margins.append({
                "step": step,
                "top_letter": "ABCD"[top_letter],
                "top_prob": top_prob,
                "margin": top_prob - second,
                "row_entropy": float(-(probs * (probs + 1e-12).log()).sum().item()),
            })
            # Greedy-fill the most confident position (mimics low-confidence remasking).
            mask_idx = (x[0] == MASK_ID).nonzero(as_tuple=True)[0]
            if mask_idx.numel() == 0: break
            ml = logits[0, mask_idx].float()
            mp = F.softmax(ml, dim=-1)
            confidences, top_ids = mp.max(dim=-1)
            best = int(confidences.argmax().item())
            x[0, int(mask_idx[best].item())] = int(top_ids[best].item())
    return margins


def run_one(prompt_id: str, prompt: str, T: int = 64) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    jsonl = OUT_DIR / "phase49_mmlu_margin.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("prompt_id") == prompt_id:
                        log(f"Phase 49 SKIP {prompt_id}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 49 prompt {prompt_id} | T={T}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        bf16_margins = collect_answer_margin(model, tok, prompt, T=T)
    except Exception as exc:
        log(f"  !! BF16 margin failed: {exc}")
        del model, tok; clear_cuda(); return
    del model; gc.collect(); torch.cuda.empty_cache()
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
        q_margins = collect_answer_margin(model, tok, prompt, T=T)
    except Exception as exc:
        log(f"  !! quantised margin failed: {exc}")
        del model, tok; clear_cuda(); return
    n = min(len(bf16_margins), len(q_margins))
    payload = {
        "prompt_id": prompt_id,
        "T": T,
        "bf16_margins": bf16_margins[:n],
        "quant_margins": q_margins[:n],
    }
    append_jsonl(jsonl, payload)
    bf_final = bf16_margins[-1]["top_letter"] if bf16_margins else "?"
    q_final = q_margins[-1]["top_letter"] if q_margins else "?"
    log(f"    {prompt_id}: BF16->{bf_final} quant->{q_final} (agree={bf_final==q_final})")
    del model, tok; clear_cuda()


def main() -> None:
    seed_everything(42)
    mm = load_hf_dataset("mmlu", max_examples=16)
    for i, ex in enumerate(mm):
        p, _ = format_mcq_example("mmlu", ex)
        run_one(f"mmlu_{i}", p, T=64)
    log("Phase 49 done")


if __name__ == "__main__":
    main()
