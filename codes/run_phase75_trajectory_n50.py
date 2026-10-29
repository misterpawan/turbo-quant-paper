# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 75: Trajectory quantisation lens at n=50 per task.

Expanded version of the existing trajectory-lens analysis: 50 MMLU
prompts and 50 GSM8K prompts, BF16 reference vs TurboLDM-(3+1) at
T=64, capturing per-step KL on the most-confident masked position.
The reviewer flagged that the n=8 version is too small to support
the "explains" framing for the MCQ vs GSM8K difference; this run
either solidifies that claim or forces a hedge to "suggests".

Reuses the trajectory-collection helpers from the existing
trajectory-lens driver to keep the schedule and KL definitions
identical.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
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
    load_model_and_tokenizer, format_mcq_example, format_gsm8k_example,
    load_hf_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda
# Reuse the existing trajectory-collector (per-step logits at the most-confident position)
from run_phase44_trajectory import collect_step_logits, kl_divergence

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
N_MMLU = 50
N_GSM8K = 50
T_VALUE = 64


def collect_prompts(dataset: str, n: int) -> list:
    ds = load_hf_dataset(dataset, max_examples=n)
    out = []
    for i, ex in enumerate(ds):
        if dataset == "gsm8k":
            p, _ = format_gsm8k_example(ex)
        else:
            p, _ = format_mcq_example(dataset, ex)
        out.append((f"{dataset}_{i}", p))
    return out


def main() -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    seed_everything(42); set_matmul_precision()
    jsonl = OUT_DIR / "phase75_trajectory_n50.jsonl"
    prompts = collect_prompts("mmlu", N_MMLU) + collect_prompts("gsm8k", N_GSM8K)

    # 1. BF16 first: collect bf16 step logits for every prompt, then free.
    log("Phase 75 build BF16")
    clear_cuda()
    bf16, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    bf16_data = {}
    for pid, prompt in prompts:
        try:
            bf16_data[pid] = collect_step_logits(bf16, tok, spec, prompt, T=T_VALUE)
            log(f"  bf16 {pid}: {len(bf16_data[pid])} steps")
        except Exception as exc:
            log(f"  !! bf16 {pid} failed: {exc}")
    del bf16; gc.collect(); torch.cuda.empty_cache()

    # 2. TurboLDM-(3+1): build, then run one prompt at a time.
    log("Phase 75 build TurboLDM-(3+1)")
    qmodel, tok2 = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    qcfg = QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                        stage_count=4, sensitivity_metric="hidden_mse",
                        residual_top_fraction=0.15,
                        calibration_max_examples=24, probe_max_examples=3,
                        calibration_max_tokens=192, exclude_regex=EXCL)
    try:
        stage_batches = prepare_stage_batches_for_model(qmodel, tok2, spec, qcfg, RunConfig())
        qmodel = quantize_model_inplace(qmodel, tok2, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        log(f"!! Phase 75 quant build failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 75, "tag": "build",
                                                  "err": str(exc), "trace": traceback.format_exc()})
        return

    # 3. Compare per prompt.
    for pid, prompt in prompts:
        if pid not in bf16_data:
            continue
        # SKIP if already logged
        skip = False
        if jsonl.exists():
            with jsonl.open() as f:
                for line in f:
                    try:
                        if json.loads(line).get("prompt_id") == pid:
                            skip = True; break
                    except Exception:
                        continue
        if skip:
            log(f"Phase 75 SKIP {pid}"); continue
        try:
            q_steps = collect_step_logits(qmodel, tok2, spec, prompt, T=T_VALUE)
        except Exception as exc:
            log(f"  !! quant {pid} failed: {exc}")
            continue
        bf16_steps = bf16_data[pid]
        n_compare = min(len(bf16_steps), len(q_steps))
        rows = []
        for k in range(n_compare):
            b = bf16_steps[k]; q = q_steps[k]
            kl = kl_divergence(b["logits_at_filled"], q["logits_at_filled"])
            same_token = int(b["filled_token"] == q["filled_token"])
            rows.append({"step": k, "kl_bf16_to_quant": kl, "top1_agreement": same_token,
                         "bf16_confidence": b["filled_confidence"],
                         "quant_confidence": q["filled_confidence"]})
        summary = {
            "prompt_id": pid, "T": T_VALUE, "method": "turboldm_3p1",
            "mean_kl": sum(r["kl_bf16_to_quant"] for r in rows) / max(len(rows), 1),
            "mean_top1_agree": sum(r["top1_agreement"] for r in rows) / max(len(rows), 1),
            "mean_kl_early": sum(r["kl_bf16_to_quant"] for r in rows[:T_VALUE//4]) / max(T_VALUE//4, 1),
            "mean_kl_late":  sum(r["kl_bf16_to_quant"] for r in rows[3*T_VALUE//4:]) / max(T_VALUE//4, 1),
            "n_steps": len(rows), "per_step": rows,
        }
        append_jsonl(jsonl, summary)
        log(f"  {pid} -> mean_kl={summary['mean_kl']:.3f} agree={summary['mean_top1_agree']:.3f} "
            f"early={summary['mean_kl_early']:.3f} late={summary['mean_kl_late']:.3f}")
    log("Phase 75 done")


if __name__ == "__main__":
    main()
