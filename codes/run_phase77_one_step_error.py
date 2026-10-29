# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 77: Causal one-step weight error decomposition.

Reviewer-recommended "wow" experiment in lightweight form. We run a
single BF16 reference trajectory per prompt, then for each method
(AWQ-4, NF4, TurboLDM-(3+1)) we re-evaluate the *same* per-step input
state through the method's model and log the per-step KL between
BF16(x_k) and method(x_k). This isolates *one-step weight error*
(the divergence introduced by the weight perturbation alone, holding
the diffusion state fixed at the BF16 trajectory) from accumulated
state drift (which would require the method to take its own steps).

Output is a per-(prompt, step, method) JSONL row; downstream analysis
plots one-step KL vs step for each method, separately for MMLU and
GSM8K.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    iter_named_linears, get_parent_module,
    prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer, model_forward_logits,
    apply_awq_fallback, format_mcq_example, format_gsm8k_example, load_hf_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
MASK_TOKEN_ID = 126336
N_MMLU = 50
N_GSM8K = 50
T_VALUE = 64
GEN_LENGTH = 64


def collect_bf16_trajectory(model, tok, prompt: str, T: int = T_VALUE,
                              device: str = "cuda") -> list:
    """Return list of dicts per step: {x_state, fill_position, bf16_logits_at_pos}."""
    tok_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prefix_len = tok_ids.shape[1]
    masked = torch.full((1, GEN_LENGTH), MASK_TOKEN_ID, dtype=tok_ids.dtype, device=device)
    x = torch.cat([tok_ids, masked], dim=1)
    saved = []
    with torch.no_grad():
        for step in range(T):
            mask_idx = (x[0] == MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
            if mask_idx.numel() == 0:
                break
            x_snapshot = x.clone()
            logits = model_forward_logits(model, x)
            ml = logits[0, mask_idx]
            probs = F.softmax(ml.float(), dim=-1)
            confidences, top_ids = probs.max(dim=-1)
            best = int(confidences.argmax().item())
            pos = int(mask_idx[best].item())
            tok_id = int(top_ids[best].item())
            saved.append({
                "step": step,
                "x": x_snapshot,
                "fill_pos": pos,
                "bf16_logits": ml[best].detach().cpu().to(torch.float32).numpy().tolist(),
            })
            x[0, pos] = tok_id
    return saved


def kl_logits(p_logits, q_logits) -> float:
    p = F.log_softmax(torch.tensor(p_logits, dtype=torch.float32), dim=-1)
    q = F.log_softmax(torch.tensor(q_logits, dtype=torch.float32), dim=-1)
    return float((p.exp() * (p - q)).sum().item())


def replay_method_logits(method_model, saved: list) -> list:
    """For each saved step, run method_model.forward on x_snapshot and log
    the row of logits at fill_pos."""
    out = []
    with torch.no_grad():
        for s in saved:
            logits = model_forward_logits(method_model, s["x"])
            row = logits[0, s["fill_pos"]].detach().cpu().to(torch.float32).numpy().tolist()
            out.append(row)
    return out


def replace_with_nf4(model: nn.Module) -> int:
    n = 0
    layer_names = [name for name, _ in iter_named_linears(model, exclude_regex=EXCL)]
    for name in layer_names:
        parent, leaf = get_parent_module(model, name)
        linear = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        bnb_layer = bnb.nn.Linear4bit(
            linear.in_features, linear.out_features,
            bias=linear.bias is not None,
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
        )
        bnb_layer.load_state_dict(linear.state_dict())
        bnb_layer = bnb_layer.to(linear.weight.device)
        setattr(parent, leaf, bnb_layer)
        n += 1
    return n


METHODS = ["awq_4", "nf4", "turboldm_3p1"]


def build_method_model(method: str, spec):
    """Returns a fresh quantised model for the given method."""
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    if method == "nf4":
        replace_with_nf4(model)
        gc.collect(); torch.cuda.empty_cache()
    elif method == "awq_4":
        qcfg = QuantConfig(method="awq", w_bits=4, group_size=128,
                            calibration_max_examples=24, probe_max_examples=3,
                            calibration_max_tokens=256, exclude_regex=EXCL)
        sb = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=sb)
        del sb; gc.collect(); torch.cuda.empty_cache()
    else:  # turboldm_3p1
        qcfg = QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                            stage_count=4, sensitivity_metric="hidden_mse",
                            residual_top_fraction=0.15,
                            calibration_max_examples=24, probe_max_examples=3,
                            calibration_max_tokens=192, exclude_regex=EXCL)
        sb = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=sb)
        del sb; gc.collect(); torch.cuda.empty_cache()
    return model, tok


def collect_prompts(dataset: str, n: int) -> list:
    ds = load_hf_dataset(dataset, max_examples=n)
    out = []
    for i, ex in enumerate(ds):
        if dataset == "gsm8k":
            p, _ = format_gsm8k_example(ex)
        else:
            p, _ = format_mcq_example(dataset, ex)
        out.append((f"{dataset}_{i}", p, dataset))
    return out


def main() -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    seed_everything(42); set_matmul_precision()
    jsonl = OUT_DIR / "phase77_one_step_error.jsonl"
    prompts = collect_prompts("mmlu", N_MMLU) + collect_prompts("gsm8k", N_GSM8K)

    # 1. BF16 trajectory: collect per-step (x, fill_pos, logits) for every prompt.
    log("Phase 77 build BF16")
    clear_cuda()
    bf16, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    bf16_data = {}
    for pid, prompt, ds in prompts:
        try:
            saved = collect_bf16_trajectory(bf16, tok, prompt, T=T_VALUE)
            bf16_data[pid] = (saved, ds)
            log(f"  bf16 {pid}: {len(saved)} steps")
        except Exception as exc:
            log(f"  !! bf16 {pid} failed: {exc}")
    del bf16; gc.collect(); torch.cuda.empty_cache()

    # 2. For each method, replay each saved trajectory and log per-step KL.
    for method in METHODS:
        log(f"Phase 77 build {method}")
        clear_cuda()
        try:
            mmodel, _ = build_method_model(method, spec)
        except Exception as exc:
            log(f"  !! method {method} build failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 77, "tag": f"build_{method}",
                                                      "err": str(exc), "trace": traceback.format_exc()})
            continue
        for pid, (saved, ds) in bf16_data.items():
            try:
                replay = replay_method_logits(mmodel, saved)
            except Exception as exc:
                log(f"  !! replay {method}/{pid} failed: {exc}")
                continue
            kls = [kl_logits(s["bf16_logits"], replay[k]) for k, s in enumerate(saved)]
            row = {
                "prompt_id": pid, "dataset": ds, "method": method, "T": T_VALUE,
                "n_steps": len(kls),
                "mean_kl": sum(kls)/max(len(kls),1),
                "mean_kl_early": sum(kls[:T_VALUE//4])/max(T_VALUE//4,1),
                "mean_kl_late":  sum(kls[3*T_VALUE//4:])/max(T_VALUE//4,1),
                "per_step_kl": kls,
            }
            append_jsonl(jsonl, row)
            log(f"    {method}/{pid} mean_kl={row['mean_kl']:.3f}")
        del mmodel; gc.collect(); torch.cuda.empty_cache()
    log("Phase 77 done")


if __name__ == "__main__":
    main()
