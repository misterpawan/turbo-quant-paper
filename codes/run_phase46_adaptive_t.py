# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 46: adaptive denoising-step controller.

Trains a per-prompt budget controller that decides how many diffusion
steps T to spend by looking at the model's own confidence: start with
T=32, extend to T=64 if the answer is not parseable / margin too low,
extend again to T=128 if still uncertain.

We compare four policies on MMLU + GSM8K:
  fixed T=32, fixed T=64, fixed T=128, adaptive

Methods: BF16 reference + TurboLDM-(3+1) (the most accuracy-sensitive
quantised configuration in this paper).

Reports accuracy, total NFE (denoising-step budget summed over the
slice), wall-clock, accuracy/sec, and accuracy/NFE.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json, re
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, GenerateConfig,
    seed_everything, set_matmul_precision, avg_bits_per_param,
    model_parameter_storage_bytes, maybe_reset_cuda_peak,
    prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer, generate_text_batch,
    parse_choice_answer, normalize_number_text,
    format_mcq_example, format_gsm8k_example, load_hf_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


def is_confident_mcq(text: str, n_choices: int = 4) -> bool:
    pred = parse_choice_answer(text, num_choices=n_choices)
    return pred is not None and pred in "ABCD"[:n_choices]


def is_confident_gsm8k(text: str) -> bool:
    norm = normalize_number_text(text)
    return bool(norm) and norm not in ("", "0") and re.fullmatch(r"[-+]?\d+(\.\d+)?", norm) is not None


def evaluate_with_policy(model, tok, spec, dataset: str, n: int, policy: str) -> dict:
    """Policies: 'fixed_8'/'fixed_16'/'fixed_32'/'fixed_64'/'fixed_128' or 'adaptive'."""
    base_gen = mcq_fast_gen_cfg(spec, dataset)
    ds = load_hf_dataset(dataset, max_examples=n)
    prompts, labels = [], []
    is_mcq = dataset != "gsm8k"
    for ex in ds:
        if is_mcq:
            p, lab = format_mcq_example(dataset, ex)
        else:
            p, lab = format_gsm8k_example(ex)
            lab = normalize_number_text(lab)
        prompts.append(p); labels.append(lab)

    def gen_at(T_value: int, batch_prompts):
        cfg = replace(base_gen, steps=T_value)
        return generate_text_batch(model, tok, spec, list(batch_prompts), cfg, device="cuda")

    correct = 0; total = 0; total_T = 0
    t0 = time.perf_counter()
    if policy == "adaptive":
        # Per-prompt: T=32 first; extend to 64, then 128 if not confident.
        for prompt, label in zip(prompts, labels):
            out = gen_at(32, [prompt])[0]; total_T += 32
            confident = is_confident_mcq(out) if is_mcq else is_confident_gsm8k(out)
            if not confident:
                out = gen_at(64, [prompt])[0]; total_T += 64
                confident = is_confident_mcq(out) if is_mcq else is_confident_gsm8k(out)
                if not confident:
                    out = gen_at(128, [prompt])[0]; total_T += 128
            pred = parse_choice_answer(out, num_choices=4) if is_mcq else normalize_number_text(out)
            if pred == label:
                correct += 1
            total += 1
    else:
        T_value = int(policy.split("_")[1])
        for prompt, label in zip(prompts, labels):
            out = gen_at(T_value, [prompt])[0]; total_T += T_value
            pred = parse_choice_answer(out, num_choices=4) if is_mcq else normalize_number_text(out)
            if pred == label:
                correct += 1
            total += 1
    wall = time.perf_counter() - t0
    return {"score": correct / max(total, 1), "n": total, "total_T": total_T,
            "wall": wall}


METHODS = {
    "bf16":          QuantConfig(method="bf16"),
    "turboldm_3p1":  QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                   stage_count=4, sensitivity_metric="hidden_mse",
                                   residual_top_fraction=0.15,
                                   calibration_max_examples=24, probe_max_examples=3,
                                   calibration_max_tokens=192, exclude_regex=EXCL),
}

POLICIES = ["fixed_32", "fixed_64", "fixed_128", "adaptive"]
DATASETS = [("mmlu", 100), ("gsm8k", 50)]


def run_one(method: str, dataset: str, n: int, policy: str) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{method}_{policy}"
    jsonl = OUT_DIR / "phase46_adaptive_t.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 46 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 46 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    qcfg = METHODS[method]
    try:
        if qcfg.method != "bf16":
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
            model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
            del stage_batches; gc.collect(); torch.cuda.empty_cache()
        maybe_reset_cuda_peak()
        res = evaluate_with_policy(model, tok, spec, dataset, n, policy)
        row = {
            "model": spec.key, "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "total_T": res["total_T"],
            "policy": policy, "method": method,
            "avg_bits_per_param": avg_bits_per_param(model),
            "param_storage_gb": model_parameter_storage_bytes(model) / 1024**3,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} total_T={res['total_T']} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 46 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 46, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for method in METHODS:
        for dataset, n in DATASETS:
            for policy in POLICIES:
                run_one(method, dataset, n, policy)
    log("Phase 46 done")


if __name__ == "__main__":
    main()
