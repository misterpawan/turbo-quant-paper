# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 58: TurboLDM-3+1 + adaptive-T on Dream-7B.

Phase 46 paired the adaptive-T controller with the paper's flagship
TurboLDM-3+1 quantisation on LLaDA-8B. Phase 50 found NF4 does not
benefit from the adaptive policy. This phase tests whether the
TurboLDM-3+1 + adaptive combination transfers to the second
masked-diffusion family in the paper.

Methods x policies = 1 x 4 on (mmlu n=100, gsm8k n=50). Dream uses
its native diffusion_generate, so we copy the per-prompt generation
pattern from Phase 48 / 55.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json, re
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    ModelSpec, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    quantize_model_inplace, prepare_stage_batches_for_model,
    load_hf_dataset, format_mcq_example, parse_choice_answer,
    normalize_number_text, format_gsm8k_example,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

from transformers import AutoModel, AutoTokenizer

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
DREAM_SPEC = ModelSpec(
    key="dream_7b_instruct", model_id=DREAM_ID,
    model_cls="AutoModel", trust_remote_code=True, dtype="bfloat16",
    default_steps=16, default_block_length=32, default_gen_length=32,
    uses_chat_template=True, mask_token_hint=151666,
    requires_left_padding=False,
)
DREAM_EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

POLICIES = ["fixed_32", "fixed_64", "fixed_128", "adaptive"]
DATASETS = [("mmlu", 100), ("gsm8k", 50)]


def load_dream():
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DREAM_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    return model, tok


@torch.no_grad()
def dream_generate(model, tok, prompt: str, max_new_tokens: int, steps: int) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    out = model.diffusion_generate(ids, max_new_tokens=max_new_tokens, steps=steps)
    seq = out if isinstance(out, torch.Tensor) else out.sequences
    gen = seq[0, ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def is_confident_mcq(text: str) -> bool:
    pred = parse_choice_answer(text, num_choices=4)
    return pred is not None and pred in "ABCD"


def is_confident_gsm8k(text: str) -> bool:
    norm = normalize_number_text(text)
    return bool(norm) and norm not in ("", "0") and re.fullmatch(r"[-+]?\d+(\.\d+)?", norm) is not None


def evaluate_with_policy(model, tok, dataset: str, n: int, policy: str) -> dict:
    is_mcq = dataset != "gsm8k"
    max_new = 8 if is_mcq else 128
    ds = load_hf_dataset(dataset, max_examples=n)
    prompts, labels = [], []
    for ex in ds:
        if is_mcq:
            p, lab = format_mcq_example(dataset, ex)
        else:
            p, lab = format_gsm8k_example(ex)
            lab = normalize_number_text(lab)
        prompts.append(p); labels.append(lab)

    def gen_at(T_value: int, prompt: str) -> str:
        return dream_generate(model, tok, prompt, max_new_tokens=max_new, steps=T_value)

    correct = 0; total = 0; total_T = 0
    t0 = time.perf_counter()
    if policy == "adaptive":
        for prompt, label in zip(prompts, labels):
            out = gen_at(32, prompt); total_T += 32
            confident = is_confident_mcq(out) if is_mcq else is_confident_gsm8k(out)
            if not confident:
                out = gen_at(64, prompt); total_T += 64
                confident = is_confident_mcq(out) if is_mcq else is_confident_gsm8k(out)
                if not confident:
                    out = gen_at(128, prompt); total_T += 128
            pred = parse_choice_answer(out, num_choices=4) if is_mcq else normalize_number_text(out)
            if pred == label:
                correct += 1
            total += 1
    else:
        T_value = int(policy.split("_")[1])
        for prompt, label in zip(prompts, labels):
            out = gen_at(T_value, prompt); total_T += T_value
            pred = parse_choice_answer(out, num_choices=4) if is_mcq else normalize_number_text(out)
            if pred == label:
                correct += 1
            total += 1
    return {"score": correct / max(total, 1), "n": total, "total_T": total_T,
            "wall": time.perf_counter() - t0}


QCFG = QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                    stage_count=4, sensitivity_metric="hidden_mse",
                    residual_top_fraction=0.15,
                    calibration_max_examples=24, probe_max_examples=3,
                    calibration_max_tokens=192, exclude_regex=DREAM_EXCL)


def run_one(dataset: str, n: int, policy: str) -> None:
    tag = f"turboldm_3p1_{policy}"
    jsonl = OUT_DIR / "phase58_dream_adaptive.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 58 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 58 build {tag}/{dataset}")
    clear_cuda()
    try:
        model, tok = load_dream()
        stage_batches = prepare_stage_batches_for_model(model, tok, DREAM_SPEC, QCFG, RunConfig())
        model = quantize_model_inplace(model, tok, DREAM_SPEC, QCFG, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        bits = avg_bits_per_param(model)
        storage = model_parameter_storage_bytes(model) / 1024**3
        log(f"  bits={bits:.2f} storage={storage:.2f}GB")
        maybe_reset_cuda_peak()
        res = evaluate_with_policy(model, tok, dataset, n, policy)
        row = {
            "model": "dream_7b_instruct", "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "total_T": res["total_T"],
            "policy": policy, "method": "turboldm_3p1", "seed": 42,
            "avg_bits_per_param": bits, "param_storage_gb": storage,
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} total_T={res['total_T']} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 58 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 58, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()


def main() -> None:
    for dataset, n in DATASETS:
        for policy in POLICIES:
            run_one(dataset, n, policy)
    log("Phase 58 done")


if __name__ == "__main__":
    main()
