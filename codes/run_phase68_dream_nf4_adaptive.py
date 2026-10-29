# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 68: NF4 + adaptive-T on Dream-7B.

Closes the (model x method x policy) cube: Phase 50 had NF4 +
adaptive on LLaDA-8B; Phase 58 had TurboLDM-(3+1) + adaptive on
Dream. The missing cell is NF4 + adaptive on Dream. Tests whether
the Phase 50 finding (adaptive policy collapses on NF4) reproduces
across model families.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json, re
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import bitsandbytes as bnb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    ModelSpec, seed_everything, set_matmul_precision, maybe_reset_cuda_peak,
    iter_named_linears, get_parent_module,
    load_hf_dataset, format_mcq_example, parse_choice_answer,
    normalize_number_text, format_gsm8k_example,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

from transformers import AutoModel, AutoTokenizer

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
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


def replace_with_nf4(model: nn.Module) -> int:
    n = 0
    layer_names = [name for name, _ in iter_named_linears(model, exclude_regex=DREAM_EXCL)]
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


def run_one(dataset: str, n: int, policy: str) -> None:
    tag = f"nf4_{policy}"
    jsonl = OUT_DIR / "phase68_dream_nf4_adaptive.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 68 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 68 build {tag}/{dataset}")
    clear_cuda()
    try:
        model, tok = load_dream()
        n_swapped = replace_with_nf4(model)
        log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
        gc.collect(); torch.cuda.empty_cache()
        maybe_reset_cuda_peak()
        res = evaluate_with_policy(model, tok, dataset, n, policy)
        row = {
            "model": "dream_7b_instruct", "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "total_T": res["total_T"],
            "policy": policy, "method": "nf4", "seed": 42,
            "avg_bits_per_param": 4.0,
            "param_storage_gb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3,
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} total_T={res['total_T']} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 68 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 68, "tag": tag, "dataset": dataset,
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
    log("Phase 68 done")


if __name__ == "__main__":
    main()
