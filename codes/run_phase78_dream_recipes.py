# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 78: Dream-7B mixed-bit recipes at n=500.

Mirrors the LLaDA-8B mixed-bit-recipe sweep (Phase 43) on Dream-7B.
Tests whether the recipe ranking that holds on LLaDA also holds on a
second masked-diffusion 7B decoder. Single seed at n=500 to keep
runtime within the overnight budget; if interesting, multi-seed CIs
can be added later.

Recipes (matched to Phase 43 on LLaDA):
  attn3_mlp4
  attn4_mlp3
  attn3_upgate3_down4
  attn2_mlp4
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
    ModelSpec, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    iter_named_linears, get_parent_module,
    UniformLinear, uniform_group_quantize, replace_module,
    load_hf_dataset, format_mcq_example, parse_choice_answer,
    normalize_number_text, format_gsm8k_example,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

# Reuse the LLaDA-style family classifier and quantize_mixed
from run_phase43_recipes_ci import family_of, quantize_mixed, RECIPES

from transformers import AutoModel, AutoTokenizer

DREAM_ID = "Dream-org/Dream-v0-Instruct-7B"
DREAM_SPEC = ModelSpec(
    key="dream_7b_instruct", model_id=DREAM_ID,
    model_cls="AutoModel", trust_remote_code=True, dtype="bfloat16",
    default_steps=16, default_block_length=32, default_gen_length=32,
    uses_chat_template=True, mask_token_hint=151666,
    requires_left_padding=False,
)

DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]


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


def evaluate(model, tok, ds_name: str, n: int) -> dict:
    is_mcq = ds_name != "gsm8k"
    if is_mcq:
        steps = 16; max_new = 8
    else:
        steps = 64; max_new = 128
    ds = load_hf_dataset(ds_name, max_examples=n)
    correct = 0; total = 0
    t0 = time.perf_counter()
    for ex in ds:
        if is_mcq:
            p, lab = format_mcq_example(ds_name, ex)
        else:
            p, lab = format_gsm8k_example(ex)
            lab = normalize_number_text(lab)
        out = dream_generate(model, tok, p, max_new_tokens=max_new, steps=steps)
        pred = parse_choice_answer(out, num_choices=4) if is_mcq else normalize_number_text(out)
        if pred == lab:
            correct += 1
        total += 1
    return {"score": correct / max(total, 1), "n": total,
            "wall": time.perf_counter() - t0}


def run_one(recipe: str, dataset: str, n: int) -> None:
    tag = f"{recipe}_s42"
    jsonl = OUT_DIR / "phase78_dream_recipes.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 78 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 78 build {tag}/{dataset}")
    clear_cuda()
    try:
        model, tok = load_dream()
        model = quantize_mixed(model, RECIPES[recipe])
        gc.collect(); torch.cuda.empty_cache()
        bits = avg_bits_per_param(model)
        storage = model_parameter_storage_bytes(model) / 1024**3
        log(f"  bits={bits:.2f} storage={storage:.2f}GiB")
        maybe_reset_cuda_peak()
        res = evaluate(model, tok, dataset, n)
        row = {
            "model": "dream_7b_instruct", "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "recipe": recipe, "seed": 42,
            "avg_bits_per_param": bits, "param_storage_gb": storage,
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 78 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 78, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()


def main() -> None:
    for recipe in RECIPES:
        for dataset, n in DATASETS:
            run_one(recipe, dataset, n)
    log("Phase 78 done")


if __name__ == "__main__":
    main()
