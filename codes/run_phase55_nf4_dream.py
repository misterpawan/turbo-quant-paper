# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 55: NF4 on Dream-7B-Instruct.

NF4 (Phase 47) is currently the only row in the main table backed by
a real 4-bit kernel and matching VRAM win, but it has only been
measured on LLaDA-8B. This phase confirms the result generalises to
the second masked-diffusion family in the paper. Four datasets at
n=500, seed 42; aggregator can later add seeds 17 / 101 if the table
needs CIs.

Dream uses its native diffusion_generate decoder via AutoModel, so
this script mirrors Phase 48's per-prompt evaluation path rather than
the helper's batched evaluate_dataset (which targets LLaDA's chat
generate path).
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
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
DREAM_SPEC = ModelSpec(
    key="dream_7b_instruct", model_id=DREAM_ID,
    model_cls="AutoModel", trust_remote_code=True, dtype="bfloat16",
    default_steps=16, default_block_length=32, default_gen_length=32,
    uses_chat_template=True, mask_token_hint=151666,
    requires_left_padding=False,
)
DREAM_EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]


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


def run_one(dataset: str, n: int) -> None:
    tag = "nf4_s42"
    jsonl = OUT_DIR / "phase55_nf4_dream.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 55 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 55 build {tag}/{dataset}")
    clear_cuda()
    try:
        model, tok = load_dream()
        n_swapped = replace_with_nf4(model)
        log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
        gc.collect(); torch.cuda.empty_cache()
        maybe_reset_cuda_peak()
        res = evaluate(model, tok, dataset, n)
        row = {
            "model": "dream_7b_instruct", "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "method": "nf4", "seed": 42,
            "avg_bits_per_param": 4.0,
            "param_storage_gb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3,
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 55 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 55, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()


def main() -> None:
    for dataset, n in DATASETS:
        run_one(dataset, n)
    log("Phase 55 done")


if __name__ == "__main__":
    main()
