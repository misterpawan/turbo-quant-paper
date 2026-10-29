# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 48: Dream-7B module-family sensitivity map.

Per-family isolated quant on Dream-7B-Instruct via its native
diffusion_generate decoder, mirroring Phase 30/42 on LLaDA-8B. Goal:
explain WHICH module family is responsible for Dream's PTQ fragility
(documented elsewhere as substantially worse than LLaDA at every bit
budget).
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict
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

# Dream uses Llama-style attention naming.
FAMILIES = {
    "q_proj":   r"\.q_proj$",
    "k_proj":   r"\.k_proj$",
    "v_proj":   r"\.v_proj$",
    "o_proj":   r"\.o_proj$",
    "attn_all": r"(\.q_proj|\.k_proj|\.v_proj|\.o_proj)$",
    "gate_proj": r"\.gate_proj$",
    "up_proj":   r"\.up_proj$",
    "down_proj": r"\.down_proj$",
    "mlp_all":   r"(\.gate_proj|\.up_proj|\.down_proj)$",
}

DATASETS = [("mmlu", 200), ("arc_challenge", 200), ("hellaswag", 200), ("gsm8k", 100)]


def load_dream():
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DREAM_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    return model, tok


@torch.no_grad()
def dream_generate_letter(model, tok, prompt: str, max_new_tokens: int, steps: int) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    out = model.diffusion_generate(ids, max_new_tokens=max_new_tokens, steps=steps)
    seq = out if isinstance(out, torch.Tensor) else out.sequences
    gen = seq[0, ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def evaluate_mcq(model, tok, ds_name, n, steps=16):
    ds = load_hf_dataset(ds_name, max_examples=n)
    correct = 0; total = 0
    t0 = time.perf_counter()
    for ex in ds:
        p, label = format_mcq_example(ds_name, ex)
        out = dream_generate_letter(model, tok, p, max_new_tokens=8, steps=steps)
        pred = parse_choice_answer(out, num_choices=4)
        if pred == label:
            correct += 1
        total += 1
    return {"score_value": correct / max(total, 1), "num_examples": total,
            "wall_sec": time.perf_counter() - t0}


def evaluate_gsm8k(model, tok, n, steps=64):
    ds = load_hf_dataset("gsm8k", max_examples=n)
    correct = 0; total = 0
    t0 = time.perf_counter()
    for ex in ds:
        p, ans = format_gsm8k_example(ex)
        gold = normalize_number_text(ans)
        out = dream_generate_letter(model, tok, p, max_new_tokens=128, steps=steps)
        if normalize_number_text(out) == gold:
            correct += 1
        total += 1
    return {"score_value": correct / max(total, 1), "num_examples": total,
            "wall_sec": time.perf_counter() - t0}


def run_one(family: str, regex: str, dataset: str, n_examples: int) -> None:
    tag = f"dream_mod_{family}_s42"
    jsonl = OUT_DIR / "phase48_dream_modsens.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 48 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=0, block_size=128,
        stage_count=4, sensitivity_metric="hidden_mse", residual_top_fraction=0.0,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192,
        include_regex=regex, exclude_regex=DREAM_EXCL,
    )
    seed_everything(42); set_matmul_precision()
    log(f"Phase 48 build {tag}/{dataset}")
    clear_cuda()
    try:
        model, tok = load_dream()
        stage_batches = prepare_stage_batches_for_model(model, tok, DREAM_SPEC, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, DREAM_SPEC, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        bits = avg_bits_per_param(model)
        storage = model_parameter_storage_bytes(model) / 1024**3
        log(f"  bits={bits:.2f} storage={storage:.2f}GB")
        maybe_reset_cuda_peak()
        if dataset == "gsm8k":
            res = evaluate_gsm8k(model, tok, n_examples, steps=64)
        else:
            res = evaluate_mcq(model, tok, dataset, n_examples, steps=16)
        row = {
            "model": "dream_7b_instruct", "baseline": tag, "dataset": dataset,
            "score_value": res["score_value"], "num_examples": res["num_examples"],
            "wall_sec": res["wall_sec"],
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
            "avg_bits_per_param": bits, "param_storage_gb": storage,
            "family": family,
            "quant_cfg": asdict(qcfg),
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score_value']:.3f} wall={res['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 48 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 48, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()


def main() -> None:
    for family, regex in FAMILIES.items():
        for dataset, n in DATASETS:
            run_one(family, regex, dataset, n)
    log("Phase 48 done")


if __name__ == "__main__":
    main()
