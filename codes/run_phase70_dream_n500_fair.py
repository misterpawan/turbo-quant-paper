# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 70: Dream-7B BF16/AWQ-4/TurboLDM-(3+1) at n=500 — fair comparison
with the Phase 55 NF4 row.

The reviewer flagged that the Dream NF4 numbers in tab:nf4-main are at
n=500, while the Dream BF16/AWQ/TurboLDM reference rows in the
appendix (phase 17) are at n=100 / n=150 / n=50. This phase produces
matched n=500 references for the same three methods so the cross-method
comparison on Dream-7B is on equal footing.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
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

DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]

METHODS = {
    "bf16": None,
    "awq_4": QuantConfig(method="awq", w_bits=4, group_size=128,
                          calibration_max_examples=24, probe_max_examples=3,
                          calibration_max_tokens=256, exclude_regex=DREAM_EXCL),
    "turboldm_3p1": QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                  stage_count=4, sensitivity_metric="hidden_mse",
                                  residual_top_fraction=0.15,
                                  calibration_max_examples=24, probe_max_examples=3,
                                  calibration_max_tokens=192, exclude_regex=DREAM_EXCL),
}


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


def run_one(method: str, dataset: str, n: int) -> None:
    tag = f"{method}_n500_s42"
    jsonl = OUT_DIR / "phase70_dream_n500_fair.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 70 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 70 build {tag}/{dataset}")
    clear_cuda()
    qcfg = METHODS[method]
    try:
        model, tok = load_dream()
        if qcfg is not None:
            stage_batches = prepare_stage_batches_for_model(model, tok, DREAM_SPEC, qcfg, RunConfig())
            model = quantize_model_inplace(model, tok, DREAM_SPEC, qcfg, RunConfig(), stage_batches=stage_batches)
            del stage_batches
            gc.collect(); torch.cuda.empty_cache()
            bits = avg_bits_per_param(model)
            storage = model_parameter_storage_bytes(model) / 1024**3
        else:
            bits, storage = 16.0, sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        log(f"  bits={bits:.2f} storage={storage:.2f}GB")
        maybe_reset_cuda_peak()
        res = evaluate(model, tok, dataset, n)
        row = {
            "model": "dream_7b_instruct", "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "method": method, "seed": 42,
            "avg_bits_per_param": bits, "param_storage_gb": storage,
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 70 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 70, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()


def main() -> None:
    for method in METHODS:
        for dataset, n in DATASETS:
            run_one(method, dataset, n)
    log("Phase 70 done")


if __name__ == "__main__":
    main()
