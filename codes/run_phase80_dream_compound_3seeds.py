# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 80: Dream-7B 3-seed corrected compound module sensitivity.

The Dream-7B per-family map is single seed; the abstract's
"compound-attention reproduces on Dream" claim should be CI-backed
to match the LLaDA promotion. This phase reruns just two cells
(true attn_all and mlp_all at 3-bit base) at seeds {17, 42, 101}
on the four-task n=500/300 grid.
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

DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 300)]
SEEDS = [17, 42, 101]

FAMILIES = {
    "attn_all": r"(\.q_proj|\.k_proj|\.v_proj|\.o_proj)$",
    "mlp_all":  r"(\.gate_proj|\.up_proj|\.down_proj)$",
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


def make_qcfg(include_regex: str) -> QuantConfig:
    return QuantConfig(method="turboldm", base_bits=3, residual_bits=0, block_size=128,
                        stage_count=4, sensitivity_metric="hidden_mse",
                        residual_top_fraction=0.0,
                        calibration_max_examples=24, probe_max_examples=3,
                        calibration_max_tokens=192,
                        include_regex=include_regex, exclude_regex=DREAM_EXCL)


def run_one(family: str, dataset: str, n: int, seed: int) -> None:
    tag = f"dream_true_{family}_s{seed}"
    jsonl = OUT_DIR / "phase80_dream_compound_3seeds.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 80 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(seed); set_matmul_precision()
    log(f"Phase 80 build {tag}/{dataset}")
    clear_cuda()
    qcfg = make_qcfg(include_regex=FAMILIES[family])
    try:
        model, tok = load_dream()
        stage_batches = prepare_stage_batches_for_model(model, tok, DREAM_SPEC, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, DREAM_SPEC, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        bits = avg_bits_per_param(model)
        storage = model_parameter_storage_bytes(model) / 1024**3
        log(f"  bits={bits:.2f} storage={storage:.2f}GiB")
        maybe_reset_cuda_peak()
        res = evaluate(model, tok, dataset, n)
        row = {
            "model": "dream_7b_instruct", "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "method": f"turboldm_3_{family}",
            "family": family, "seed": seed,
            "avg_bits_per_param": bits, "param_storage_gb": storage,
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 80 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 80, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()


def main() -> None:
    for seed in SEEDS:
        for family in FAMILIES:
            for dataset, n in DATASETS:
                run_one(family, dataset, n, seed)
    log("Phase 80 done")


if __name__ == "__main__":
    main()
