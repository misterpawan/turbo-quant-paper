# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 65: TurboLDM-(4+1) step-budget ablation.

Mirrors Phase 60 (TurboLDM-(3+1) steps) and Phase 52 (NF4 steps) at
the higher TurboLDM bit-budget. Completes the step-budget panel so
the paper can compare three quantisations on the same T grid.
Runs the model once and re-evaluates at each T to amortise calibration.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    load_model_and_tokenizer,
    prepare_stage_batches_for_model, quantize_model_inplace,
    generate_text_batch, parse_choice_answer, normalize_number_text,
    format_mcq_example, format_gsm8k_example, load_hf_dataset,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"

GRID = [
    ("mmlu",  200, [2, 4, 8, 16, 32, 64]),
    ("gsm8k", 100, [16, 32, 64, 128]),
]

QCFG = QuantConfig(method="turboldm", base_bits=4, residual_bits=1, block_size=128,
                    stage_count=4, sensitivity_metric="hidden_mse",
                    residual_top_fraction=0.15,
                    calibration_max_examples=24, probe_max_examples=3,
                    calibration_max_tokens=192, exclude_regex=EXCL)


def evaluate_at_T(model, tok, spec, dataset: str, n: int, T_value: int) -> dict:
    base_gen = mcq_fast_gen_cfg(spec, dataset)
    cfg = replace(base_gen, steps=T_value)
    ds = load_hf_dataset(dataset, max_examples=n)
    is_mcq = dataset != "gsm8k"
    correct = 0; total = 0
    t0 = time.perf_counter()
    for ex in ds:
        if is_mcq:
            p, lab = format_mcq_example(dataset, ex)
        else:
            p, lab = format_gsm8k_example(ex)
            lab = normalize_number_text(lab)
        out = generate_text_batch(model, tok, spec, [p], cfg, device="cuda")[0]
        pred = parse_choice_answer(out, num_choices=4) if is_mcq else normalize_number_text(out)
        if pred == lab:
            correct += 1
        total += 1
    return {"score": correct / max(total, 1), "n": total,
            "wall": time.perf_counter() - t0}


def run_one(dataset: str, n: int, T_value: int, _cached_state: dict) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"turboldm_4p1_T{T_value}"
    jsonl = OUT_DIR / "phase65_turboldm_4p1_steps.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 65 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 65 build {tag}/{dataset}")
    if _cached_state.get("model") is None:
        clear_cuda()
        model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
        try:
            stage_batches = prepare_stage_batches_for_model(model, tok, spec, QCFG, RunConfig())
            model = quantize_model_inplace(model, tok, spec, QCFG, RunConfig(), stage_batches=stage_batches)
            del stage_batches; gc.collect(); torch.cuda.empty_cache()
        except Exception as exc:
            log(f"  !! Phase 65 quant build failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 65, "tag": "build",
                                                      "err": str(exc), "trace": traceback.format_exc()})
            return
        _cached_state["model"] = model
        _cached_state["tok"] = tok
        _cached_state["bits"] = avg_bits_per_param(model)
        _cached_state["storage"] = model_parameter_storage_bytes(model) / 1024**3
        log(f"  bits={_cached_state['bits']:.2f} storage={_cached_state['storage']:.2f}GB")
    model, tok = _cached_state["model"], _cached_state["tok"]
    try:
        maybe_reset_cuda_peak()
        res = evaluate_at_T(model, tok, spec, dataset, n, T_value)
        row = {
            "model": spec.key, "baseline": tag, "dataset": dataset,
            "score_value": res["score"], "num_examples": res["n"],
            "wall_sec": res["wall"], "T": T_value,
            "method": "turboldm_4p1", "seed": 42,
            "avg_bits_per_param": _cached_state["bits"],
            "param_storage_gb": _cached_state["storage"],
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(jsonl, row)
        log(f"    -> score={res['score']:.3f} T={T_value} wall={res['wall']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 65 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 65, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})


def main() -> None:
    state: dict = {}
    try:
        for dataset, n, Ts in GRID:
            for T_value in Ts:
                run_one(dataset, n, T_value, state)
    finally:
        if state.get("model") is not None:
            try:
                del state["model"], state["tok"]
            except Exception:
                pass
            clear_cuda()
    log("Phase 65 done")


if __name__ == "__main__":
    main()
