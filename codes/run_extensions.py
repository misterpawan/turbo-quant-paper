"""
Phase 8–14 extensions for the NeurIPS paper.

Phase 8  — 3-bit scalar baselines (AWQ-3, uniform-3) on LLaDA-8B.
Phase 9  — Mixed-family recipes driven by the Phase 1 sensitivity map.
Phase 10 — Per-depth sensitivity: quantize a block-range and leave the rest BF16.
Phase 11 — Step-count x quantization interaction.
Phase 12 — Additional MCQ datasets that do NOT require deprecated loader scripts.
Phase 13 — Three-seed variance measurement on top methods.
Phase 14 — Throughput / tokens/sec wall-clock benchmark.

Each phase writes to its own JSONL in runs/overnight/ and supports resume
via the done-keys set.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import TurboLDM_experiments_helper as H  # noqa: E402
from TurboLDM_experiments_helper import (  # noqa: E402
    GenerateConfig,
    MODEL_SPECS,
    QuantConfig,
    RunConfig,
    DATASET_SPECS,
    avg_bits_per_param,
    evaluate_dataset,
    iter_named_linears,
    load_model_and_tokenizer,
    maybe_reset_cuda_peak,
    model_parameter_storage_bytes,
    prepare_stage_batches_for_model,
    quantize_model_inplace,
    seed_everything,
    set_matmul_precision,
    apply_turboldm,
    apply_uniform_quantization,
    apply_awq_fallback,
    replace_module,
    get_parent_module,
    turboquant_vector_quantize,
    uniform_group_quantize,
    TurboLDMLinear,
    UniformLinear,
    StageContext,
    ratio_to_stage,
    compute_stage_layer_scores,
    build_turboldm_schedule,
    generic_masked_diffusion_generate,
    infer_mask_token_id,
    build_chat_prompt,
)

# Reuse the logger + helpers from the main driver.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_overnight import (  # noqa: E402
    OUT_DIR,
    MASTER_LOG,
    SLICE_SIZES,
    SMALL_SLICE_SIZES,
    append_jsonl,
    build_method_model,
    clear_cuda,
    load_done_keys,
    log,
    mcq_fast_gen_cfg,
    run_and_log,
)


# ---------- Extra dataset loaders (parquet only, no scripts) ----------
# We add these three because they are script-free on the datasets v4.7 loader
# and they broaden the MCQ evaluation beyond the original four.
EXTRA_DATASETS = {
    "winogrande": {
        "path": "allenai/winogrande", "name": "winogrande_debiased",
        "split": "validation", "kind": "mcq", "trust_remote_code": True,
    },
    "openbookqa": {
        "path": "allenai/openbookqa", "name": "main",
        "split": "test", "kind": "mcq", "trust_remote_code": False,
    },
    "boolq": {
        "path": "google/boolq", "name": None,
        "split": "validation", "kind": "mcq", "trust_remote_code": False,
    },
}
for k, v in EXTRA_DATASETS.items():
    if k not in DATASET_SPECS:
        DATASET_SPECS[k] = v

# Minimal format / parse helpers for the new datasets. We monkey-patch
# format_mcq_example in the helper by swapping the format function based on
# dataset name.


def _format_winogrande(ex):
    prompt = (
        "Fill the blank in the sentence with the better option. Respond with only 'A' or 'B'.\n\n"
        f"Sentence: {ex['sentence']}\n"
        f"A. {ex['option1']}\nB. {ex['option2']}\n\nAnswer:"
    )
    answer = {"1": "A", "2": "B"}[str(ex["answer"])]
    return prompt, answer


def _format_openbookqa(ex):
    labels = ex["choices"]["label"]
    texts = ex["choices"]["text"]
    opts = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    prompt = (
        "Answer the science question. Respond with only the option letter.\n\n"
        f"Question: {ex['question_stem']}\n"
        f"Options:\n{opts}\n\nAnswer:"
    )
    return prompt, ex["answerKey"].strip()


def _format_boolq(ex):
    prompt = (
        "Answer the yes/no question based on the passage. Respond with only 'A' for yes or 'B' for no.\n\n"
        f"Passage: {ex['passage']}\n"
        f"Question: {ex['question']}\n\nAnswer:"
    )
    return prompt, "A" if ex["answer"] else "B"


_EXTRA_FORMATTERS = {
    "winogrande": _format_winogrande,
    "openbookqa": _format_openbookqa,
    "boolq": _format_boolq,
}

# Patch the helper's format_mcq_example to dispatch on the extra datasets too.
_orig_format_mcq = H.format_mcq_example


def _patched_format_mcq(dataset_name, ex):
    if dataset_name in _EXTRA_FORMATTERS:
        return _EXTRA_FORMATTERS[dataset_name](ex)
    return _orig_format_mcq(dataset_name, ex)


H.format_mcq_example = _patched_format_mcq

# Also patch evaluate_mcq_dataset to accept the new datasets by passing
# num_choices correctly.
_orig_parse_choice = H.parse_choice_answer


def _num_choices_for(ds):
    return {
        "mmlu": 4, "arc_challenge": 4, "hellaswag": 4, "piqa": 2,
        "winogrande": 2, "openbookqa": 4, "boolq": 2,
    }.get(ds, 4)


_orig_evaluate_mcq = H.evaluate_mcq_dataset


def _patched_evaluate_mcq(model, tokenizer, spec, dataset_name, gen_cfg,
                          max_examples=None, device="cuda"):
    # Fall through to the original but with an override of num_choices inside
    # the scoring loop is hard because it's inlined; instead re-implement just
    # for the extras.
    if dataset_name not in EXTRA_DATASETS:
        return _orig_evaluate_mcq(model, tokenizer, spec, dataset_name, gen_cfg,
                                   max_examples=max_examples, device=device)
    from TurboLDM_experiments_helper import (
        load_hf_dataset, format_mcq_example, maybe_reset_cuda_peak, batched,
        generate_text_batch, current_peak_mem_gb, ResultRecord, DATASET_SPECS,
        parse_choice_answer, build_chat_prompt,
    )
    ds = load_hf_dataset(dataset_name, max_examples=max_examples)
    prompts, labels = [], []
    for ex in ds:
        p, lab = format_mcq_example(dataset_name, ex)
        prompts.append(p)
        labels.append(lab)
    maybe_reset_cuda_peak()
    t0 = time.perf_counter()
    preds: List[str] = []
    for batch in batched(prompts, gen_cfg.batch_size):
        preds.extend(generate_text_batch(model, tokenizer, spec, list(batch), gen_cfg, device=device))
    elapsed = time.perf_counter() - t0
    n = _num_choices_for(dataset_name)
    correct = sum(1 for p, g in zip(preds, labels) if parse_choice_answer(p, num_choices=n) == g)
    score = correct / max(len(labels), 1)
    return ResultRecord(
        model=spec.key, baseline="", dataset=dataset_name,
        split=DATASET_SPECS[dataset_name]["split"],
        num_examples=len(labels), score_name="accuracy",
        score_value=score, elapsed_sec=elapsed,
        peak_mem_gb=current_peak_mem_gb(),
        param_storage_gb=0.0, avg_bits_per_param=0.0,
        notes="Generative exact-match MCQ evaluation (extra dataset)",
        extra={"predictions": preds[:5], "labels": labels[:5]},
    )


H.evaluate_mcq_dataset = _patched_evaluate_mcq


# Make evaluate_dataset aware of the extras.
_orig_evaluate_dataset = H.evaluate_dataset


def _patched_evaluate_dataset(model, tokenizer, spec, dataset_name, gen_cfg,
                               max_examples, device="cuda"):
    if dataset_name in EXTRA_DATASETS:
        return _patched_evaluate_mcq(model, tokenizer, spec, dataset_name,
                                     gen_cfg, max_examples=max_examples, device=device)
    return _orig_evaluate_dataset(model, tokenizer, spec, dataset_name, gen_cfg,
                                   max_examples, device=device)


H.evaluate_dataset = _patched_evaluate_dataset

# Also patch the already-imported names in run_overnight's namespace so the
# driver there picks up our patched versions.
import run_overnight as _ro
_ro.evaluate_dataset = _patched_evaluate_dataset
_ro.evaluate_mcq_dataset = _patched_evaluate_mcq

# Extend slice sizes for the extra datasets.
SLICE_SIZES.update({"winogrande": 200, "openbookqa": 200, "boolq": 200})
SMALL_SLICE_SIZES.update({"winogrande": 150, "openbookqa": 150, "boolq": 150})

# Teach mcq_fast_gen_cfg about the new MCQ datasets so they use
# short gen_length / low step counts like the other MCQs.
_orig_mcq_fast = _ro.mcq_fast_gen_cfg


def _patched_mcq_fast_gen_cfg(spec, dataset):
    if dataset in EXTRA_DATASETS:
        from TurboLDM_experiments_helper import GenerateConfig, default_generate_config_for_model
        cfg = default_generate_config_for_model(spec, "mmlu")  # same as MCQ defaults
        cfg.steps = 16
        cfg.gen_length = 8
        cfg.block_length = 8
        cfg.batch_size = 2
        cfg.stop_strings = tuple()
        return cfg
    return _orig_mcq_fast(spec, dataset)


_ro.mcq_fast_gen_cfg = _patched_mcq_fast_gen_cfg
mcq_fast_gen_cfg = _patched_mcq_fast_gen_cfg  # local name too


# -----------------------------------------------------------------
# Shared exclude regex, consistent with run_overnight.
EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


# ============================================================
# Phase 8 — 3-bit scalar baselines
# ============================================================
def phase8_low_bit_baselines(spec_key: str = "llada_8b_instruct") -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase8_low_bit.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 8: 3-bit scalar baselines ===")
    methods = [
        ("uniform_int3", QuantConfig(method="uniform", w_bits=3, group_size=128,
                                      exclude_regex=EXCL)),
        ("awq_3", QuantConfig(method="awq", w_bits=3, group_size=128,
                               calibration_max_examples=32, probe_max_examples=4,
                               calibration_max_tokens=256, exclude_regex=EXCL)),
        ("gptq_3", QuantConfig(method="gptq", w_bits=3, group_size=128,
                                calibration_max_examples=32, probe_max_examples=4,
                                calibration_max_tokens=256, exclude_regex=EXCL)),
        ("uniform_int2", QuantConfig(method="uniform", w_bits=2, group_size=128,
                                      exclude_regex=EXCL)),
    ]
    datasets = ["mmlu", "arc_challenge", "hellaswag", "gsm8k"]
    for tag, qcfg in methods:
        if all((spec.key, tag, ds, "") in done for ds in datasets):
            log(f"SKIP phase8 {tag}")
            continue
        try:
            model, tok = build_method_model(spec, qcfg)
        except Exception as exc:
            log(f"  !! build failed {tag}: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 8, "tag": tag, "err": str(exc),
                                                      "trace": traceback.format_exc()})
            continue
        run_and_log(model, tok, spec, tag, qcfg, datasets, jsonl, SLICE_SIZES, done)
        del model, tok; clear_cuda()


# ============================================================
# Phase 9 — Mixed-family quantization recipes.
# We quantize different families at different bit-widths in a SINGLE model
# and evaluate the cross-family trade-off. This is the paper's "modular
# deployment recipe" contribution.
# ============================================================
def _apply_mixed_family(model, spec, recipe: Dict[str, QuantConfig]) -> None:
    """Apply different quantization configs to different name-regex buckets.

    Layers are bucketed by the first matching regex in `recipe`; remaining
    layers stay BF16.
    """
    # Build stage batches once (shared across recipes that need turbo).
    any_turbo = any(c.method == "turboldm" for c in recipe.values())
    if any_turbo:
        # Use the most permissive include set to fit all experts.
        probe_cfg = next(iter(recipe.values()))
        stage_batches = prepare_stage_batches_for_model(model, model._cached_tok, spec,
                                                        probe_cfg, RunConfig())
    else:
        stage_batches = None
    for fam_name, qcfg in recipe.items():
        q = copy.deepcopy(qcfg)
        if q.method == "uniform":
            # Apply uniform to the bucket only.
            _apply_uniform_only(model, q)
        elif q.method == "turboldm":
            _apply_turboldm_only(model, stage_batches, q)
        else:
            raise ValueError(f"mixed recipe only supports uniform/turboldm, got {q.method}")
    del stage_batches
    clear_cuda()


def _apply_uniform_only(model, qcfg: QuantConfig) -> None:
    names = [n for n, _ in iter_named_linears(model, include_regex=qcfg.include_regex,
                                              exclude_regex=qcfg.exclude_regex)]
    for name in names:
        parent, leaf = get_parent_module(model, name)
        linear = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        pack = uniform_group_quantize(linear.weight.detach(), bits=qcfg.w_bits, group_size=qcfg.group_size)
        qmod = UniformLinear(linear, pack)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
        del pack, linear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _apply_turboldm_only(model, stage_batches, qcfg: QuantConfig) -> None:
    if qcfg.compute_stage_scores:
        scores = compute_stage_layer_scores(model, stage_batches, qcfg)
    else:
        rows = [{"layer": n, "stage": s, "score": float(s + 1)}
                for n, _ in iter_named_linears(model, include_regex=qcfg.include_regex, exclude_regex=qcfg.exclude_regex)
                for s in range(qcfg.stage_count)]
        scores = pd.DataFrame(rows)
    layer_to_stages, entropy_thresholds = build_turboldm_schedule(scores, qcfg)
    model._turboldm_stage_ratios = tuple(qcfg.stage_ratios)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    names = [n for n, _ in iter_named_linears(model, include_regex=qcfg.include_regex,
                                               exclude_regex=qcfg.exclude_regex)]
    for name in names:
        parent, leaf = get_parent_module(model, name)
        linear = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        base_pack = turboquant_vector_quantize(linear.weight.detach(), bits=qcfg.base_bits,
                                                block_size=qcfg.block_size, rotation=qcfg.rotation)
        residual_pack = None
        if name in layer_to_stages:
            residual = linear.weight.detach() - base_pack["deq"]
            residual_pack = turboquant_vector_quantize(residual, bits=max(qcfg.residual_bits, 1),
                                                        block_size=qcfg.block_size, rotation=qcfg.rotation)
            del residual
        qmod = TurboLDMLinear(linear, base_pack=base_pack, residual_pack=residual_pack,
                               active_stages=layer_to_stages.get(name, []),
                               entropy_thresholds=entropy_thresholds.get(name))
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
        del base_pack, residual_pack, linear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def phase9_mixed_recipes(spec_key: str = "llada_8b_instruct") -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase9_mixed.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 9: mixed-family recipes ===")
    ATTN_INCL = r"(q_proj|k_proj|v_proj|attn_out)$"
    MLP_INCL = r"(ff_proj|up_proj|ff_out)$"
    MLP_DOWN_INCL = r"ff_out$"
    MLP_UPGATE_INCL = r"(ff_proj|up_proj)$"

    recipes = {
        "mix_attn3_mlp4": [
            QuantConfig(method="uniform", w_bits=3, group_size=128, include_regex=ATTN_INCL, exclude_regex=EXCL),
            QuantConfig(method="uniform", w_bits=4, group_size=128, include_regex=MLP_INCL, exclude_regex=EXCL),
        ],
        "mix_attn4_mlp3": [
            QuantConfig(method="uniform", w_bits=4, group_size=128, include_regex=ATTN_INCL, exclude_regex=EXCL),
            QuantConfig(method="uniform", w_bits=3, group_size=128, include_regex=MLP_INCL, exclude_regex=EXCL),
        ],
        "mix_attn3_mlpup3_down4": [
            QuantConfig(method="uniform", w_bits=3, group_size=128, include_regex=ATTN_INCL, exclude_regex=EXCL),
            QuantConfig(method="uniform", w_bits=3, group_size=128, include_regex=MLP_UPGATE_INCL, exclude_regex=EXCL),
            QuantConfig(method="uniform", w_bits=4, group_size=128, include_regex=MLP_DOWN_INCL, exclude_regex=EXCL),
        ],
        "mix_attn2_mlp4": [
            QuantConfig(method="uniform", w_bits=2, group_size=128, include_regex=ATTN_INCL, exclude_regex=EXCL),
            QuantConfig(method="uniform", w_bits=4, group_size=128, include_regex=MLP_INCL, exclude_regex=EXCL),
        ],
    }
    datasets = ["mmlu", "arc_challenge", "hellaswag", "gsm8k"]
    for tag, steps in recipes.items():
        if all((spec.key, tag, ds, "") in done for ds in datasets):
            log(f"SKIP phase9 {tag}")
            continue
        try:
            clear_cuda()
            model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
            model._cached_tok = tok  # used by _apply_mixed_family if turboldm recipes added later
            for sub_cfg in steps:
                _apply_uniform_only(model, sub_cfg)
            log(f"  built {tag} — bits={avg_bits_per_param(model):.2f}, storage={model_parameter_storage_bytes(model)/1024**3:.2f} GB")
        except Exception as exc:
            log(f"  !! build failed {tag}: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 9, "tag": tag, "err": str(exc),
                                                      "trace": traceback.format_exc()})
            continue
        merged = QuantConfig(method="mixed")
        run_and_log(model, tok, spec, tag, merged, datasets, jsonl, SLICE_SIZES, done)
        del model, tok; clear_cuda()


# ============================================================
# Phase 10 — Per-depth sensitivity
# ============================================================
def phase10_depth(spec_key: str = "llada_8b_instruct") -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase10_depth.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 10: per-depth sensitivity ===")
    # LLaDA-8B has 32 blocks numbered 0..31.
    depth_ranges = [(0, 7), (8, 15), (16, 23), (24, 31)]
    datasets = ["mmlu", "gsm8k"]
    for lo, hi in depth_ranges:
        tag = f"depth_b{lo:02d}_{hi:02d}"
        if all((spec.key, tag, ds, "") in done for ds in datasets):
            log(f"SKIP phase10 {tag}")
            continue
        # Use a regex that captures blocks in [lo, hi].
        block_regex = r"blocks\.(" + "|".join(str(i) for i in range(lo, hi + 1)) + r")\."
        include = block_regex + r".*(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)$"
        qcfg = QuantConfig(method="uniform", w_bits=3, group_size=128,
                           include_regex=include, exclude_regex=EXCL)
        try:
            model, tok = build_method_model(spec, qcfg)
            log(f"  built {tag} bits={avg_bits_per_param(model):.2f} storage={model_parameter_storage_bytes(model)/1024**3:.2f} GB")
        except Exception as exc:
            log(f"  !! build failed {tag}: {exc}")
            continue
        run_and_log(model, tok, spec, tag, qcfg, datasets, jsonl, SMALL_SLICE_SIZES, done)
        del model, tok; clear_cuda()


# ============================================================
# Phase 11 — Step-count interaction
# ============================================================
def phase11_steps(spec_key: str = "llada_8b_instruct") -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase11_steps.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 11: step-count x quantization ===")
    method_configs = {
        "bf16": QuantConfig(method="bf16"),
        "awq_4": QuantConfig(method="awq", w_bits=4, group_size=128,
                              calibration_max_examples=32, probe_max_examples=4,
                              calibration_max_tokens=256, exclude_regex=EXCL),
        "turboldm_3p1": QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                     block_size=128, stage_count=4,
                                     sensitivity_metric="hidden_mse",
                                     residual_top_fraction=0.15,
                                     calibration_max_examples=24, probe_max_examples=3,
                                     calibration_max_tokens=192, exclude_regex=EXCL),
    }
    step_choices = [16, 32, 64]
    datasets = ["mmlu", "gsm8k"]
    for meth_tag, qcfg in method_configs.items():
        built = None
        for steps in step_choices:
            tag = f"{meth_tag}_steps{steps}"
            if all((spec.key, tag, ds, f"steps={steps}") in done for ds in datasets):
                log(f"SKIP phase11 {tag}")
                continue
            if built is None:
                try:
                    built = build_method_model(spec, qcfg)
                except Exception as exc:
                    log(f"  !! build failed {meth_tag}: {exc}")
                    break
            model, tok = built
            for ds in datasets:
                gen_cfg = mcq_fast_gen_cfg(spec, ds)
                # Override steps; adjust block_length to keep divisibility.
                gen_cfg.steps = steps
                if ds == "gsm8k":
                    gen_cfg.gen_length = 128
                    gen_cfg.block_length = 32
                    if steps % 4 != 0:
                        steps = (steps // 4) * 4 or 4
                        gen_cfg.steps = steps
                run_cfg = RunConfig()
                key = (spec.key, tag, ds, f"steps={steps}")
                if key in done:
                    continue
                log(f"  eval {meth_tag}/{ds} steps={steps}")
                t0 = time.perf_counter()
                maybe_reset_cuda_peak()
                try:
                    rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                                            dataset_name=ds, gen_cfg=gen_cfg,
                                            max_examples=SMALL_SLICE_SIZES.get(ds))
                    rec.baseline = tag
                    rec.param_storage_gb = model_parameter_storage_bytes(model) / 1024**3
                    rec.avg_bits_per_param = avg_bits_per_param(model)
                    row = rec.to_dict()
                    row["extra_tag"] = f"steps={steps}"
                    row["wall_sec"] = time.perf_counter() - t0
                    row["quant_cfg"] = asdict(qcfg)
                    row["gen_cfg"] = asdict(gen_cfg)
                    append_jsonl(jsonl, row)
                    log(f"    -> score={rec.score_value:.3f} wall={row['wall_sec']:.1f}s")
                except Exception as exc:
                    log(f"    !! failed: {exc}")
                    append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 11, "tag": tag,
                                                               "err": str(exc)})
        if built is not None:
            model, tok = built
            del model, tok
        clear_cuda()


# ============================================================
# Phase 12 — Additional MCQ datasets
# ============================================================
def phase12_extra_datasets(spec_key: str = "llada_8b_instruct") -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase12_extra.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 12: extra MCQ datasets ===")
    methods = {
        "bf16": QuantConfig(method="bf16"),
        "awq_4": QuantConfig(method="awq", w_bits=4, group_size=128,
                              calibration_max_examples=32, probe_max_examples=4,
                              calibration_max_tokens=256, exclude_regex=EXCL),
        "turboldm_4p1": QuantConfig(method="turboldm", base_bits=4, residual_bits=1,
                                     block_size=128, stage_count=4,
                                     sensitivity_metric="hidden_mse",
                                     residual_top_fraction=0.15,
                                     calibration_max_examples=24, probe_max_examples=3,
                                     calibration_max_tokens=192, exclude_regex=EXCL),
        "turboldm_3p1": QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                     block_size=128, stage_count=4,
                                     sensitivity_metric="hidden_mse",
                                     residual_top_fraction=0.15,
                                     calibration_max_examples=24, probe_max_examples=3,
                                     calibration_max_tokens=192, exclude_regex=EXCL),
    }
    datasets = ["winogrande", "openbookqa", "boolq"]
    for tag, qcfg in methods.items():
        if all((spec.key, tag, ds, "") in done for ds in datasets):
            log(f"SKIP phase12 {tag}")
            continue
        try:
            model, tok = build_method_model(spec, qcfg)
        except Exception as exc:
            log(f"  !! build failed {tag}: {exc}")
            continue
        run_and_log(model, tok, spec, tag, qcfg, datasets, jsonl, SLICE_SIZES, done)
        del model, tok; clear_cuda()


# ============================================================
# Phase 13 — Multiple seeds for statistical significance
# ============================================================
def phase13_seeds(spec_key: str = "llada_8b_instruct") -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase13_seeds.jsonl"
    done = load_done_keys(jsonl)
    log("=== Phase 13: seed variance ===")
    methods = {
        "bf16": QuantConfig(method="bf16"),
        "awq_4": QuantConfig(method="awq", w_bits=4, group_size=128,
                              calibration_max_examples=32, probe_max_examples=4,
                              calibration_max_tokens=256, exclude_regex=EXCL),
        "turboldm_3p1": QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                     block_size=128, stage_count=4,
                                     sensitivity_metric="hidden_mse",
                                     residual_top_fraction=0.15,
                                     calibration_max_examples=24, probe_max_examples=3,
                                     calibration_max_tokens=192, exclude_regex=EXCL),
    }
    seeds = [11, 22, 33]
    datasets = ["mmlu", "gsm8k"]
    for seed in seeds:
        seed_everything(seed)
        for tag, qcfg in methods.items():
            key_prefix = (spec.key, tag)
            if all((spec.key, tag, ds, f"seed={seed}") in done for ds in datasets):
                log(f"SKIP phase13 {tag} seed={seed}")
                continue
            try:
                model, tok = build_method_model(spec, qcfg)
            except Exception as exc:
                log(f"  !! build failed {tag} seed={seed}: {exc}")
                continue
            for ds in datasets:
                if (spec.key, tag, ds, f"seed={seed}") in done:
                    continue
                gen_cfg = mcq_fast_gen_cfg(spec, ds)
                log(f"  eval {tag}/{ds} seed={seed}")
                t0 = time.perf_counter()
                maybe_reset_cuda_peak()
                try:
                    rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                                            dataset_name=ds, gen_cfg=gen_cfg,
                                            max_examples=SMALL_SLICE_SIZES.get(ds))
                    rec.baseline = tag
                    rec.param_storage_gb = model_parameter_storage_bytes(model) / 1024**3
                    rec.avg_bits_per_param = avg_bits_per_param(model)
                    row = rec.to_dict()
                    row["extra_tag"] = f"seed={seed}"
                    row["wall_sec"] = time.perf_counter() - t0
                    row["quant_cfg"] = asdict(qcfg)
                    row["gen_cfg"] = asdict(gen_cfg)
                    append_jsonl(jsonl, row)
                    log(f"    -> score={rec.score_value:.3f}")
                except Exception as exc:
                    log(f"    !! failed: {exc}")
            del model, tok; clear_cuda()


# ============================================================
# Phase 14 — Throughput benchmark (tokens / sec)
# ============================================================
def phase14_throughput(spec_key: str = "llada_8b_instruct") -> None:
    spec = MODEL_SPECS[spec_key]
    jsonl = OUT_DIR / "phase14_throughput.jsonl"
    done = set()
    log("=== Phase 14: throughput ===")
    methods = {
        "bf16": QuantConfig(method="bf16"),
        "uniform_int4": QuantConfig(method="uniform", w_bits=4, group_size=128, exclude_regex=EXCL),
        "awq_4": QuantConfig(method="awq", w_bits=4, group_size=128,
                              calibration_max_examples=32, probe_max_examples=4,
                              calibration_max_tokens=256, exclude_regex=EXCL),
        "turboldm_4p1": QuantConfig(method="turboldm", base_bits=4, residual_bits=1,
                                     block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                     residual_top_fraction=0.15,
                                     calibration_max_examples=24, probe_max_examples=3,
                                     calibration_max_tokens=192, exclude_regex=EXCL),
        "turboldm_3p1": QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                     block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                     residual_top_fraction=0.15,
                                     calibration_max_examples=24, probe_max_examples=3,
                                     calibration_max_tokens=192, exclude_regex=EXCL),
    }
    PROMPT = ("Explain the difference between supervised and unsupervised machine learning "
              "in a few sentences.")
    cfgs = {
        "gen64": GenerateConfig(steps=32, gen_length=64, block_length=32, batch_size=1, temperature=0.0),
        "gen128": GenerateConfig(steps=64, gen_length=128, block_length=32, batch_size=1, temperature=0.0),
    }
    for tag, qcfg in methods.items():
        try:
            model, tok = build_method_model(spec, qcfg)
        except Exception as exc:
            log(f"  !! build failed {tag}: {exc}")
            continue
        for cfg_tag, gen_cfg in cfgs.items():
            # Warm-up once.
            try:
                _ = H.generate_text_batch(model, tok, spec, [PROMPT], gen_cfg, device="cuda")
            except Exception as exc:
                log(f"  !! warm-up failed {tag}/{cfg_tag}: {exc}")
                continue
            # Timed pass.
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                out = H.generate_text_batch(model, tok, spec, [PROMPT] * 2, gen_cfg, device="cuda")
            except Exception as exc:
                log(f"  !! timed pass failed {tag}/{cfg_tag}: {exc}")
                continue
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tokens = 2 * gen_cfg.gen_length
            row = {
                "model": spec.key, "baseline": tag, "dataset": cfg_tag,
                "avg_bits_per_param": avg_bits_per_param(model),
                "param_storage_gb": model_parameter_storage_bytes(model) / 1024**3,
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0,
                "wall_sec": elapsed,
                "tokens": tokens,
                "tokens_per_sec": tokens / elapsed,
                "gen_cfg": asdict(gen_cfg),
            }
            append_jsonl(jsonl, row)
            log(f"  {tag}/{cfg_tag}: {tokens/elapsed:.1f} tok/s peak {row['peak_mem_gb']:.1f} GB")
        del model, tok; clear_cuda()


# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", type=str, default="8,9,10,11,12,13,14")
    args = parser.parse_args()
    phases = {int(p) for p in args.phases.split(",") if p}
    seed_everything(42)
    set_matmul_precision()
    log(f"extension driver start, phases={sorted(phases)}")
    dispatch = {
        8: phase8_low_bit_baselines,
        9: phase9_mixed_recipes,
        10: phase10_depth,
        11: phase11_steps,
        12: phase12_extra_datasets,
        13: phase13_seeds,
        14: phase14_throughput,
    }
    for p in sorted(phases):
        fn = dispatch.get(p)
        if fn is None:
            continue
        try:
            fn()
        except Exception as exc:
            log(f"!! phase {p} crashed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl",
                         {"phase": p, "err": str(exc), "trace": traceback.format_exc()})
    log("extension driver done")


if __name__ == "__main__":
    main()
