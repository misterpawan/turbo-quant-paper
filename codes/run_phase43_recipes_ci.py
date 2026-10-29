# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 43: rerun mixed-bit recipes at n=500 with 3 seeds.

Reviewer round 4: the "attn@3 + MLP@4 reaches within noise of W4"
claim was based on a single-seed n=100 pilot whose numbers (39.3, 79.0,
59.5, 60.0) are below W4's n=500 numbers (61.8, 85.0, 66.6, 70.0). The
"within noise" claim does not survive on the same eval slice. This
phase reruns four recipes at n=500 with 3 seeds so we can either
recover the claim with CIs or honestly walk it back.

Recipes (all on LLaDA-8B-Instruct):
    (a) attn@3 + MLP@4
    (b) attn@4 + MLP@3
    (c) attn@3 + up&gate@3 + down@4
    (d) attn@2 + MLP@4
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict, replace

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    MODEL_SPECS, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    evaluate_dataset, prepare_stage_batches_for_model, quantize_model_inplace,
    load_model_and_tokenizer, iter_named_linears, get_parent_module,
    UniformLinear, uniform_group_quantize, replace_module,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"
DATASETS = [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]
SEEDS = [42, 43, 44]


def family_of(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "att_proj", "attn_out", "o_proj", "qkv_proj")):
        return "attn"
    if "ff_out" in n or "down_proj" in n:
        return "down"
    if "gate_proj" in n or n.endswith(".gate"):
        return "gate"
    if "up_proj" in n or "ff_proj" in n:
        return "up"
    if "mlp" in n:
        return "mlp"
    return "other"


def quantize_mixed(model: nn.Module, family_to_bits: dict, group_size: int = 128) -> nn.Module:
    """Apply per-family uniform quantisation at the bits given in family_to_bits."""
    layer_names = [n for n, _ in iter_named_linears(model, exclude_regex=EXCL)]
    for name in layer_names:
        fam = family_of(name)
        # Map "down/gate/up" to "mlp" if not separately specified.
        bits = family_to_bits.get(fam, family_to_bits.get("mlp", 4))
        parent, leaf = get_parent_module(model, name)
        linear = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        pack = uniform_group_quantize(linear.weight.detach(), bits=bits, group_size=group_size)
        qmod = UniformLinear(linear, pack, cache_dequantized=False)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
    return model


RECIPES = {
    "attn3_mlp4":            {"attn": 3, "mlp": 4, "down": 4, "gate": 4, "up": 4},
    "attn4_mlp3":            {"attn": 4, "mlp": 3, "down": 3, "gate": 3, "up": 3},
    "attn3_upgate3_down4":   {"attn": 3, "up": 3, "gate": 3, "down": 4},
    "attn2_mlp4":            {"attn": 2, "mlp": 4, "down": 4, "gate": 4, "up": 4},
}


def run_one(recipe: str, dataset: str, n_examples: int, seed: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{recipe}_s{seed}"
    jsonl = OUT_DIR / "phase43_recipes_ci.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if (r.get("model") == spec.key and r.get("baseline") == tag
                            and r.get("dataset") == dataset and r.get("seed") == seed):
                        log(f"Phase 43 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    seed_everything(seed); set_matmul_precision()
    log(f"Phase 43 build {tag}/{dataset} seed={seed}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        model = quantize_mixed(model, RECIPES[recipe])
        gc.collect(); torch.cuda.empty_cache()
        storage_gb = model_parameter_storage_bytes(model) / 1024**3
        bits = avg_bits_per_param(model)
        gen_cfg = mcq_fast_gen_cfg(spec, dataset)
        log(f"  eval {tag}/{dataset} n={n_examples} bits={bits:.2f} storage={storage_gb:.2f}GB")
        t0 = time.perf_counter()
        maybe_reset_cuda_peak()
        rec = evaluate_dataset(model=model, tokenizer=tok, spec=spec,
                               dataset_name=dataset, gen_cfg=gen_cfg,
                               max_examples=n_examples)
        rec.baseline = tag
        rec.param_storage_gb = storage_gb
        rec.avg_bits_per_param = bits
        row = rec.to_dict()
        row["wall_sec"] = time.perf_counter() - t0
        row["recipe"] = recipe
        row["seed"] = seed
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} bits={bits:.2f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 43 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 43, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for seed in SEEDS:
        for recipe in RECIPES:
            for dataset, n in DATASETS:
                run_one(recipe, dataset, n, seed)
    log("Phase 43 done")


if __name__ == "__main__":
    main()
