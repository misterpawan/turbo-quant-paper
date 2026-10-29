# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 67: Dream-7B throughput counterpart of Phase 54.

Same protocol as Phase 54 (BF16, NF4, TurboLDM-(3+1) at three
gen lengths) but on Dream-7B with its native diffusion_generate
decoder. Confirms (or refutes) the NF4 VRAM-win claim on a second
model family.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import bitsandbytes as bnb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    ModelSpec, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    iter_named_linears, get_parent_module,
    prepare_stage_batches_for_model, quantize_model_inplace,
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
PROMPT = ("Explain the difference between supervised and unsupervised machine learning "
          "in a few sentences.")
GEN_SETTINGS = [("gen64", 64, 32), ("gen128", 128, 64), ("gen256", 256, 128)]

METHODS = {
    "bf16": None,
    "nf4": "nf4",
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


def run_method(method: str) -> None:
    tag = f"{method}_s42"
    jsonl = OUT_DIR / "phase67_dream_throughput.jsonl"
    done = set()
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag:
                        done.add(r.get("dataset"))
                except Exception:
                    continue
    if all(cfg_tag in done for (cfg_tag, _, _) in GEN_SETTINGS):
        log(f"Phase 67 SKIP {tag} (all gen-cfgs done)")
        return
    seed_everything(42); set_matmul_precision()
    log(f"Phase 67 build {tag}")
    clear_cuda()
    try:
        model, tok = load_dream()
        config = METHODS[method]
        if method == "nf4":
            n_swapped = replace_with_nf4(model)
            log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
            gc.collect(); torch.cuda.empty_cache()
            bits, storage = 4.0, sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        elif config is not None:
            stage_batches = prepare_stage_batches_for_model(model, tok, DREAM_SPEC, config, RunConfig())
            model = quantize_model_inplace(model, tok, DREAM_SPEC, config, RunConfig(), stage_batches=stage_batches)
            del stage_batches; gc.collect(); torch.cuda.empty_cache()
            bits = avg_bits_per_param(model)
            storage = model_parameter_storage_bytes(model) / 1024**3
        else:
            bits = 16.0
            storage = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

        for cfg_tag, gen_len, T in GEN_SETTINGS:
            if cfg_tag in done:
                log(f"  Phase 67 SKIP {tag}/{cfg_tag}"); continue
            try:
                _ = dream_generate(model, tok, PROMPT, max_new_tokens=gen_len, steps=T)
            except Exception as exc:
                log(f"  !! warm-up failed {tag}/{cfg_tag}: {exc}")
                continue
            torch.cuda.synchronize()
            maybe_reset_cuda_peak()
            t0 = time.perf_counter()
            try:
                _ = dream_generate(model, tok, PROMPT, max_new_tokens=gen_len, steps=T)
                _ = dream_generate(model, tok, PROMPT, max_new_tokens=gen_len, steps=T)
            except Exception as exc:
                log(f"  !! timed pass failed {tag}/{cfg_tag}: {exc}")
                continue
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tokens = 2 * gen_len
            row = {
                "model": "dream_7b_instruct", "baseline": tag, "dataset": cfg_tag,
                "method": method, "seed": 42,
                "avg_bits_per_param": bits, "param_storage_gb": storage,
                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
                "wall_sec": elapsed, "tokens": tokens,
                "tokens_per_sec": tokens / elapsed,
                "gen_length": gen_len, "T": T,
            }
            append_jsonl(jsonl, row)
            log(f"    {tag}/{cfg_tag}: {tokens/elapsed:.1f} tok/s peak {row['peak_mem_gb']:.1f} GB")
    except Exception as exc:
        log(f"  !! Phase 67 {tag} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 67, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()


def main() -> None:
    for method in METHODS:
        run_method(method)
    log("Phase 67 done")


if __name__ == "__main__":
    main()
