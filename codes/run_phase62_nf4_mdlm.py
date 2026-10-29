# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 62: NF4 on MDLM-OWT (best-effort).

MDLM-OWT (170M, kuleshov-group) is the small dense masked-diffusion
LM in the paper. Phase 16 covered it at BF16 / uniform-int4 /
TurboLDM-(4+1, 3+1). This phase adds an NF4 row by converting the
checkpoint to BF16 (bnb.Linear4bit's compute_dtype requirement) and
swapping every matched linear. If MDLM's custom forward path doesn't
accept the bnb tensor layouts, the script logs a failure row and
exits cleanly; the rest of chain_round9 is unaffected.
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
    seed_everything, set_matmul_precision,
    iter_named_linears, get_parent_module,
    sample_mask_positions, model_forward_logits,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

from transformers import AutoModelForMaskedLM, GPT2TokenizerFast
from datasets import load_dataset

MDLM_MODEL_ID = "kuleshov-group/mdlm-owt"
MDLM_MASK_TOKEN_ID = 50257
MDLM_EXCL = r"(lm_head|embed|embedding|cls\.|pooler)"


def load_mdlm():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if "<mask>" not in tok.get_vocab():
        tok.add_tokens(["<mask>"], special_tokens=True)
    tok.mask_token = "<mask>"
    tok.mask_token_id = MDLM_MASK_TOKEN_ID
    tok.padding_side = "right"
    # Load in bf16 because bnb.Linear4bit needs compute_dtype=bf16.
    model = AutoModelForMaskedLM.from_pretrained(
        MDLM_MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda")
    model.eval()
    return model, tok


def replace_with_nf4(model: nn.Module) -> int:
    n = 0
    layer_names = [name for name, _ in iter_named_linears(model, exclude_regex=MDLM_EXCL)]
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
def masked_nll(model, tok, texts, mask_ratio=0.15, max_tokens=256, batch_size=4):
    total_loss, total_count = 0.0, 0
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(list(batch), padding="longest", truncation=True,
                  max_length=max_tokens, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to("cuda")
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
        masked, mask_idx = sample_mask_positions(
            input_ids, attn, ratio=mask_ratio, mask_token_id=MDLM_MASK_TOKEN_ID,
            pad_token_id=tok.pad_token_id, preserve_prefix=0,
        )
        sigma = torch.zeros(masked.shape[0], device=masked.device)
        try:
            out = model(input_ids=masked, timesteps=sigma)
            logits = out.logits if hasattr(out, "logits") and out.logits is not None else out
        except TypeError:
            logits = model_forward_logits(model, masked, attention_mask=attn)
        lp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        true_ids = input_ids[mask_idx]
        selected = lp[mask_idx].gather(-1, true_ids.view(-1, 1)).squeeze(-1)
        total_loss -= float(selected.sum().item())
        total_count += int(mask_idx.sum().item())
    nll = total_loss / max(total_count, 1)
    return {"nll": nll, "ppl": float(torch.tensor(nll).exp().item()),
            "tokens": total_count, "wall_sec": time.perf_counter() - t0}


def main() -> None:
    tag = "nf4_s42"
    jsonl = OUT_DIR / "phase62_nf4_mdlm.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag:
                        log(f"Phase 62 SKIP {tag}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 62 build mdlm_owt/{tag}")
    clear_cuda()
    try:
        model, tok = load_mdlm()
        n_swapped = replace_with_nf4(model)
        log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
        gc.collect(); torch.cuda.empty_cache()
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        texts = []
        for ex in ds:
            v = (ex.get("text") or "").strip()
            if len(v) >= 64:
                texts.append(v)
            if len(texts) >= 128:
                break
        r = masked_nll(model, tok, texts, mask_ratio=0.15)
        row = {
            "model": "mdlm_owt", "baseline": tag, "dataset": "wikitext_nll",
            "score_value": r["nll"], "ppl": r["ppl"],
            "tokens": r["tokens"], "num_examples": len(texts),
            "wall_sec": r["wall_sec"],
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
            "avg_bits_per_param": 4.0,
            "param_storage_gb": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3,
            "method": "nf4", "seed": 42,
        }
        append_jsonl(jsonl, row)
        log(f"    -> nll={r['nll']:.3f} ppl={r['ppl']:.2f} wall={r['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 62 {tag} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 62, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()
    log("Phase 62 done")


if __name__ == "__main__":
    main()
