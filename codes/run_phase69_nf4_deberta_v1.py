# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 69: NF4 on DeBERTa-v1 (encoder-family completion).

Phase 56 added NF4 rows for BERT-base and RoBERTa-base. DeBERTa-v3
omits the MLM head and is unevaluable; DeBERTa-v1 (microsoft/deberta-base)
ships a usable head per Phase 40. This phase adds the NF4 row for
DeBERTa-v1 so the encoder-family panel covers all three architectures.
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
    sample_mask_positions,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import load_dataset

DEBERTA_V1_ID = "microsoft/deberta-base"
ENC_EXCL = r"(cls\.|lm_head|embeddings|pooler)"


def load_deberta():
    tok = AutoTokenizer.from_pretrained(DEBERTA_V1_ID)
    model = AutoModelForMaskedLM.from_pretrained(DEBERTA_V1_ID, torch_dtype=torch.bfloat16)
    model = model.to("cuda"); model.eval()
    return model, tok


def replace_with_nf4(model: nn.Module) -> int:
    n = 0
    layer_names = [name for name, _ in iter_named_linears(model, exclude_regex=ENC_EXCL)]
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
    mask_id = tok.mask_token_id
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(list(batch), padding="longest", truncation=True,
                  max_length=max_tokens, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to("cuda")
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to("cuda")
        masked, mask_idx = sample_mask_positions(
            input_ids, attn, ratio=mask_ratio, mask_token_id=mask_id,
            pad_token_id=tok.pad_token_id, preserve_prefix=0,
        )
        out = model(input_ids=masked, attention_mask=attn)
        logits = out.logits if hasattr(out, "logits") else out[0]
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
    jsonl = OUT_DIR / "phase69_nf4_deberta_v1.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("baseline") == tag:
                        log(f"Phase 69 SKIP {tag}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 69 build deberta_v1_base/{tag}")
    clear_cuda()
    try:
        model, tok = load_deberta()
        n_swapped = replace_with_nf4(model)
        log(f"  swapped {n_swapped} linears -> bnb.Linear4bit (NF4)")
        gc.collect(); torch.cuda.empty_cache()
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        texts = []
        for ex in ds:
            v = (ex.get("text") or "").strip()
            if len(v) >= 64:
                texts.append(v)
            if len(texts) >= 300:
                break
        r = masked_nll(model, tok, texts, mask_ratio=0.15)
        row = {
            "model": "deberta_v1_base", "baseline": tag, "dataset": "wikitext_nll",
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
        log(f"  !! Phase 69 {tag} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 69, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    finally:
        try:
            del model, tok
        except Exception:
            pass
        clear_cuda()
    log("Phase 69 done")


if __name__ == "__main__":
    main()
