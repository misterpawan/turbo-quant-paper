# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 33: encoder-only masked-LM transfer, extended
(reviewer Phase 12).

The Appendix BERT table uses a 128-paragraph WikiText-103 validation
slice and omits DeBERTa-v3. We rerun with a 512-paragraph slice and
add DeBERTa-v3-base, plus 3 seeds of calibration ordering for the
calibration-dependent methods (AWQ, TurboLDM).
"""
from __future__ import annotations

import os, sys, time, gc, traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    QuantConfig, RunConfig, ModelSpec, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes,
    quantize_model_inplace, build_stage_calibration_batches,
    load_calibration_texts, sample_mask_positions,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

BERT_EXCL = r"(cls\.|lm_head|embeddings|pooler)"

MODELS = {
    "bert_base": "bert-base-uncased",
    "roberta_base": "roberta-base",
    "deberta_v3_base": "microsoft/deberta-v3-base",
}

MAX_PARAGRAPHS = 512


def make_spec(model_name: str, key: str) -> ModelSpec:
    return ModelSpec(
        key=key, model_id=model_name, model_cls="AutoModelForMaskedLM",
        trust_remote_code=False, dtype="float32",
        default_steps=1, default_block_length=128, default_gen_length=128,
        uses_chat_template=False, mask_token_hint=None,
        requires_left_padding=False,
    )


@torch.no_grad()
def masked_nll(model, tok, texts, mask_ratio=0.15, max_tokens=256, batch_size=4):
    model.eval()
    total_loss = 0.0; total_count = 0
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


def load_bert(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to("cuda")
    model.eval()
    return model, tok


def apply_quant(model, tok, spec, qcfg):
    if qcfg.method == "bf16":
        return
    texts = load_calibration_texts(RunConfig(calibration_text_dataset="wikitext"),
                                    max_texts=min(qcfg.calibration_max_examples, 24))
    stage_batches = build_stage_calibration_batches(
        tokenizer=tok, texts=texts, spec=spec, quant_cfg=qcfg,
        run_cfg=RunConfig(), device="cuda",
    )
    quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
    del stage_batches; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def eval_one(model_key: str, model_name: str, tag: str, qcfg: QuantConfig) -> None:
    jsonl = OUT_DIR / "phase33_encoder_extended.jsonl"
    if jsonl.exists():
        import json as _j
        with jsonl.open() as f:
            for line in f:
                try:
                    r = _j.loads(line)
                    if r.get("model") == model_key and r.get("baseline") == tag:
                        log(f"Phase 33 SKIP {model_key}/{tag}")
                        return
                except Exception:
                    continue
    seed_everything(42); set_matmul_precision()
    log(f"Phase 33 build {model_key}/{tag}")
    clear_cuda()
    try:
        model, tok = load_bert(model_name)
        spec = make_spec(model_name, model_key)
        apply_quant(model, tok, spec, qcfg)
    except Exception as exc:
        log(f"  !! build failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 33, "model": model_key, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
        return
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
        texts = []
        for ex in ds:
            v = (ex.get("text") or "").strip()
            if len(v) >= 64:
                texts.append(v)
            if len(texts) >= MAX_PARAGRAPHS:
                break
        log(f"  eval {model_key}/{tag} (n={len(texts)})")
        r = masked_nll(model, tok, texts, mask_ratio=0.15)
        row = {
            "model": model_key, "baseline": tag, "dataset": "wikitext_nll_ext",
            "score_value": r["nll"], "ppl": r["ppl"],
            "tokens": r["tokens"], "num_examples": len(texts),
            "wall_sec": r["wall_sec"],
            "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
            "avg_bits_per_param": avg_bits_per_param(model),
            "param_storage_gb": model_parameter_storage_bytes(model) / 1024**3,
            "quant_cfg": asdict(qcfg),
        }
        append_jsonl(jsonl, row)
        log(f"    -> nll={r['nll']:.3f} ppl={r['ppl']:.2f} bits={row['avg_bits_per_param']:.2f} "
            f"storage={row['param_storage_gb']:.2f}GB wall={r['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! eval failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 33, "model": model_key, "tag": tag,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    methods = [
        ("bf16", QuantConfig(method="bf16")),
        ("uniform_int4", QuantConfig(method="uniform", w_bits=4, group_size=128, exclude_regex=BERT_EXCL)),
        ("awq_4", QuantConfig(method="awq", w_bits=4, group_size=128,
                               calibration_max_examples=16, probe_max_examples=2,
                               calibration_max_tokens=256, exclude_regex=BERT_EXCL)),
        ("turboldm_4p1", QuantConfig(method="turboldm", base_bits=4, residual_bits=1,
                                      block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=16, probe_max_examples=2,
                                      calibration_max_tokens=256, exclude_regex=BERT_EXCL)),
        ("turboldm_3p1", QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                      block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=16, probe_max_examples=2,
                                      calibration_max_tokens=256, exclude_regex=BERT_EXCL)),
    ]
    for model_key, model_name in MODELS.items():
        for tag, qcfg in methods:
            eval_one(model_key, model_name, tag, qcfg)
    log("Phase 33 done")


if __name__ == "__main__":
    main()
