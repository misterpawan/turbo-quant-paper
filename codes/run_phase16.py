# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 16: MDLM-OWT (170M, Kuleshov group) masked-token NLL study.

MDLM-OWT is a small BERT-style masked diffusion LM pre-trained on
OpenWebText. It uses a custom HF class, so we load the model via
``AutoModelForMaskedLM`` with ``trust_remote_code`` and use a GPT-2
tokenizer augmented with a ``<mask>`` token (vocab_size 50258 in the
checkpoint = GPT-2 50257 plus one mask id).

The natural evaluation for a small masked-diffusion pre-trained LM is
*denoiser quality* at a fixed mask ratio. We report masked-token NLL
(equivalently, conditional cross-entropy at masking ratio 0.15) on a
held-out WikiText-103 validation slice. Lower is better. This is the
same metric MDLM itself reports during training.

We compare: BF16, uniform int4, TurboLDM (4+1), TurboLDM (3+1), and a
module-family sensitivity study (attention vs MLP) on this second dense
model to validate the transfer of our Phase 1 findings on LLaDA-8B.
"""
from __future__ import annotations

import os, sys, time, torch, gc, traceback, copy
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes,
    iter_named_linears, turboquant_vector_quantize, uniform_group_quantize,
    apply_turboldm, apply_uniform_quantization, apply_awq_fallback,
    replace_module, get_parent_module, TurboLDMLinear, UniformLinear,
    prepare_stage_batches_for_model, quantize_model_inplace,
    load_calibration_texts, build_stage_calibration_batches,
    ModelSpec, sample_mask_positions, model_forward_logits,
    StageContext, ratio_to_stage,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda


# ---- MDLM-specific loader ----
from transformers import AutoModelForMaskedLM, GPT2TokenizerFast

MDLM_MODEL_ID = "kuleshov-group/mdlm-owt"
MDLM_MASK_TOKEN_ID = 50257  # id appended after GPT-2 vocab


def load_mdlm() -> tuple:
    """Load MDLM-OWT with a GPT-2 tokenizer augmented by a ``<mask>`` token."""
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Register the mask token at id 50257 (matches the checkpoint's vocab_size of 50258).
    if "<mask>" not in tok.get_vocab():
        tok.add_tokens(["<mask>"], special_tokens=True)
    tok.mask_token = "<mask>"
    tok.mask_token_id = MDLM_MASK_TOKEN_ID
    tok.padding_side = "right"  # MDLM is bidirectional
    model = AutoModelForMaskedLM.from_pretrained(
        MDLM_MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32,
    )
    model = model.to("cuda")
    model.eval()
    return model, tok


def mdlm_spec_stub() -> ModelSpec:
    return ModelSpec(
        key="mdlm_owt",
        model_id=MDLM_MODEL_ID,
        model_cls="AutoModelForMaskedLM",
        trust_remote_code=True,
        dtype="float32",
        default_steps=128,
        default_block_length=128,
        default_gen_length=128,
        uses_chat_template=False,
        mask_token_hint=MDLM_MASK_TOKEN_ID,
        requires_left_padding=False,
        native_generate=False,
        notes="kuleshov-group/mdlm-owt, 170M, OpenWebText pre-training, BERT-style mask diffusion.",
    )


# ---- Masked-token NLL evaluator ----
@torch.no_grad()
def masked_nll(model, tok, texts: List[str], mask_ratio: float = 0.15,
                max_tokens: int = 256, batch_size: int = 4, device: str = "cuda") -> dict:
    model.eval()
    spec = mdlm_spec_stub()
    total_loss = 0.0
    total_count = 0
    t0 = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tok(list(batch_texts), padding="longest", truncation=True,
                  max_length=max_tokens, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
        masked, mask_idx = sample_mask_positions(
            input_ids, attn, ratio=mask_ratio, mask_token_id=MDLM_MASK_TOKEN_ID,
            pad_token_id=tok.pad_token_id, preserve_prefix=0,
        )
        # MDLM needs a `timesteps` kwarg (sigma for the forward process); at a
        # fixed mask ratio 0.15 we can pass sigma=0 (loss reduces to
        # standard masked-LM cross-entropy at that ratio).
        sigma = torch.zeros(masked.shape[0], device=masked.device)
        try:
            out = model(input_ids=masked, timesteps=sigma)
            logits = out.logits if hasattr(out, "logits") and out.logits is not None else out
        except TypeError:
            logits = model_forward_logits(model, masked, attention_mask=attn)
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        true_ids = input_ids[mask_idx]
        selected = log_probs[mask_idx].gather(-1, true_ids.view(-1, 1)).squeeze(-1)
        total_loss -= float(selected.sum().item())
        total_count += int(mask_idx.sum().item())
    elapsed = time.perf_counter() - t0
    nll = total_loss / max(total_count, 1)
    return {"nll": nll, "ppl": float(torch.tensor(nll).exp().item()),
            "tokens": total_count, "wall_sec": elapsed}


def evaluate_mdlm(model, tok, tag: str, quant_cfg: QuantConfig, extra_tag: str = "") -> dict:
    # Load 128 WikiText-103 texts (same calibration corpus style).
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    texts = []
    for ex in ds:
        v = ex.get("text", "").strip()
        if len(v) >= 64:
            texts.append(v)
        if len(texts) >= 128:
            break
    log(f"  eval MDLM-OWT / {tag} (n={len(texts)} texts, mask_ratio=0.15)")
    res = masked_nll(model, tok, texts, mask_ratio=0.15, max_tokens=256, batch_size=4)
    storage = model_parameter_storage_bytes(model) / 1024**3
    bits = avg_bits_per_param(model)
    row = {
        "model": "mdlm_owt", "baseline": tag, "dataset": "wikitext_nll",
        "split": "validation", "num_examples": len(texts),
        "score_name": "masked_nll", "score_value": res["nll"],
        "ppl": res["ppl"],
        "tokens": res["tokens"], "wall_sec": res["wall_sec"],
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
        "param_storage_gb": storage, "avg_bits_per_param": bits,
        "extra_tag": extra_tag, "quant_cfg": asdict(quant_cfg),
    }
    log(f"    -> nll={res['nll']:.3f} ppl={res['ppl']:.2f} bits={bits:.2f} storage={storage:.2f}GB wall={res['wall_sec']:.1f}s")
    return row


def apply_quant_to_mdlm(model, tok, qcfg: QuantConfig) -> None:
    """Reuse our quantize_model_inplace on MDLM; mdlm has its own prepare path."""
    if qcfg.method in ("bf16", "baseline", "none"):
        return
    spec = mdlm_spec_stub()
    # Build stage batches from WikiText-103 directly.
    texts = load_calibration_texts(RunConfig(calibration_text_dataset="wikitext"),
                                    max_texts=min(qcfg.calibration_max_examples, 24))
    stage_batches = build_stage_calibration_batches(
        tokenizer=tok, texts=texts, spec=spec, quant_cfg=qcfg,
        run_cfg=RunConfig(), device="cuda",
    )
    quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
    del stage_batches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


MDLM_MODULE_FAMILIES = {
    # MDLM-OWT uses a packed QKV linear (`attn_qkv`) and an `attn_out`;
    # the MLP is an `nn.Sequential` so its two linears show up as `mlp.0`
    # and `mlp.2`.
    "attn_qkv":  r"attn_qkv$",
    "attn_out":  r"attn_out$",
    "mlp_up":    r"mlp\.0$",
    "mlp_down":  r"mlp\.2$",
}


def phase16_main() -> None:
    seed_everything(42)
    set_matmul_precision()
    jsonl = OUT_DIR / "phase16_mdlm.jsonl"

    # bf16 baseline
    log("Phase 16: MDLM-OWT baseline")
    model, tok = load_mdlm()
    append_jsonl(jsonl, evaluate_mdlm(model, tok, "bf16", QuantConfig(method="bf16")))

    # Dump module name leaf histogram once to guide family regexes.
    import re as _re
    from collections import Counter
    leafs = Counter(n.split(".")[-1] for n, _ in iter_named_linears(model))
    log(f"MDLM linear leafs: {dict(leafs.most_common())}")
    del model, tok; clear_cuda()

    # Main baseline sweep.
    for tag, qcfg in [
        ("uniform_int4", QuantConfig(method="uniform", w_bits=4, group_size=128,
                                      exclude_regex=r"(vocab_embed|output\.)")),
        ("awq_4", QuantConfig(method="awq", w_bits=4, group_size=128,
                               calibration_max_examples=16, probe_max_examples=2,
                               calibration_max_tokens=256,
                               exclude_regex=r"(vocab_embed|output\.)")),
        ("turboldm_4p1", QuantConfig(method="turboldm", base_bits=4, residual_bits=1,
                                      block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=16, probe_max_examples=2,
                                      calibration_max_tokens=256,
                                      exclude_regex=r"(vocab_embed|output\.)")),
        ("turboldm_3p1", QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                      block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=16, probe_max_examples=2,
                                      calibration_max_tokens=256,
                                      exclude_regex=r"(vocab_embed|output\.)")),
    ]:
        try:
            model, tok = load_mdlm()
            apply_quant_to_mdlm(model, tok, qcfg)
            append_jsonl(jsonl, evaluate_mdlm(model, tok, tag, qcfg))
            del model, tok; clear_cuda()
        except Exception as exc:
            log(f"  !! Phase 16 {tag} failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 16, "tag": tag,
                                                      "err": str(exc), "trace": traceback.format_exc()})
            clear_cuda()

    # Module-family sensitivity on MDLM.
    for fam, regex in MDLM_MODULE_FAMILIES.items():
        tag = f"mdlm_only_{fam}"
        qcfg = QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                            block_size=128, stage_count=4, sensitivity_metric="hidden_mse",
                            residual_top_fraction=0.0,
                            include_regex=regex, exclude_regex=r"(vocab_embed|output\.)",
                            calibration_max_examples=16, probe_max_examples=2,
                            calibration_max_tokens=256)
        try:
            model, tok = load_mdlm()
            apply_quant_to_mdlm(model, tok, qcfg)
            append_jsonl(jsonl, evaluate_mdlm(model, tok, tag, qcfg))
            del model, tok; clear_cuda()
        except Exception as exc:
            log(f"  !! Phase 16 {tag} failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 16, "tag": tag,
                                                      "err": str(exc), "trace": traceback.format_exc()})
            clear_cuda()

    log("Phase 16 done")


if __name__ == "__main__":
    phase16_main()
