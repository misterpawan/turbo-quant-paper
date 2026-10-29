# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 21: Storage audit.

Per-parameter accounting of every quantization method applied to
LLaDA-8B-Instruct. Reviewer concern: "8.02B params at 3 bits is 3.0 GB,
but the paper reports 2.87 GB. Either the paper is wrong, or it is
hiding the BF16 embeddings/head/norms in the 'quantized' total."

This phase produces:
- runs/overnight/storage_audit.csv (one row per (method, category))
- papers/gen_table_storage_audit.tex (NeurIPS-ready table)
- papers/gen_table_storage_summary.tex (compression ratios at the
  TOTAL-MODEL level, not just the quantized linears)
"""
from __future__ import annotations

import os, sys, csv, json, math, gc, torch
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    QuantConfig, RunConfig, MODEL_SPECS, seed_everything, set_matmul_precision,
    load_model_and_tokenizer, prepare_stage_batches_for_model,
    quantize_model_inplace, iter_named_linears, model_parameter_storage_bytes,
    avg_bits_per_param, QuantizedLinearBase,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda

import torch.nn as nn

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


def categorize(name: str, module: nn.Module) -> str:
    """Coarse category for accounting.

    Note: LLaDA / OLMo uses a fused QKV projection named ``att_proj``; keep the
    attention family *before* the MLP family and match ``att_proj`` explicitly.
    The LM head is ``transformer.ff_out`` (top-level), while each decoder block
    has its own ``...blocks.N.ff_out`` — distinguish by the full suffix path.
    """
    n = name.lower()
    # LM head first so we do not route the top-level ff_out into "mlp".
    if (n.endswith("transformer.ff_out") or n.endswith("transformer.ff_out.weight")
            or "lm_head" in n):
        return "lm_head"
    if "embed" in n or "wte" in n or "embeddings" in n:
        return "embed"
    if "norm" in n or "layernorm" in n or "rms_norm" in n or "ln_f" in n:
        return "norm"
    if ("q_proj" in n or "k_proj" in n or "v_proj" in n or "o_proj" in n
            or "att_proj" in n or "attn_out" in n or "qkv_proj" in n):
        return "attn"
    if ("ff_proj" in n or "up_proj" in n or "gate_proj" in n or "down_proj" in n
            or "ff_out" in n or "mlp" in n or "fc1" in n or "fc2" in n):
        return "mlp"
    return "other"


def measure_bf16(model: nn.Module) -> dict:
    """Bytes per category for the unquantized BF16 reference."""
    cats = {"attn": 0, "mlp": 0, "embed": 0, "lm_head": 0, "norm": 0, "other": 0}
    for name, p in model.named_parameters():
        cat = categorize(name, model)
        cats[cat] = cats.get(cat, 0) + p.numel() * p.element_size()
    return cats


def measure_quantized(model: nn.Module) -> dict:
    """Bytes per category accounting both QuantizedLinearBase modules
    and any leftover BF16 parameters (embeddings, head, norms, biases)."""
    cats = {"attn": 0, "mlp": 0, "embed": 0, "lm_head": 0, "norm": 0, "other": 0,
            "quant_indices": 0, "quant_scales": 0, "quant_signs": 0,
            "quant_codebook": 0, "quant_residual": 0, "quant_other": 0,
            "bias_bf16": 0}
    seen_modules = set()
    # Quantized linears
    for name, mod in model.named_modules():
        if isinstance(mod, QuantizedLinearBase):
            seen_modules.add(name)
            cat = categorize(name, mod)
            for k, v in mod.quant_metadata.items():
                if not torch.is_tensor(v):
                    continue
                if k in ("deq", "res_deq"):
                    continue
                # Use the same accounting that quant_storage_bytes uses.
                if k in ("indices", "q"):
                    # ``q`` is the uniform-quant int-code tensor stored in int8
                    # but logically at ``bits`` bits/entry. ``indices`` is the
                    # TurboLDM Gaussian-codebook index tensor (same accounting).
                    bits = int(mod.quant_metadata.get("bits", torch.tensor([8])).item())
                    bytes_ = (v.numel() * bits + 7) // 8
                    cats["quant_indices"] += bytes_; cats[cat] += bytes_
                elif k in ("res_indices", "res_q"):
                    bits = int(mod.quant_metadata.get("res_bits", torch.tensor([1])).item())
                    bytes_ = (v.numel() * bits + 7) // 8
                    cats["quant_residual"] += bytes_; cats[cat] += bytes_
                elif k in ("rms", "scales", "input_scale"):
                    bytes_ = v.numel() * v.element_size()
                    cats["quant_scales"] += bytes_; cats[cat] += bytes_
                elif k in ("signs", "res_signs"):
                    bytes_ = (v.numel() + 7) // 8
                    cats["quant_signs"] += bytes_; cats[cat] += bytes_
                elif k in ("sign_bits", "qjl_sign_bits", "res_qjl_sign_bits"):
                    bytes_ = (v.numel() + 7) // 8
                    cats["quant_signs"] += bytes_; cats[cat] += bytes_
                elif k == "sparse_residual":
                    nnz = int((v != 0).sum().item())
                    bytes_ = nnz * v.element_size() + nnz * 8
                    cats["quant_residual"] += bytes_; cats[cat] += bytes_
                elif k in ("pad", "bits", "shape", "res_bits"):
                    bytes_ = v.numel() * v.element_size()
                    cats["quant_other"] += bytes_
                else:
                    bytes_ = v.numel() * v.element_size()
                    cats["quant_other"] += bytes_; cats[cat] += bytes_
            if mod.bias is not None:
                bytes_ = mod.bias.numel() * mod.bias.element_size()
                cats["bias_bf16"] += bytes_; cats[cat] += bytes_
    # Leftover BF16 params (anything not covered by a QuantizedLinearBase).
    for name, p in model.named_parameters():
        # Skip params that belong to any QuantizedLinearBase (we accounted for them above).
        owned = False
        for sm in seen_modules:
            if name.startswith(sm + "."):
                owned = True; break
        if owned:
            continue
        cat = categorize(name, model)
        cats[cat] += p.numel() * p.element_size()
    return cats


def build_method(spec, qcfg: QuantConfig):
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    if qcfg.method == "bf16":
        return model, tok
    stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
    model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
    del stage_batches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tok


def main() -> None:
    seed_everything(42); set_matmul_precision()
    spec = MODEL_SPECS["llada_8b_instruct"]
    methods = [
        ("bf16", QuantConfig(method="bf16")),
        ("uniform_int4", QuantConfig(method="uniform", w_bits=4, group_size=128, exclude_regex=EXCL)),
        ("awq_4", QuantConfig(method="awq", w_bits=4, group_size=128,
                               calibration_max_examples=24, probe_max_examples=3,
                               calibration_max_tokens=192, exclude_regex=EXCL)),
        ("turboldm_4p1", QuantConfig(method="turboldm", base_bits=4, residual_bits=1, block_size=128,
                                      stage_count=4, sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=24, probe_max_examples=3,
                                      calibration_max_tokens=192, exclude_regex=EXCL)),
        ("turboldm_3p1", QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128,
                                      stage_count=4, sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=24, probe_max_examples=3,
                                      calibration_max_tokens=192, exclude_regex=EXCL)),
    ]
    rows = []
    summary = []
    for tag, qcfg in methods:
        log(f"Phase 21 audit {tag}")
        try:
            model, tok = build_method(spec, qcfg)
        except Exception as exc:
            log(f"  build failed: {exc}")
            continue
        if qcfg.method == "bf16":
            cats = measure_bf16(model)
        else:
            cats = measure_quantized(model)
        # Total params for the full model (always counted from BF16-equivalent shapes).
        n_params = sum(p.numel() for p in model.parameters())
        for cat, bytes_ in cats.items():
            rows.append({"method": tag, "category": cat, "bytes": bytes_,
                         "MiB": bytes_ / 1024**2, "GiB": bytes_ / 1024**3})
        total = sum(bytes_ for k, bytes_ in cats.items()
                    if k in ("attn", "mlp", "embed", "lm_head", "norm", "other"))
        # BF16 residual = bytes that are still stored in BF16 *after* quantization.
        # This is just the bytes of the leftover nn.Parameter objects (embeddings,
        # LM head, norms, biases) — exactly the second loop of measure_quantized,
        # which we separately tally in ``bias_bf16`` plus the category slots that
        # were populated *only* by that second loop (embed/lm_head/norm). The
        # per-family cats (attn/mlp) are entirely quantized bytes for all of our
        # methods (we never leave attn/mlp linears in BF16).
        if qcfg.method == "bf16":
            bf16_only = total
            quant_part = 0
        else:
            bf16_only = (cats.get("embed", 0) + cats.get("lm_head", 0)
                         + cats.get("norm", 0) + cats.get("bias_bf16", 0))
            quant_part = cats.get("attn", 0) + cats.get("mlp", 0) + cats.get("other", 0)
        summary.append({
            "method": tag,
            "n_params": n_params,
            "total_GiB": total / 1024**3,
            "bf16_residual_GiB": bf16_only / 1024**3,
            "quantized_part_GiB": (quant_part if qcfg.method != "bf16" else 0) / 1024**3,
            "avg_bits_per_quantized_param":
                (8 * (quant_part) / max(1, sum(p.numel() for n, m in model.named_modules() if isinstance(m, QuantizedLinearBase) for p in m.parameters()) +
                                          sum(int(m.quant_metadata.get("shape", torch.tensor([0,0]))[0].item()) *
                                              int(m.quant_metadata.get("shape", torch.tensor([0,0]))[1].item())
                                              for n, m in model.named_modules() if isinstance(m, QuantizedLinearBase))))
                if qcfg.method != "bf16" else 16.0,
        })
        log(f"  {tag}: total {total/1024**3:.2f} GiB | bf16-residual {bf16_only/1024**3:.2f} GiB")
        del model, tok; clear_cuda()
    # Write CSV
    csv_path = OUT_DIR / "storage_audit.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "category", "bytes", "MiB", "GiB"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    log(f"wrote {csv_path}")
    sum_path = OUT_DIR / "storage_audit_summary.csv"
    with sum_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for r in summary:
            writer.writerow(r)
    log(f"wrote {sum_path}")
    # Render LaTeX
    lines = [
        "\\begin{table}[t]\\centering\\small\n",
        "\\caption{Storage audit on LLaDA-8B-Instruct. \\emph{Total} is the full model "
        "footprint (quantized linears + BF16 embeddings/head/norms). \\emph{BF16 residual} "
        "is the unquantized portion (embeddings + LM head + layer norms + biases). "
        "Compression ratios are reported \\emph{against the full BF16 model}, not just "
        "the quantized linears, so they reflect what a deployed checkpoint actually weighs.}\n",
        "\\label{tab:storage}\n",
        "\\begin{tabular}{lrrrr}\n\\toprule\n",
        "Method & Total GiB & BF16 residual GiB & Quantized part GiB & Compression vs BF16 \\\\\n\\midrule\n",
    ]
    bf16_total = next((r["total_GiB"] for r in summary if r["method"] == "bf16"), None)
    pretty = {"bf16": "BF16 baseline",
              "uniform_int4": "W4 uniform",
              "awq_4": "AWQ-4",
              "turboldm_4p1": "TurboLDM (4+1)",
              "turboldm_3p1": "TurboLDM (3+1)"}
    for r in summary:
        comp = bf16_total / r["total_GiB"] if bf16_total else 1.0
        lines.append(f"{pretty.get(r['method'], r['method'])} & "
                     f"{r['total_GiB']:.2f} & {r['bf16_residual_GiB']:.2f} & "
                     f"{r['quantized_part_GiB']:.2f} & {comp:.2f}$\\times$ \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out = "".join(lines)
    Path("papers/neurips2026").mkdir(parents=True, exist_ok=True)
    Path("papers/gen_table_storage.tex").write_text(out)
    Path("papers/neurips2026/gen_table_storage.tex").write_text(out)
    log("wrote gen_table_storage.tex")


if __name__ == "__main__":
    main()
