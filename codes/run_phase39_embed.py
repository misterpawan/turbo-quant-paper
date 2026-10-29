# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 39: 4-bit groupwise QuantizedEmbedding for the input embedding table.

LLaDA-8B-Instruct's input embedding is shape ``[126464, 4096]``
(vocab x hidden) and weighs $1.04$\,GiB at BF16 -- about $50\%$ of the
$1.93$\,GiB BF16 residual we leave behind in the default recipe.
This phase replaces that single ``nn.Embedding`` module with a
4-bit groupwise quantised version that stores int4 codes plus FP16
scales (one per row of $128$ values), and dequantises only the
gathered rows on each forward pass.

Storage saving:
  vocab x hidden / 8 bytes per int4 weight  +  vocab x hidden/group * 2 bytes scales
  = 126464 x 4096 / 2 + 126464 x 4096 / 128 * 2
  = 259 MiB + 8 MiB = 267 MiB
vs.\ 1.04 GiB BF16 -> roughly $4 \times$ saving, $\sim 780$\,MiB freed.

We measure the accuracy delta vs.\ the default TurboLDM-(3+1) recipe
(which keeps the embedding BF16) on MMLU/ARC-C/HellaSwag/\textsc{gsm8k}
at $n=500$ single seed.
"""
from __future__ import annotations

import os, sys, time, gc, traceback, json
from dataclasses import asdict
from pathlib import Path

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
    load_model_and_tokenizer, uniform_group_quantize,
)
from run_overnight import OUT_DIR, log, append_jsonl, clear_cuda, mcq_fast_gen_cfg

EXCL = r"(lm_head|embed|embedding|router|gate\.weight$|transformer\.ff_out$)"


class QuantizedEmbedding(nn.Module):
    """4-bit groupwise quantised replacement for ``nn.Embedding``.

    Stores int8-packed codes + per-(row, group) FP16 scales. Forward:
    gather the code rows for the input ids, dequantise them on the fly,
    return a (batch, seq, hidden) BF16 tensor matching the original
    embedding's dtype.
    """
    def __init__(self, weight: torch.Tensor, bits: int = 4, group_size: int = 128):
        super().__init__()
        device = weight.device
        dtype = weight.dtype
        pack = uniform_group_quantize(weight.detach(), bits=bits, group_size=group_size)
        self.register_buffer("q_codes", pack["q"].to(device))
        scales = pack["scales"]
        if scales.dim() == 2:
            scales = scales.unsqueeze(-1)
        self.register_buffer("scales", scales.to(device, dtype=torch.float32))
        self.vocab = int(pack["shape"][0].item())
        self.hidden = int(pack["shape"][1].item())
        self.bits = bits
        self.group_size = group_size
        self.target_dtype = dtype
        self.padding_idx = None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (...,) int64 with values in [0, vocab).
        flat_ids = input_ids.reshape(-1).to(torch.long)
        # Gather quantised rows; dequantise only the gathered rows.
        codes = self.q_codes.index_select(0, flat_ids).to(torch.float32)  # (B, n_groups, group_size)
        s = self.scales.index_select(0, flat_ids)
        deq = (codes * s).reshape(flat_ids.shape[0], -1)[:, :self.hidden]
        out = deq.to(self.target_dtype)
        return out.reshape(*input_ids.shape, self.hidden)


def replace_embedding(model: nn.Module) -> int:
    """Find the first nn.Embedding with vocab > 1000 and swap it. Returns 1 / 0."""
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Embedding) and mod.num_embeddings > 1000:
            qmod = QuantizedEmbedding(mod.weight.data).to(mod.weight.device, dtype=mod.weight.dtype)
            qmod.padding_idx = mod.padding_idx
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
            setattr(parent, parts[-1], qmod)
            log(f"  swapped {name} ({mod.num_embeddings} x {mod.embedding_dim}) -> QuantizedEmbedding")
            return 1
    return 0


def run_one(method: str, dataset: str, n_examples: int) -> None:
    spec = MODEL_SPECS["llada_8b_instruct"]
    tag = f"{method}_quantembed"
    jsonl = OUT_DIR / "phase39_embed.jsonl"
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("model") == spec.key and r.get("baseline") == tag and r.get("dataset") == dataset:
                        log(f"Phase 39 SKIP {tag}/{dataset}")
                        return
                except Exception:
                    continue
    qcfg = QuantConfig(
        method="turboldm", base_bits=3, residual_bits=1, block_size=128,
        stage_count=4, sensitivity_metric="hidden_mse", residual_top_fraction=0.15,
        calibration_max_examples=24, probe_max_examples=3,
        calibration_max_tokens=192, exclude_regex=EXCL,
    )
    seed_everything(42); set_matmul_precision()
    log(f"Phase 39 build {tag}/{dataset}")
    clear_cuda()
    model, tok = load_model_and_tokenizer(spec, device="cuda", dtype="bfloat16")
    try:
        stage_batches = prepare_stage_batches_for_model(model, tok, spec, qcfg, RunConfig())
        model = quantize_model_inplace(model, tok, spec, qcfg, RunConfig(), stage_batches=stage_batches)
        del stage_batches; gc.collect(); torch.cuda.empty_cache()
        n = replace_embedding(model)
        if n == 0:
            log("  no embedding swapped (no nn.Embedding found above 1000 vocab)")
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
        row["quant_cfg"] = asdict(qcfg)
        row["gen_cfg"] = asdict(gen_cfg)
        append_jsonl(jsonl, row)
        log(f"    -> score={rec.score_value:.3f} bits={bits:.2f} wall={row['wall_sec']:.1f}s")
    except Exception as exc:
        log(f"  !! Phase 39 {tag}/{dataset} failed: {exc}")
        append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 39, "tag": tag, "dataset": dataset,
                                                  "err": str(exc), "trace": traceback.format_exc()})
    del model, tok; clear_cuda()


def main() -> None:
    for dataset, n in [("mmlu", 500), ("arc_challenge", 500), ("hellaswag", 500), ("gsm8k", 500)]:
        run_one("turboldm_3p1", dataset, n)
    log("Phase 39 done")


if __name__ == "__main__":
    main()
