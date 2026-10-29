# Release portability changes: use the current user cache; create generated output directories where needed.
"""
Phase 17: Dream-7B-Instruct via its native ``diffusion_generate`` decoder.

Dream registers ``AutoModel`` (not ``AutoModelForCausalLM``) and exposes a
``diffusion_generate`` method that implements the bespoke masked-diffusion
sampler described in the Dream paper. The HF ``pipeline("text-generation")``
path does not work because the model is not a causal LM.

We load via ``AutoModel`` with ``trust_remote_code=True``, then reuse our
existing TurboLDM / baseline quantizers to replace the ``nn.Linear`` modules
in place. Our quantized linears expose the same ``.forward(x)`` API so
Dream's native ``diffusion_generate`` continues to work unchanged.

Datasets: MMLU, ARC-Challenge, HellaSwag (short letter), plus GSM8K
(final-number match). Slice sizes are smaller than the LLaDA sweep because
Dream's native decoder runs at higher step-count (steps=16 minimum works in
practice; we also run a shorter steps=8 variant for the quantized rows to
cap wall-clock).
"""
from __future__ import annotations

import os, sys, time, torch, gc, traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import TurboLDM_experiments_helper as H
from TurboLDM_experiments_helper import (
    ModelSpec, QuantConfig, RunConfig, seed_everything, set_matmul_precision,
    avg_bits_per_param, model_parameter_storage_bytes, maybe_reset_cuda_peak,
    quantize_model_inplace, prepare_stage_batches_for_model,
    load_hf_dataset, format_mcq_example, parse_choice_answer,
    normalize_number_text, format_gsm8k_example,
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


def load_dream():
    tok = AutoTokenizer.from_pretrained(DREAM_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        DREAM_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    return model, tok


@torch.no_grad()
def dream_generate_letter(model, tok, prompt: str, max_new_tokens: int, steps: int) -> str:
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    out = model.diffusion_generate(ids, max_new_tokens=max_new_tokens, steps=steps)
    seq = out if isinstance(out, torch.Tensor) else out.sequences
    gen = seq[0, ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True)


def evaluate_dream_mcq(model, tok, dataset_name: str, max_examples: int, steps: int) -> dict:
    ds = load_hf_dataset(dataset_name, max_examples=max_examples)
    prompts, labels = [], []
    for ex in ds:
        p, lab = format_mcq_example(dataset_name, ex)
        prompts.append(p); labels.append(lab)
    t0 = time.perf_counter()
    correct = 0
    preds = []
    for prompt, lab in zip(prompts, labels):
        try:
            txt = dream_generate_letter(model, tok, prompt, max_new_tokens=8, steps=steps)
        except Exception as exc:
            txt = f"<err:{exc}>"
        preds.append(txt)
        n = 4 if dataset_name != "piqa" else 2
        if parse_choice_answer(txt, num_choices=n) == lab:
            correct += 1
    return {
        "dataset": dataset_name,
        "score_value": correct / max(len(labels), 1),
        "num_examples": len(labels),
        "wall_sec": time.perf_counter() - t0,
        "preds": preds[:5],
        "labels": labels[:5],
    }


def evaluate_dream_gsm(model, tok, max_examples: int, steps: int = 64) -> dict:
    ds = load_hf_dataset("gsm8k", max_examples=max_examples)
    t0 = time.perf_counter()
    correct = 0
    preds, labels = [], []
    for ex in ds:
        prompt, ans = format_gsm8k_example(ex)
        try:
            txt = dream_generate_letter(model, tok, prompt, max_new_tokens=128, steps=steps)
        except Exception as exc:
            txt = f"<err:{exc}>"
        preds.append(txt[:80])
        labels.append(normalize_number_text(ans))
        if normalize_number_text(txt) == normalize_number_text(ans):
            correct += 1
    return {
        "dataset": "gsm8k",
        "score_value": correct / max(len(labels), 1),
        "num_examples": len(labels),
        "wall_sec": time.perf_counter() - t0,
        "preds": preds[:3], "labels": labels[:3],
    }


def main() -> None:
    jsonl = OUT_DIR / "phase17_dream.jsonl"
    seed_everything(42)
    set_matmul_precision()
    mcq_slices = {"mmlu": 100, "arc_challenge": 150, "hellaswag": 150}
    gsm_slice = 50

    methods = [
        ("bf16", QuantConfig(method="bf16")),
        ("awq_4", QuantConfig(method="awq", w_bits=4, group_size=128,
                               calibration_max_examples=32, probe_max_examples=4,
                               calibration_max_tokens=256, exclude_regex=DREAM_EXCL)),
        ("turboldm_4p1", QuantConfig(method="turboldm", base_bits=4, residual_bits=1,
                                      block_size=128, stage_count=4,
                                      sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=24, probe_max_examples=3,
                                      calibration_max_tokens=192, exclude_regex=DREAM_EXCL)),
        ("turboldm_3p1", QuantConfig(method="turboldm", base_bits=3, residual_bits=1,
                                      block_size=128, stage_count=4,
                                      sensitivity_metric="hidden_mse",
                                      residual_top_fraction=0.15,
                                      calibration_max_examples=24, probe_max_examples=3,
                                      calibration_max_tokens=192, exclude_regex=DREAM_EXCL)),
    ]
    for tag, qcfg in methods:
        # Resume support.
        if jsonl.exists():
            import json as _json
            done = set()
            with jsonl.open() as f:
                for line in f:
                    try:
                        r = _json.loads(line)
                        done.add((r["baseline"], r["dataset"]))
                    except Exception:
                        continue
            need = [(ds, n) for ds, n in [*mcq_slices.items(), ("gsm8k", gsm_slice)]
                    if (tag, ds) not in done]
            if not need:
                log(f"Phase 17 SKIP {tag} (all datasets done)")
                continue
        log(f"Phase 17 {tag} on Dream-7B-Instruct")
        try:
            clear_cuda()
            model, tok = load_dream()
            if qcfg.method != "bf16":
                stage_batches = prepare_stage_batches_for_model(model, tok, DREAM_SPEC, qcfg, RunConfig())
                model = quantize_model_inplace(model, tok, DREAM_SPEC, qcfg, RunConfig(), stage_batches=stage_batches)
                del stage_batches
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log(f"  quantized: bits={avg_bits_per_param(model):.2f} storage={model_parameter_storage_bytes(model)/1024**3:.2f}GB")
            for ds, n in mcq_slices.items():
                log(f"  eval {tag}/{ds} (n={n}, steps=16)")
                maybe_reset_cuda_peak()
                res = evaluate_dream_mcq(model, tok, ds, n, steps=16)
                row = {
                    "model": "dream_7b_instruct", "baseline": tag, "dataset": ds,
                    "score_value": res["score_value"], "num_examples": res["num_examples"],
                    "wall_sec": res["wall_sec"],
                    "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
                    "avg_bits_per_param": avg_bits_per_param(model),
                    "param_storage_gb": model_parameter_storage_bytes(model) / 1024**3,
                    "extra": {"preds": res["preds"], "labels": res["labels"]},
                    "quant_cfg": asdict(qcfg),
                }
                append_jsonl(jsonl, row)
                log(f"    -> score={res['score_value']:.3f} peak={row['peak_mem_gb']:.1f}GB wall={res['wall_sec']:.1f}s")
            # GSM8K
            log(f"  eval {tag}/gsm8k (n={gsm_slice}, steps=64)")
            maybe_reset_cuda_peak()
            res = evaluate_dream_gsm(model, tok, gsm_slice, steps=64)
            row = {
                "model": "dream_7b_instruct", "baseline": tag, "dataset": "gsm8k",
                "score_value": res["score_value"], "num_examples": res["num_examples"],
                "wall_sec": res["wall_sec"],
                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else 0.0,
                "avg_bits_per_param": avg_bits_per_param(model),
                "param_storage_gb": model_parameter_storage_bytes(model) / 1024**3,
                "extra": {"preds": res["preds"], "labels": res["labels"]},
                "quant_cfg": asdict(qcfg),
            }
            append_jsonl(jsonl, row)
            log(f"    -> score={res['score_value']:.3f} wall={res['wall_sec']:.1f}s")
        except Exception as exc:
            log(f"  !! {tag} failed: {exc}")
            append_jsonl(OUT_DIR / "failures.jsonl", {"phase": 17, "tag": tag,
                                                      "err": str(exc), "trace": traceback.format_exc()})
        finally:
            try:
                del model, tok
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log("Phase 17 done")


if __name__ == "__main__":
    main()
