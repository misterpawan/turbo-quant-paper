
from __future__ import annotations

import copy
import gc
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import types
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover
    load_dataset = None  # type: ignore

try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    AutoConfig = None  # type: ignore
    AutoModel = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore


# ---------------------------------------------------------------------------
# Registry objects
# ---------------------------------------------------------------------------

OFFICIAL_REPOS: Dict[str, str] = {
    "mdlm": "https://github.com/kuleshov-group/mdlm.git",
    "llada": "https://github.com/ML-GSAI/LLaDA.git",
    "dream": "https://github.com/DreamLM/Dream.git",
    "llada2": "https://github.com/inclusionAI/LLaDA2.X.git",
    "dinfer": "https://github.com/inclusionAI/dInfer.git",
    "bitsandbytes": "https://github.com/bitsandbytes-foundation/bitsandbytes.git",
    "autogptq": "https://github.com/AutoGPTQ/AutoGPTQ.git",
    "autoawq": "https://github.com/casper-hansen/AutoAWQ.git",
    "awq_official": "https://github.com/mit-han-lab/llm-awq.git",
    "spqr": "https://github.com/Vahe1994/SpQR.git",
    "aqlm": "https://github.com/Vahe1994/AQLM.git",
    "quip_sharp": "https://github.com/Cornell-RelaxML/quip-sharp.git",
    "kvquant": "https://github.com/SqueezeAILab/KVQuant.git",
    "kivi": "https://github.com/jy-yuan/KIVI.git",
    "qjl": "https://github.com/amirzandieh/QJL.git",
}

HF_MODELS: Dict[str, str] = {
    "mdlm_owt": "kuleshov-group/mdlm-owt",
    "llada_8b_base": "GSAI-ML/LLaDA-8B-Base",
    "llada_8b_instruct": "GSAI-ML/LLaDA-8B-Instruct",
    "dream_7b_base": "Dream-org/Dream-v0-Base-7B",
    "dream_7b_instruct": "Dream-org/Dream-v0-Instruct-7B",
    "dream_coder_7b_instruct": "Dream-org/Dream-Coder-v0-Instruct-7B",
    "llada_moe_7b_base": "inclusionAI/LLaDA-MoE-7B-A1B-Base",
    "llada_moe_7b_instruct": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
    "llada_moe_7b_instruct_td": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct-TD",
    "llada2_mini": "inclusionAI/LLaDA2.0-mini",
    "llada2_mini_cap": "inclusionAI/LLaDA2.0-mini-CAP",
    "llada2_flash": "inclusionAI/LLaDA2.0-flash",
    "llada2_flash_cap": "inclusionAI/LLaDA2.0-flash-CAP",
}

DATASET_SPECS: Dict[str, Dict[str, Any]] = {
    "mmlu": {"path": "cais/mmlu", "name": "all", "split": "test", "kind": "mcq"},
    "arc_challenge": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test", "kind": "mcq"},
    "hellaswag": {"path": "hellaswag", "name": None, "split": "validation", "kind": "mcq"},
    "piqa": {"path": "ybisk/piqa", "name": "plain_text", "split": "validation", "kind": "mcq", "trust_remote_code": True},
    "gsm8k": {"path": "gsm8k", "name": "main", "split": "test", "kind": "qa"},
    "humaneval": {"path": "openai/openai_humaneval", "name": None, "split": "test", "kind": "code"},
    "mbpp": {"path": "google-research-datasets/mbpp", "name": "full", "split": "test", "kind": "code"},
    "wikitext": {"path": "Salesforce/wikitext", "name": "wikitext-103-raw-v1", "split": "validation", "kind": "text"},
}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def set_matmul_precision() -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Small shell helpers
# ---------------------------------------------------------------------------

def run(cmd: str, cwd: Optional[str] = None, check: bool = True, env: Optional[Mapping[str, str]] = None) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        text=True,
        env=None if env is None else {**os.environ, **dict(env)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {cmd}\n{proc.stdout}")
    return proc


def ensure_repo(name: str, root: str = "official_repos", branch: Optional[str] = None, pull: bool = False) -> str:
    url = OFFICIAL_REPOS[name]
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    repo_path = root_path / name
    if repo_path.exists():
        if pull:
            run("git pull", cwd=str(repo_path))
    else:
        branch_arg = f" --branch {branch}" if branch else ""
        run(f"git clone --depth 1{branch_arg} {url} {repo_path}")
    return str(repo_path)


def sync_official_repos(root: str = "official_repos", pull: bool = False) -> Dict[str, str]:
    out = {}
    for name in ("mdlm", "llada", "dream", "llada2", "dinfer"):
        try:
            out[name] = ensure_repo(name, root=root, pull=pull)
        except Exception as exc:
            warnings.warn(f"Could not clone {name}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    key: str
    model_id: str
    model_cls: str = "AutoModel"
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    default_steps: int = 128
    default_block_length: int = 32
    default_gen_length: int = 128
    uses_chat_template: bool = True
    mask_token_hint: Optional[int] = None
    requires_left_padding: bool = True
    native_generate: bool = False
    notes: str = ""


MODEL_SPECS: Dict[str, ModelSpec] = {
    "llada_8b_instruct": ModelSpec(
        key="llada_8b_instruct",
        model_id=HF_MODELS["llada_8b_instruct"],
        model_cls="AutoModel",
        default_steps=128,
        default_block_length=32,
        default_gen_length=128,
        uses_chat_template=True,
        mask_token_hint=126336,
        native_generate=False,
        notes="LLaDA-style masked diffusion generation.",
    ),
    "llada_8b_base": ModelSpec(
        key="llada_8b_base",
        model_id=HF_MODELS["llada_8b_base"],
        model_cls="AutoModel",
        default_steps=128,
        default_block_length=32,
        default_gen_length=128,
        uses_chat_template=False,
        mask_token_hint=126336,
        native_generate=False,
    ),
    "llada_moe_7b_base": ModelSpec(
        key="llada_moe_7b_base",
        model_id=HF_MODELS["llada_moe_7b_base"],
        model_cls="AutoModelForCausalLM",
        default_steps=64,
        default_block_length=64,
        default_gen_length=256,
        uses_chat_template=False,
        mask_token_hint=126336,
        native_generate=False,
        notes="Optional LLaDA-MoE scaling checkpoint; dInfer can benchmark it directly.",
    ),
    "llada_moe_7b_instruct": ModelSpec(
        key="llada_moe_7b_instruct",
        model_id=HF_MODELS["llada_moe_7b_instruct"],
        model_cls="AutoModelForCausalLM",
        default_steps=64,
        default_block_length=64,
        default_gen_length=256,
        uses_chat_template=True,
        mask_token_hint=126336,
        native_generate=False,
        notes="Open 7B-A1B MoE diffusion LLM released by InclusionAI.",
    ),
    "llada_moe_7b_instruct_td": ModelSpec(
        key="llada_moe_7b_instruct_td",
        model_id=HF_MODELS["llada_moe_7b_instruct_td"],
        model_cls="AutoModelForCausalLM",
        default_steps=32,
        default_block_length=64,
        default_gen_length=256,
        uses_chat_template=True,
        mask_token_hint=126336,
        native_generate=False,
        notes="Trajectory-distilled MoE diffusion model for step-reduction combination studies.",
    ),
    "dream_7b_instruct": ModelSpec(
        key="dream_7b_instruct",
        model_id=HF_MODELS["dream_7b_instruct"],
        model_cls="AutoModel",
        default_steps=128,
        default_block_length=32,
        default_gen_length=128,
        uses_chat_template=True,
        mask_token_hint=None,
        native_generate=False,
        notes="The notebook uses a generic LLaDA-style sampler by default for consistent stage tracking.",
    ),
    "dream_7b_base": ModelSpec(
        key="dream_7b_base",
        model_id=HF_MODELS["dream_7b_base"],
        model_cls="AutoModel",
        default_steps=128,
        default_block_length=32,
        default_gen_length=128,
        uses_chat_template=False,
        mask_token_hint=None,
        native_generate=False,
    ),
    "dream_coder_7b_instruct": ModelSpec(
        key="dream_coder_7b_instruct",
        model_id=HF_MODELS["dream_coder_7b_instruct"],
        model_cls="AutoModel",
        default_steps=128,
        default_block_length=32,
        default_gen_length=256,
        uses_chat_template=True,
        mask_token_hint=None,
        native_generate=False,
    ),
    "llada2_mini": ModelSpec(
        key="llada2_mini",
        model_id=HF_MODELS["llada2_mini"],
        model_cls="AutoModelForCausalLM",
        default_steps=32,
        default_block_length=32,
        default_gen_length=256,
        uses_chat_template=True,
        mask_token_hint=None,
        native_generate=False,
    ),
    "llada2_mini_cap": ModelSpec(
        key="llada2_mini_cap",
        model_id=HF_MODELS["llada2_mini_cap"],
        model_cls="AutoModelForCausalLM",
        default_steps=32,
        default_block_length=32,
        default_gen_length=256,
        uses_chat_template=True,
        mask_token_hint=None,
        native_generate=False,
        notes="CAP-enhanced mini MoE checkpoint for fast inference comparisons.",
    ),
    "llada2_flash": ModelSpec(
        key="llada2_flash",
        model_id=HF_MODELS["llada2_flash"],
        model_cls="AutoModelForCausalLM",
        default_steps=32,
        default_block_length=32,
        default_gen_length=256,
        uses_chat_template=True,
        mask_token_hint=None,
        native_generate=False,
        notes="Optional 100B MoE scaling section. Use reduced slices, device_map=auto, or dInfer when memory is tight.",
    ),
    "llada2_flash_cap": ModelSpec(
        key="llada2_flash_cap",
        model_id=HF_MODELS["llada2_flash_cap"],
        model_cls="AutoModelForCausalLM",
        default_steps=32,
        default_block_length=32,
        default_gen_length=256,
        uses_chat_template=True,
        mask_token_hint=None,
        native_generate=False,
        notes="CAP-enhanced 100B MoE checkpoint for appendix throughput studies.",
    ),
    "mdlm_owt": ModelSpec(
        key="mdlm_owt",
        model_id=HF_MODELS["mdlm_owt"],
        model_cls="AutoModel",
        default_steps=1000,
        default_block_length=64,
        default_gen_length=128,
        uses_chat_template=False,
        native_generate=False,
        notes="Appendix/scaling checkpoint; use OpenWebText/WikiText perplexity or sample quality.",
    ),
}


@dataclass
class GenerateConfig:
    steps: int = 128
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    top_p: float = 1.0
    do_sample: bool = False
    stop_strings: Tuple[str, ...] = tuple()
    batch_size: int = 1


@dataclass
class QuantConfig:
    method: str = "turboldm"
    w_bits: int = 4
    a_bits: Optional[int] = None
    base_bits: int = 3
    residual_bits: int = 1
    group_size: int = 128
    block_size: int = 128
    stage_count: int = 4
    stage_ratios: Tuple[float, ...] = (0.85, 0.60, 0.35, 0.15)
    rotation: str = "hadamard"
    damping: float = 0.01
    awq_alpha: float = 0.5
    smoothquant_alpha: float = 0.5
    residual_top_fraction: float = 0.15
    # Phase 4 reviewer ablation. Default "stage" = current sensitivity-ranked
    # (layer, stage) pairs; "none" = pure base quant; "random" = random pairs
    # at the same top-fraction; "static_layer" = pick ``residual_top_fraction``
    # fraction of *layers* and activate every stage on them (so residuals are
    # layer-only, no stage gating); "stage_entropy" = stage + entropy_gating
    # (kept for symmetry).
    schedule_mode: str = "stage"
    static_high_precision_fraction: float = 0.2
    outlier_fraction: float = 0.001
    sensitivity_metric: str = "hidden_mse"
    entropy_gating: bool = False
    entropy_quantile: float = 0.50
    inner_product_attention_residual: bool = True
    quantize_output_head: bool = False
    # Phase 5 reviewer ablation. If True, the LM head (LLaDA's
    # ``transformer.ff_out``, shape [vocab, hidden]) is included in the set
    # of quantised linears; this bumps ``iter_named_linears``'
    # ``max_out_features`` cap. If ``include_embeddings`` is True we also
    # quantise ``nn.Embedding`` tables; currently this is unsupported and
    # raises NotImplementedError if set.
    include_lm_head: bool = False
    include_embeddings: bool = False
    include_regex: Optional[str] = None
    exclude_regex: str = r"(lm_head|output|embed|embedding|router|gate\.weight$)"
    calibration_batch_size: int = 2
    calibration_max_tokens: int = 512
    calibration_max_examples: int = 128
    probe_max_examples: int = 16
    cache_dequantized: bool = False
    compute_stage_scores: bool = True
    external_first: bool = True


@dataclass
class RunConfig:
    output_dir: str = "runs/turboldm_notebook"
    cache_dir: str = "cache"
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    calibration_text_dataset: str = "wikitext"
    calibration_text_field: str = "text"
    default_context_length: int = 512
    max_examples_per_dataset: Dict[str, Optional[int]] = field(default_factory=lambda: {
        "mmlu": 200,
        "arc_challenge": 300,
        "hellaswag": 500,
        "piqa": 500,
        "gsm8k": 200,
        "humaneval": 164,
        "mbpp": 200,
    })


@dataclass
class ResultRecord:
    model: str
    baseline: str
    dataset: str
    split: str
    num_examples: int
    score_name: str
    score_value: float
    elapsed_sec: float
    peak_mem_gb: float
    param_storage_gb: float
    avg_bits_per_param: float
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["extra"] = json.dumps(self.extra, sort_keys=True)
        return payload


# ---------------------------------------------------------------------------
# Stage / entropy context
# ---------------------------------------------------------------------------

class StageContext:
    current_stage: int = 0
    current_ratio: float = 1.0
    current_step: int = 0
    prev_entropy: float = float("inf")

    @classmethod
    def reset(cls) -> None:
        cls.current_stage = 0
        cls.current_ratio = 1.0
        cls.current_step = 0
        cls.prev_entropy = float("inf")


def ratio_to_stage(mask_ratio: float, stage_ratios: Sequence[float]) -> int:
    for idx, threshold in enumerate(stage_ratios):
        if mask_ratio >= threshold:
            return idx
    return max(len(stage_ratios) - 1, 0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_torch_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping[dtype]


def load_model_and_tokenizer(
    spec: ModelSpec,
    device: str = "cuda",
    dtype: Optional[str] = None,
    device_map: Optional[str] = None,
    low_cpu_mem_usage: bool = True,
) -> Tuple[nn.Module, Any]:
    if AutoTokenizer is None:
        raise ImportError("transformers is not installed")
    dtype = dtype or spec.dtype
    torch_dtype = resolve_torch_dtype(dtype)
    tok = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
    if spec.requires_left_padding and getattr(tok, "padding_side", None) != "left":
        tok.padding_side = "left"
    if spec.model_cls == "AutoModelForCausalLM":
        cls = AutoModelForCausalLM
    else:
        cls = AutoModel
    kwargs = {
        "trust_remote_code": spec.trust_remote_code,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    else:
        kwargs["torch_dtype"] = torch_dtype
    model = cls.from_pretrained(spec.model_id, **kwargs)
    if device_map is None:
        model = model.to(device=device, dtype=torch_dtype)
    else:
        try:
            model = model.to(torch_dtype)
        except Exception:
            pass
    model.eval()
    return model, tok


def infer_mask_token_id(tokenizer: Any, model: Optional[nn.Module] = None, spec: Optional[ModelSpec] = None) -> int:
    for value in (
        getattr(tokenizer, "mask_token_id", None),
        getattr(getattr(model, "config", None), "mask_token_id", None),
        getattr(spec, "mask_token_hint", None),
    ):
        if value is not None:
            return int(value)
    vocab = None
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = {}
    candidates = []
    for token in ("[MASK]", "<mask>", "<|mask|>", "<mask_1>", "<extra_id_0>"):
        if token in vocab:
            candidates.append(vocab[token])
    if candidates:
        return int(candidates[0])
    specials = getattr(tokenizer, "special_tokens_map_extended", {})
    for _, token in specials.items():
        if isinstance(token, str) and "mask" in token.lower():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id >= 0:
                return int(token_id)
    raise ValueError("Could not infer the mask token id. Set spec.mask_token_hint manually.")


def model_forward_logits(model: nn.Module, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # LLaDA2.0 requires a 4D block attention mask of shape (B, 1, L, L) as
    # *float* (transformers' _prepare_4d_causal_attention_mask_for_sdpa
    # calls into _unmask_unattended which rejects BoolTensor).
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    # Match the model's compute dtype (SDPA requires matching dtype for mask + query).
    try:
        model_dtype = next(p.dtype for p in model.parameters() if p.is_floating_point())
    except StopIteration:
        model_dtype = torch.float32
    four_d_float = None
    if attention_mask is not None and attention_mask.dim() == 2:
        row = attention_mask.to(model_dtype).unsqueeze(1).unsqueeze(2)
        col = attention_mask.to(model_dtype).unsqueeze(1).unsqueeze(-1)
        four_d_float = (row * col).to(device)
    elif attention_mask is None:
        four_d_float = torch.ones(bsz, 1, seq_len, seq_len, dtype=model_dtype, device=device)
    # MDLM-style diffusion LMs take a ``timesteps`` kwarg in place of
    # attention_mask; sigma=0 corresponds to the pre-training loss at the
    # sampled masking ratio.
    zero_sigma = torch.zeros(bsz, device=device, dtype=torch.float32)
    # LLaDA2.0 requires a 4D block attention mask at the outer boundary.
    # Detect it by class name and try that candidate FIRST so we don't
    # pay a failed-forward penalty for every batch.
    model_class_name = type(model).__name__.lower()
    prefers_4d = "llada2" in model_class_name
    if prefers_4d and four_d_float is not None:
        candidates = [
            {"input_ids": input_ids, "attention_mask": four_d_float},
            {"input_ids": input_ids, "attention_mask": attention_mask},
            {"input_ids": input_ids, "timesteps": zero_sigma},
            {"input_ids": input_ids},
        ]
    else:
        candidates = [
            {"input_ids": input_ids, "attention_mask": attention_mask},
            {"input_ids": input_ids, "attention_mask": four_d_float} if four_d_float is not None else None,
            {"input_ids": input_ids, "timesteps": zero_sigma},
            {"input_ids": input_ids},
        ]
    candidates = [c for c in candidates if c is not None]
    last_error = None
    for kwargs in candidates:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            out = model(**kwargs)
            if hasattr(out, "logits") and out.logits is not None:
                return out.logits
            if isinstance(out, (tuple, list)) and len(out) > 0:
                return out[0]
            if torch.is_tensor(out):
                return out
        except Exception as exc:
            last_error = exc
            continue
    try:
        out = model(input_ids, attention_mask=attention_mask)
        if hasattr(out, "logits") and out.logits is not None:
            return out.logits
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        if torch.is_tensor(out):
            return out
    except Exception as exc:
        last_error = exc
    raise RuntimeError(f"Could not obtain logits from model forward() call: {last_error}")


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_chat_prompt(tokenizer: Any, text: str, add_generation_prompt: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        except Exception:
            pass
    return text


def choice_block(options: Sequence[str]) -> str:
    return "\n".join(f"{CHOICE_LABELS[i]}. {opt}" for i, opt in enumerate(options))


def format_mcq_example(dataset_name: str, ex: Mapping[str, Any]) -> Tuple[str, str]:
    if dataset_name == "mmlu":
        options = list(ex["choices"])
        answer_idx = int(ex["answer"])
        prompt = (
            "Answer the multiple-choice question. Respond with only the option letter.\n\n"
            f"Question: {ex['question']}\n"
            f"Options:\n{choice_block(options)}\n\nAnswer:"
        )
        return prompt, CHOICE_LABELS[answer_idx]
    if dataset_name == "arc_challenge":
        choices = ex["choices"]
        labels = list(choices["label"])
        texts = list(choices["text"])
        gold = str(ex["answerKey"]).strip()
        order = []
        for idx, text in enumerate(texts):
            label = labels[idx] if idx < len(labels) else CHOICE_LABELS[idx]
            order.append((label, text))
        prompt = (
            "Answer the multiple-choice science question. Respond with only the option letter.\n\n"
            f"Question: {ex['question']}\n"
            f"Options:\n" + "\n".join(f"{label}. {text}" for label, text in order) + "\n\nAnswer:"
        )
        return prompt, gold
    if dataset_name == "hellaswag":
        ctx = ex.get("ctx") or ((ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip())
        options = list(ex["endings"])
        answer_idx = int(ex["label"])
        prompt = (
            "Choose the best ending for the context. Respond with only the option letter.\n\n"
            f"Context: {ctx}\n"
            f"Options:\n{choice_block(options)}\n\nAnswer:"
        )
        return prompt, CHOICE_LABELS[answer_idx]
    if dataset_name == "piqa":
        options = [ex["sol1"], ex["sol2"]]
        answer_idx = int(ex["label"])
        prompt = (
            "Choose the better solution to the goal. Respond with only the option letter.\n\n"
            f"Goal: {ex['goal']}\n"
            f"Options:\n{choice_block(options)}\n\nAnswer:"
        )
        return prompt, CHOICE_LABELS[answer_idx]
    raise KeyError(f"Unsupported MCQ dataset: {dataset_name}")


def format_gsm8k_example(ex: Mapping[str, Any]) -> Tuple[str, str]:
    prompt = (
        "Solve the math word problem. Show brief reasoning if useful, but end with a final line in the form '#### <answer>'.\n\n"
        f"Question: {ex['question']}\n\nAnswer:"
    )
    answer = str(ex["answer"]).split("####")[-1].strip()
    return prompt, answer


def format_humaneval_example(ex: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    prompt = (
        "Complete the following Python function. Output only valid Python code.\n\n"
        f"{ex['prompt']}"
    )
    meta = {"test": ex["test"], "entry_point": ex["entry_point"]}
    return prompt, meta


def format_mbpp_example(ex: Mapping[str, Any]) -> Tuple[str, Dict[str, Any]]:
    tests = ex.get("test_list") or ex.get("test_setup_code", [])
    if isinstance(tests, str):
        tests = [tests]
    prompt = (
        "Write a correct Python function for the following task. Output only Python code.\n\n"
        f"Task: {ex['text']}\n"
    )
    meta = {"test_list": list(tests), "code": ex.get("code")}
    return prompt, meta


def parse_choice_answer(text: str, num_choices: int = 4) -> str:
    text = text.strip()
    if not text:
        return ""
    m = re.search(r"\b([A-Z])\b", text[:32])
    if m:
        c = m.group(1)
        if c in CHOICE_LABELS[:num_choices]:
            return c
    # fallback: look at the first capital letter
    for ch in text:
        if ch in CHOICE_LABELS[:num_choices]:
            return ch
    return ""


def normalize_number_text(text: str) -> str:
    text = text.replace(",", "")
    if "####" in text:
        text = text.split("####")[-1]
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()


def truncate_stop_strings(text: str, stop_strings: Sequence[str]) -> str:
    out = text
    for stop in stop_strings:
        if stop and stop in out:
            out = out.split(stop)[0]
    return out


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_hf_dataset(name: str, split_override: Optional[str] = None, max_examples: Optional[int] = None) -> Any:
    if load_dataset is None:
        raise ImportError("datasets is not installed")
    spec = DATASET_SPECS[name]
    kwargs = {}
    if spec.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    ds = load_dataset(spec["path"], spec["name"], split=split_override or spec["split"], **kwargs)
    if max_examples is not None:
        max_examples = min(max_examples, len(ds))
        ds = ds.select(range(max_examples))
    return ds


def load_calibration_texts(run_cfg: RunConfig, max_texts: int = 128) -> List[str]:
    ds_name = run_cfg.calibration_text_dataset
    field_name = run_cfg.calibration_text_field
    ds = load_hf_dataset(ds_name, max_examples=max_texts * 2)
    texts = []
    for ex in ds:
        value = ex.get(field_name)
        if not value:
            continue
        value = str(value).strip()
        if len(value) < 32:
            continue
        texts.append(value)
        if len(texts) >= max_texts:
            break
    if not texts:
        raise RuntimeError(f"No calibration texts found for dataset={ds_name} field={field_name}")
    return texts


# ---------------------------------------------------------------------------
# Tokenization and masking helpers
# ---------------------------------------------------------------------------

def batch_tokenize_texts(
    tokenizer: Any,
    texts: Sequence[str],
    max_length: int,
    padding: bool = True,
    truncation: bool = True,
    add_special_tokens: bool = True,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        padding="longest" if padding else False,
        truncation=truncation,
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    )
    return {k: v for k, v in encoded.items()}


def sample_mask_positions(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    ratio: float,
    mask_token_id: int,
    pad_token_id: Optional[int] = None,
    preserve_prefix: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = input_ids.device
    x = input_ids.clone()
    bsz, seq_len = x.shape
    attn = attention_mask if attention_mask is not None else torch.ones_like(x)
    candidate = attn.bool()
    if pad_token_id is not None:
        candidate &= x.ne(pad_token_id)
    if preserve_prefix > 0:
        candidate[:, :preserve_prefix] = False
    num_valid = candidate.sum(dim=1)
    mask = torch.zeros_like(candidate)
    for i in range(bsz):
        n = int(round(float(num_valid[i].item()) * ratio))
        n = max(1 if ratio > 0 else 0, min(n, int(num_valid[i].item())))
        if n <= 0:
            continue
        idx = torch.nonzero(candidate[i], as_tuple=False).squeeze(-1)
        perm = torch.randperm(len(idx), device=device)
        select = idx[perm[:n]]
        mask[i, select] = True
    x[mask] = mask_token_id
    return x, mask


def build_stage_calibration_batches(
    tokenizer: Any,
    texts: Sequence[str],
    spec: ModelSpec,
    quant_cfg: QuantConfig,
    run_cfg: RunConfig,
    device: str = "cuda",
) -> Dict[int, List[Dict[str, torch.Tensor]]]:
    encoded = batch_tokenize_texts(tokenizer, texts, max_length=quant_cfg.calibration_max_tokens)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    mask_token_id = infer_mask_token_id(tokenizer, spec=spec)
    stage_batches: Dict[int, List[Dict[str, torch.Tensor]]] = defaultdict(list)
    batch_size = quant_cfg.calibration_batch_size
    preserve_prefix = min(32, input_ids.shape[1] // 4)
    for stage, ratio in enumerate(quant_cfg.stage_ratios):
        for start in range(0, len(input_ids), batch_size):
            ids = input_ids[start:start+batch_size]
            attn = attention_mask[start:start+batch_size]
            masked_ids, mask_index = sample_mask_positions(
                ids,
                attn,
                ratio=ratio,
                mask_token_id=mask_token_id,
                pad_token_id=getattr(tokenizer, "pad_token_id", None),
                preserve_prefix=preserve_prefix,
            )
            stage_batches[stage].append({
                "input_ids": masked_ids.to(device),
                "attention_mask": attn.to(device),
                "labels": ids.to(device),
                "mask_index": mask_index.to(device),
                "ratio": torch.tensor([ratio], device=device),
            })
    return stage_batches


# ---------------------------------------------------------------------------
# Multiple model generation adapters
# ---------------------------------------------------------------------------

def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits64 = logits.to(torch.float64)
    noise = torch.rand_like(logits64)
    gumbel_noise = (-torch.log(noise)).pow(temperature)
    return logits64.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    out = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        out[i, :remainder[i]] += 1
    return out


@torch.no_grad()
def generic_masked_diffusion_generate(
    model: nn.Module,
    tokenizer: Any,
    prompt_input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    gen_cfg: GenerateConfig,
    mask_token_id: int,
    stage_ratios: Sequence[float],
    eos_token_blocklist: Optional[Sequence[int]] = None,
    confidence_token_blocklist: Optional[Sequence[int]] = None,
    return_history: bool = False,
) -> Dict[str, Any]:
    device = prompt_input_ids.device
    bsz = prompt_input_ids.shape[0]
    gen_length = gen_cfg.gen_length
    block_length = gen_cfg.block_length
    if gen_length % block_length != 0:
        raise ValueError("gen_length must be divisible by block_length")
    num_blocks = gen_length // block_length
    if gen_cfg.steps % num_blocks != 0:
        raise ValueError("steps must be divisible by num_blocks")
    steps_per_block = gen_cfg.steps // num_blocks

    x = torch.full((bsz, prompt_input_ids.shape[1] + gen_length), mask_token_id, dtype=torch.long, device=device)
    x[:, :prompt_input_ids.shape[1]] = prompt_input_ids
    if attention_mask is not None:
        full_attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((bsz, gen_length), dtype=attention_mask.dtype, device=device),
            ],
            dim=-1,
        )
    else:
        full_attention_mask = None
    prompt_index = x.ne(mask_token_id)
    history = []
    StageContext.reset()

    for num_block in range(num_blocks):
        block_start = prompt_input_ids.shape[1] + num_block * block_length
        block_end = prompt_input_ids.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end].eq(mask_token_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for local_step in range(steps_per_block):
            mask_index = x.eq(mask_token_id)
            mask_ratio = mask_index.float().mean().item()
            StageContext.current_ratio = mask_ratio
            StageContext.current_stage = ratio_to_stage(mask_ratio, stage_ratios)
            StageContext.current_step = local_step + num_block * steps_per_block

            if gen_cfg.cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_token_id
                x_ = torch.cat([x, un_x], dim=0)
                attn_ = None
                if full_attention_mask is not None:
                    attn_ = torch.cat([full_attention_mask, full_attention_mask], dim=0)
                logits = model_forward_logits(model, x_, attention_mask=attn_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (gen_cfg.cfg_scale + 1.0) * (logits - un_logits)
            else:
                logits = model_forward_logits(model, x, attention_mask=full_attention_mask)

            if eos_token_blocklist:
                for tok in eos_token_blocklist:
                    if 0 <= tok < logits.shape[-1]:
                        logits[:, :, tok] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=gen_cfg.temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if confidence_token_blocklist:
                for tok in confidence_token_blocklist:
                    if 0 <= tok < logits_with_noise.shape[-1]:
                        logits_with_noise[:, :, tok] = -torch.inf

            if gen_cfg.remasking == "low_confidence":
                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)
                masked_entropy = entropy[mask_index]
                if masked_entropy.numel() > 0:
                    StageContext.prev_entropy = float(masked_entropy.mean().item())
            elif gen_cfg.remasking == "random":
                x0_p = torch.rand(x0.shape, device=x0.device)
                StageContext.prev_entropy = float("inf")
            else:
                raise NotImplementedError(gen_cfg.remasking)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -np.inf))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for row in range(confidence.shape[0]):
                k = int(num_transfer_tokens[row, local_step].item())
                if k <= 0:
                    continue
                _, select_index = torch.topk(confidence[row], k=k)
                transfer_index[row, select_index] = True
            x[transfer_index] = x0[transfer_index]
            if return_history:
                history.append(x.clone().detach().cpu())

    StageContext.reset()
    return {"sequences": x, "history": history}


def generate_text_batch(
    model: nn.Module,
    tokenizer: Any,
    spec: ModelSpec,
    prompts: Sequence[str],
    gen_cfg: GenerateConfig,
    device: str = "cuda",
    return_raw_sequences: bool = False,
) -> List[str]:
    texts = [build_chat_prompt(tokenizer, prompt) if spec.uses_chat_template else prompt for prompt in prompts]
    enc = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=not spec.uses_chat_template)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    mask_token_id = infer_mask_token_id(tokenizer, model=model, spec=spec)
    out = generic_masked_diffusion_generate(
        model=model,
        tokenizer=tokenizer,
        prompt_input_ids=input_ids,
        attention_mask=attention_mask,
        gen_cfg=gen_cfg,
        mask_token_id=mask_token_id,
        stage_ratios=tuple(getattr(model, '_turboldm_stage_ratios', tuple(sorted(gen_cfg_to_stage_ratios(gen_cfg), reverse=True)))),
    )
    seq = out["sequences"]
    gen_only = seq[:, input_ids.shape[1]:]
    texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    texts = [truncate_stop_strings(t, gen_cfg.stop_strings) for t in texts]
    if return_raw_sequences:
        return texts, seq
    return texts


def gen_cfg_to_stage_ratios(gen_cfg: GenerateConfig) -> Tuple[float, ...]:
    # A generation config does not natively carry stage ratios; this helper keeps the
    # sampling API standalone. The quantization pipeline injects explicit stage ratios.
    if gen_cfg.steps >= 128:
        return (0.85, 0.60, 0.35, 0.15)
    return (0.75, 0.50, 0.25, 0.10)


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1) == 0)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform on the last dimension.

    Returns a unitary (self-inverse) normalized transform: ``fwht(fwht(x)) == x``.
    """
    N = x.shape[-1]
    if not is_power_of_two(N):
        raise ValueError("FWHT requires a power-of-two last dimension")
    lead_shape = x.shape[:-1]
    # Work on a flat-last-dim view and apply butterflies in place.
    y = x.reshape(-1, N).clone()
    h = 1
    while h < N:
        # View each length-N row as pairs of (h, h) blocks.
        y2 = y.reshape(y.shape[0], N // (2 * h), 2, h)
        a = y2[:, :, 0, :]
        b = y2[:, :, 1, :]
        y2 = torch.stack([a + b, a - b], dim=2)
        y = y2.reshape(y.shape[0], N)
        h *= 2
    return y.reshape(*lead_shape, N) / math.sqrt(N)


def make_rademacher(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.empty(*shape, device=device).bernoulli_(0.5).mul_(2).sub_(1)


def hadamard_rotate(x: torch.Tensor, signs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    if signs is None:
        signs = make_rademacher((x.shape[-1],), x.device)
    return fwht(x * signs), signs


def hadamard_inverse(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return fwht(x) * signs


def gaussian_codebook(bits: int, device: torch.device = torch.device("cpu"), num_samples: int = 200_000, seed: int = 42) -> torch.Tensor:
    levels = 2 ** bits
    rng = np.random.default_rng(seed + bits * 17)
    samples = rng.standard_normal(num_samples).astype(np.float32)
    boundaries = np.quantile(samples, np.linspace(0, 1, levels + 1))
    centroids = []
    for i in range(levels):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        mask = (samples >= lo) & (samples <= hi)
        subset = samples[mask]
        if subset.size == 0:
            centroids.append((lo + hi) / 2.0)
        else:
            centroids.append(float(subset.mean()))
    codebook = torch.tensor(centroids, dtype=torch.float32, device=device)
    return codebook


_CODEBOOK_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def get_codebook(bits: int, device: torch.device) -> torch.Tensor:
    key = (bits, str(device))
    if key not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[key] = gaussian_codebook(bits, device=device)
    return _CODEBOOK_CACHE[key]


def nearest_code_indices(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Map each element of ``x`` to the int8 index of the closest entry in ``codebook``.

    We use ``torch.bucketize`` on the midpoint boundaries of a sorted codebook so
    we never materialize a ``(..., d, K)`` broadcast tensor, and we chunk along
    the leading dim to keep the int64 output of ``bucketize`` bounded — a
    4096x96x128 tensor at int64 would be 3.2 GiB. Returned dtype is int8 when
    the codebook has at most 128 entries (all practical bit-widths for this
    project) or int16 otherwise.
    """
    if x.ndim == 0 or x.numel() == 0:
        return torch.empty(0, dtype=torch.int8, device=x.device)
    idx_dtype = torch.int8 if codebook.numel() <= 128 else torch.int16
    cb_sorted, perm = torch.sort(codebook)
    boundaries = (cb_sorted[:-1] + cb_sorted[1:]) * 0.5
    inv_perm = torch.empty_like(perm, dtype=torch.int32)
    inv_perm[perm] = torch.arange(perm.numel(), device=perm.device, dtype=torch.int32)
    # Chunk so the transient int64 from bucketize stays below ~32 MiB per step.
    lead = x.shape[0]
    chunk = max(1, min(lead, (1 << 22) // max(1, x.numel() // max(1, lead))))
    out_parts: List[torch.Tensor] = []
    for i in range(0, lead, chunk):
        xi = x[i:i+chunk]
        idx_sorted = torch.bucketize(xi, boundaries)  # int64
        narrowed = inv_perm[idx_sorted.to(torch.long)].to(idx_dtype)
        out_parts.append(narrowed)
        del idx_sorted, narrowed
    return torch.cat(out_parts, dim=0)


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int = -1) -> Tuple[torch.Tensor, int]:
    size = x.shape[dim]
    pad = (multiple - size % multiple) % multiple
    if pad == 0:
        return x, 0
    pad_shape = list(x.shape)
    pad_shape[dim] = pad
    pad_tensor = torch.zeros(*pad_shape, device=x.device, dtype=x.dtype)
    y = torch.cat([x, pad_tensor], dim=dim)
    return y, pad


def uniform_group_quantize(
    weight: torch.Tensor,
    bits: int,
    group_size: int = 128,
    symmetric: bool = True,
    importance: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    target_device = torch.device(device) if device is not None else weight.device
    w = weight.detach().to(target_device).float()
    out_dim, in_dim = w.shape
    w_pad, pad = pad_to_multiple(w, group_size, dim=1)
    groups = w_pad.reshape(out_dim, -1, group_size)
    if importance is not None:
        imp = importance.detach().to(target_device).float()
        if imp.numel() != in_dim:
            raise ValueError("importance length mismatch")
        imp_pad, _ = pad_to_multiple(imp.view(1, -1), group_size, dim=1)
        imp_g = imp_pad.reshape(1, -1, group_size)
        groups_scaled = groups * imp_g.sqrt()
    else:
        groups_scaled = groups
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        scales = groups_scaled.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(groups_scaled / scales).clamp(-qmax - 1, qmax).to(torch.int8)
        deq_scaled = q.float() * scales
        if importance is not None:
            deq = deq_scaled / imp_g.sqrt().clamp_min(1e-8)
        else:
            deq = deq_scaled
        return {
            "q": q,
            "scales": scales.squeeze(-1),
            "pad": torch.tensor([pad]),
            "bits": torch.tensor([bits]),
            "shape": torch.tensor([out_dim, in_dim]),
            "deq": deq.reshape(out_dim, -1)[:, :in_dim].to(weight.dtype),
        }
    raise NotImplementedError("Asymmetric quantization is omitted in this notebook for simplicity")


def turboquant_vector_quantize(
    weight: torch.Tensor,
    bits: int,
    block_size: int = 128,
    rotation: str = "hadamard",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Hadamard-rotation + Gaussian-codebook block vector quantization.

    `device` controls where the temporary fp32 tensors live. Pass `cpu` for
    large layers to keep GPU headroom; the returned `deq` keeps the caller's
    original dtype but lives on the chosen device — move it with `.to()` at
    the call site.
    """
    target_device = torch.device(device) if device is not None else weight.device
    w = weight.detach().to(target_device).float()
    out_dim, in_dim = w.shape
    w_pad, pad = pad_to_multiple(w, block_size, dim=1)
    groups = w_pad.reshape(out_dim, -1, block_size)
    if rotation != "hadamard":
        raise NotImplementedError("Only Hadamard-sign rotation is implemented in the notebook reference")
    signs = make_rademacher((groups.shape[1], block_size), groups.device)
    # Vectorized Hadamard rotation over all blocks simultaneously.
    rotated = fwht(groups * signs.unsqueeze(0))
    del w, w_pad, groups
    rms = rotated.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    normalized = rotated / rms
    codebook = get_codebook(bits, normalized.device)
    idx = nearest_code_indices(normalized, codebook)
    # Dequantize in chunks so we never hold an int64-sized lookup tensor in
    # memory at once; ``codebook[idx_long]`` requires int32/int64 and would
    # otherwise balloon the transient.
    q = torch.empty_like(normalized)
    lead = idx.shape[0]
    chunk = max(1, min(lead, (1 << 22) // max(1, idx.numel() // max(1, lead))))
    for i in range(0, lead, chunk):
        q[i:i+chunk] = codebook[idx[i:i+chunk].to(torch.long)]
    del normalized
    deq_rot = q * rms
    del q
    # Inverse: FWHT is its own inverse up to scale; sign multiplication undoes the sign twist.
    restored = fwht(deq_rot) * signs.unsqueeze(0)
    del deq_rot
    restored = restored.reshape(out_dim, -1)[:, :in_dim]
    return {
        "indices": idx,
        "rms": rms.squeeze(-1),
        "signs": signs.to(torch.int8),
        "pad": torch.tensor([pad]),
        "bits": torch.tensor([bits]),
        "shape": torch.tensor([out_dim, in_dim]),
        "deq": restored.to(weight.dtype),
    }


def qjl_residual_sign_quantize(weight: torch.Tensor, block_size: int = 128) -> Dict[str, Any]:
    w = weight.detach().float()
    out_dim, in_dim = w.shape
    w_pad, pad = pad_to_multiple(w, block_size, dim=1)
    groups = w_pad.reshape(out_dim, -1, block_size)
    signs = make_rademacher((groups.shape[1], block_size), groups.device)
    rotated = []
    for g_idx in range(groups.shape[1]):
        g = groups[:, g_idx, :]
        r, _ = hadamard_rotate(g, signs[g_idx])
        rotated.append(r.unsqueeze(1))
    rotated = torch.cat(rotated, dim=1)
    scale = rotated.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    sign_bits = rotated.sign().to(torch.int8)
    deq_rot = sign_bits.float() * scale
    restored = []
    for g_idx in range(deq_rot.shape[1]):
        g = deq_rot[:, g_idx, :]
        inv = hadamard_inverse(g, signs[g_idx])
        restored.append(inv.unsqueeze(1))
    restored = torch.cat(restored, dim=1).reshape(out_dim, -1)[:, :in_dim]
    return {
        "sign_bits": sign_bits,
        "scale": scale.squeeze(-1),
        "signs": signs.to(torch.int8),
        "pad": torch.tensor([pad]),
        "shape": torch.tensor([out_dim, in_dim]),
        "deq": restored.to(weight.dtype),
    }


# ---------------------------------------------------------------------------
# Quantized Linear modules
# ---------------------------------------------------------------------------

class QuantizedLinearBase(nn.Module):
    def __init__(self, bias: Optional[torch.Tensor], cache_dequantized: bool = False):
        super().__init__()
        if bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.cache_dequantized = cache_dequantized
        self._cached_weight: Optional[torch.Tensor] = None
        self.quant_metadata: Dict[str, Any] = {}

    def get_weight(self) -> torch.Tensor:
        raise NotImplementedError

    # Keys inside quant_metadata that are merely reference dequantized weights
    # and should not be counted as part of the low-bit storage footprint.
    _DEQ_STORAGE_KEYS = {"deq", "res_deq"}

    def quant_storage_bytes(self) -> int:
        total = 0
        for key, value in self.quant_metadata.items():
            if key in self._DEQ_STORAGE_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                # Integer index buffers represent their *actual* bit width,
                # not their storage dtype (indices are packed into int8/int16
                # only for tensor convenience). Convert using the recorded
                # `bits` / `res_bits` when available.
                if key in ("q",):
                    bits = int(self.quant_metadata.get("bits", torch.tensor([8])).item())
                    total += (value.numel() * bits + 7) // 8
                elif key in ("indices",):
                    bits = int(self.quant_metadata.get("bits", torch.tensor([8])).item())
                    total += (value.numel() * bits + 7) // 8
                elif key in ("res_indices", "res_q"):
                    bits = int(self.quant_metadata.get("res_bits", torch.tensor([1])).item())
                    total += (value.numel() * bits + 7) // 8
                elif key in ("signs", "res_signs"):
                    # Rademacher signs need 1 bit per entry.
                    total += (value.numel() + 7) // 8
                elif key in ("sign_bits", "qjl_sign_bits", "res_qjl_sign_bits"):
                    total += (value.numel() + 7) // 8
                elif key == "sparse_residual":
                    # Count only non-zero entries (SpQR stores sparse COO).
                    nnz = int((value != 0).sum().item())
                    total += nnz * value.element_size() + nnz * 8  # value + index
                else:
                    total += value.numel() * value.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def avg_bits_per_param(self) -> float:
        shape = self.quant_metadata.get("shape")
        if isinstance(shape, torch.Tensor):
            total_params = int(shape[0].item()) * int(shape[1].item())
        else:
            return float("nan")
        return 8.0 * self.quant_storage_bytes() / max(total_params, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)


class UniformLinear(QuantizedLinearBase):
    def __init__(self, linear: nn.Linear, quant_pack: Dict[str, torch.Tensor], input_scale: Optional[torch.Tensor] = None, cache_dequantized: bool = False):
        super().__init__(linear.bias, cache_dequantized=cache_dequantized)
        self.register_buffer("_weight", quant_pack["deq"].detach().clone())
        # Store metadata on CPU and drop the redundant `deq` tensor (already a buffer).
        self.quant_metadata = {
            k: v.detach().to("cpu") for k, v in quant_pack.items()
            if isinstance(v, torch.Tensor) and k != "deq"
        }
        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.detach().clone())
        else:
            self.input_scale = None

    def get_weight(self) -> torch.Tensor:
        return self._weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is not None:
            view = [1] * x.ndim
            view[-1] = self.input_scale.shape[0]
            x = x / self.input_scale.view(*view)
        return super().forward(x)


class UniformLinearChunked(QuantizedLinearBase):
    """Same packed storage as ``UniformLinear`` but reconstructs the BF16
    weight in row-chunks at forward time, never materialising the full
    matrix on GPU. Saves the per-layer BF16 buffer (~ in*out*2 bytes per
    layer) at the cost of a slightly heavier forward pass. This is the
    closest stand-in for a true low-bit kernel inside our pipeline.
    """
    def __init__(self, linear: nn.Linear, quant_pack: Dict[str, torch.Tensor],
                 input_scale: Optional[torch.Tensor] = None, chunk_size: int = 512,
                 cache_dequantized: bool = False):
        super().__init__(linear.bias, cache_dequantized=False)
        # Persist the small int8 tensor + scales on the same device as the bias / x.
        device = linear.weight.device
        dtype = linear.weight.dtype
        q = quant_pack["q"].detach()
        scales = quant_pack["scales"].detach()
        if scales.dim() == 2:
            scales = scales.unsqueeze(-1)
        # Move codes to GPU so forward doesn't pay PCIe per call.
        self.register_buffer("q_codes", q.to(device=device))
        self.register_buffer("scales", scales.to(device=device, dtype=torch.float32))
        self.quant_metadata = {
            k: v.detach().to("cpu") for k, v in quant_pack.items()
            if isinstance(v, torch.Tensor) and k != "deq" and k not in ("q", "scales")
        }
        # Keep these in metadata too (for storage accounting only).
        self.quant_metadata["q"] = q.detach().to("cpu")
        self.quant_metadata["scales"] = scales.detach().to("cpu")
        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.detach().clone())
        else:
            self.input_scale = None
        self.chunk_size = chunk_size
        self.out_dim = int(quant_pack["shape"][0].item())
        self.in_dim = int(quant_pack["shape"][1].item())
        self.target_dtype = dtype

    def _reconstruct_chunk(self, i: int, j: int) -> torch.Tensor:
        # q_codes shape: (out_dim, n_groups, group_size)
        # scales shape:  (out_dim, n_groups, 1)
        q = self.q_codes[i:j].to(torch.float32)
        s = self.scales[i:j]
        w = (q * s).reshape(j - i, -1)[:, :self.in_dim]
        return w.to(self.target_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_scale is not None:
            view = [1] * x.ndim
            view[-1] = self.input_scale.shape[0]
            x = x / self.input_scale.view(*view)
        out_shape = list(x.shape[:-1]) + [self.out_dim]
        y = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        for i in range(0, self.out_dim, self.chunk_size):
            j = min(i + self.chunk_size, self.out_dim)
            w_chunk = self._reconstruct_chunk(i, j)
            y[..., i:j] = torch.nn.functional.linear(x, w_chunk)
        if self.bias is not None:
            y = y + self.bias
        return y


class SpQRLinear(QuantizedLinearBase):
    def __init__(self, linear: nn.Linear, base_pack: Dict[str, torch.Tensor], sparse_residual: torch.Tensor, cache_dequantized: bool = False):
        super().__init__(linear.bias, cache_dequantized=cache_dequantized)
        self.register_buffer("base_weight", base_pack["deq"].detach().clone())
        self.register_buffer("sparse_residual", sparse_residual.detach().clone())
        # CPU metadata only, drop `deq` duplicate.
        self.quant_metadata = {
            k: v.detach().to("cpu") for k, v in base_pack.items()
            if isinstance(v, torch.Tensor) and k != "deq"
        }
        self.quant_metadata["sparse_residual"] = sparse_residual.detach().to("cpu")

    def get_weight(self) -> torch.Tensor:
        return self.base_weight + self.sparse_residual


class LazyTurboLDMLinear(QuantizedLinearBase):
    """TurboLDM linear that reconstructs the BF16 weight on every forward.

    Stores only the compact TurboQuant representation (int8 indices, per-block
    RMS, random sign vector, and a tiny codebook) plus, optionally, a
    residual code. Produces genuine low-bit STORAGE so 16B-class models fit
    on 32 GB VRAM; pays a small per-forward reconstruction cost in return.
    """

    def __init__(
        self,
        linear: nn.Linear,
        base_pack: Dict[str, Any],
        residual_pack: Optional[Dict[str, Any]] = None,
        active_stages: Sequence[int] = (),
        entropy_thresholds: Optional[Mapping[int, float]] = None,
    ):
        super().__init__(linear.bias, cache_dequantized=False)
        self.out_dim = int(base_pack["shape"][0].item())
        self.in_dim = int(base_pack["shape"][1].item())
        self.pad = int(base_pack["pad"].item())
        self.base_bits = int(base_pack["bits"].item())
        self.block_size = int(base_pack["indices"].shape[-1])
        self.register_buffer("base_indices", base_pack["indices"].detach().clone())
        self.register_buffer("base_rms", base_pack["rms"].detach().float().clone())
        self.register_buffer("base_signs", base_pack["signs"].detach().to(torch.int8).clone())
        cb_device = self.base_indices.device
        self.register_buffer("base_codebook", get_codebook(self.base_bits, torch.device("cpu")).to(cb_device).clone())
        self.has_residual = residual_pack is not None
        if residual_pack is not None:
            self.res_bits = int(residual_pack["bits"].item())
            self.register_buffer("res_indices", residual_pack["indices"].detach().clone())
            self.register_buffer("res_rms", residual_pack["rms"].detach().float().clone())
            self.register_buffer("res_signs", residual_pack["signs"].detach().to(torch.int8).clone())
            self.register_buffer("res_codebook", get_codebook(self.res_bits, torch.device("cpu")).to(cb_device).clone())
        self.active_stages = set(int(s) for s in active_stages)
        self.entropy_thresholds = {int(k): float(v) for k, v in (entropy_thresholds or {}).items()}
        # CPU-side metadata so quant_storage_bytes() reports the true
        # compressed footprint.
        self.quant_metadata = {
            "indices": self.base_indices.cpu(),
            "rms": self.base_rms.cpu(),
            "signs": self.base_signs.cpu(),
            "bits": torch.tensor([self.base_bits]),
            "shape": torch.tensor([self.out_dim, self.in_dim]),
        }
        if residual_pack is not None:
            self.quant_metadata["res_indices"] = self.res_indices.cpu()
            self.quant_metadata["res_rms"] = self.res_rms.cpu()
            self.quant_metadata["res_signs"] = self.res_signs.cpu()
            self.quant_metadata["res_bits"] = torch.tensor([self.res_bits])

    def _residual_active(self) -> bool:
        if not self.has_residual:
            return False
        stage = int(StageContext.current_stage)
        if stage not in self.active_stages:
            return False
        if self.entropy_thresholds:
            threshold = self.entropy_thresholds.get(stage, None)
            if threshold is not None and StageContext.prev_entropy < threshold:
                return False
        return True

    def _reconstruct(self, indices: torch.Tensor, rms: torch.Tensor,
                      signs: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        # indices: (out_dim, n_blocks, block_size) int8
        # rms: (out_dim, n_blocks) float32
        # signs: (n_blocks, block_size) int8
        idx = indices.to(torch.long)
        q = codebook[idx]                               # (O, B, S) float32
        deq_rot = q * rms.unsqueeze(-1)                 # (O, B, S) float32
        # Inverse Hadamard: fwht is self-inverse; then multiply by signs.
        signs_f = signs.to(deq_rot.dtype).unsqueeze(0)  # (1, B, S)
        restored = fwht(deq_rot) * signs_f              # (O, B, S)
        out = restored.reshape(self.out_dim, -1)[:, :self.in_dim]
        return out.to(self.base_rms.dtype if self.base_rms.dtype == torch.float32 else self.base_rms.dtype)

    def get_weight(self) -> torch.Tensor:
        base = self._reconstruct(self.base_indices, self.base_rms, self.base_signs, self.base_codebook)
        if self._residual_active():
            res = self._reconstruct(self.res_indices, self.res_rms, self.res_signs, self.res_codebook)
            base = base + res
        # Cast to the bias dtype so F.linear produces the expected dtype.
        target_dtype = self.bias.dtype if self.bias is not None else torch.bfloat16
        return base.to(target_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.get_weight()
        out = F.linear(x, w, self.bias)
        # Release the reconstructed weight so VRAM doesn't grow across layers.
        del w
        return out


class TurboLDMLinear(QuantizedLinearBase):
    def __init__(
        self,
        linear: nn.Linear,
        base_pack: Dict[str, Any],
        residual_pack: Optional[Dict[str, Any]],
        active_stages: Sequence[int],
        entropy_thresholds: Optional[Mapping[int, float]] = None,
        cache_dequantized: bool = False,
    ):
        super().__init__(linear.bias, cache_dequantized=cache_dequantized)
        self.register_buffer("base_weight", base_pack["deq"].detach().clone())
        if residual_pack is None:
            self.residual_weight = None
        else:
            self.register_buffer("residual_weight", residual_pack["deq"].detach().clone())
        self.active_stages = set(int(s) for s in active_stages)
        self.entropy_thresholds = {int(k): float(v) for k, v in (entropy_thresholds or {}).items()}
        # Metadata kept on CPU and deduped — the dequantized tensors are already
        # live as GPU buffers (`base_weight` / `residual_weight`).
        self.quant_metadata = {
            k: v.detach().to("cpu") for k, v in base_pack.items()
            if isinstance(v, torch.Tensor) and k != "deq"
        }
        if residual_pack is not None:
            for k, v in residual_pack.items():
                if isinstance(v, torch.Tensor) and k != "deq":
                    self.quant_metadata[f"res_{k}"] = v.detach().to("cpu")

    def residual_is_active(self) -> bool:
        if self.residual_weight is None:
            return False
        stage = int(StageContext.current_stage)
        if stage not in self.active_stages:
            return False
        if self.entropy_thresholds:
            threshold = self.entropy_thresholds.get(stage, None)
            if threshold is not None and StageContext.prev_entropy < threshold:
                return False
        return True

    def get_weight(self) -> torch.Tensor:
        if self.residual_is_active():
            return self.base_weight + self.residual_weight
        return self.base_weight


# ---------------------------------------------------------------------------
# Module discovery / replacement
# ---------------------------------------------------------------------------

def iter_named_linears(
    model: nn.Module,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    max_out_features: int = 50000,
) -> Iterator[Tuple[str, nn.Linear]]:
    """Iterate over `nn.Linear` modules matching the include / exclude regex.

    `max_out_features` is a safety filter: projections with an output dim larger
    than this are assumed to be vocabulary heads (e.g. LLaDA's
    `model.transformer.ff_out` with shape [126464, 4096]) and are excluded,
    even if they do not match the exclude regex. We also skip modules whose
    weight parameter is on the ``meta`` device (accelerate lazy-dispatch)
    because we have no data to quantize there.
    """
    include = re.compile(include_regex) if include_regex else None
    exclude = re.compile(exclude_regex) if exclude_regex else None
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if include and not include.search(name):
            continue
        if exclude and exclude.search(name):
            continue
        if module.weight.shape[0] > max_out_features or module.weight.shape[1] > max_out_features:
            continue
        if module.weight.device.type == "meta":
            continue
        yield name, module


def get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, leaf = get_parent_module(model, module_name)
    if leaf.isdigit():
        parent[int(leaf)] = new_module  # type: ignore[index]
    else:
        setattr(parent, leaf, new_module)


# ---------------------------------------------------------------------------
# Activation stats collection
# ---------------------------------------------------------------------------

class ActivationCollector:
    def __init__(self, module_names: Sequence[str], max_tokens_per_stage_layer: int = 4096):
        self.module_names = set(module_names)
        self.max_tokens_per_stage_layer = max_tokens_per_stage_layer
        self.inputs: Dict[Tuple[str, int], List[torch.Tensor]] = defaultdict(list)
        self._key_token_count: Dict[Tuple[str, int], int] = defaultdict(int)
        self.handles: List[Any] = []

    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            key = (name, int(StageContext.current_stage))
            max_keep = self.max_tokens_per_stage_layer
            already = self._key_token_count[key]
            if already >= max_keep:
                return
            remaining = max_keep - already
            # Store in half precision to cap CPU RAM; we cast to float32 on read.
            flat = x.detach().reshape(-1, x.shape[-1]).to(torch.bfloat16)
            if flat.numel() == 0:
                return
            if flat.shape[0] > remaining:
                idx = torch.randperm(flat.shape[0], device=flat.device)[:remaining]
                flat = flat[idx]
            self.inputs[key].append(flat.cpu())
            self._key_token_count[key] += flat.shape[0]
        return hook

    def attach(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if name in self.module_names:
                self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def close(self) -> None:
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def get_stage_tensor(self, name: str, stage: int) -> Optional[torch.Tensor]:
        chunks = self.inputs.get((name, stage), [])
        if not chunks:
            return None
        x = torch.cat(chunks, dim=0).float()
        if x.shape[0] > self.max_tokens_per_stage_layer:
            x = x[:self.max_tokens_per_stage_layer]
        return x


def collect_activation_stats(
    model: nn.Module,
    module_names: Sequence[str],
    stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]],
    max_tokens_per_stage_layer: int = 1024,
) -> ActivationCollector:
    collector = ActivationCollector(module_names=module_names, max_tokens_per_stage_layer=max_tokens_per_stage_layer)
    collector.attach(model)
    model.eval()
    with torch.no_grad():
        for stage, batches in stage_batches.items():
            StageContext.current_stage = int(stage)
            for batch in batches:
                _ = model_forward_logits(model, batch["input_ids"], attention_mask=batch.get("attention_mask"))
    StageContext.reset()
    collector.close()
    return collector


# ---------------------------------------------------------------------------
# Sensitivity estimates
# ---------------------------------------------------------------------------

def layer_output_mse_score(linear: nn.Linear, activations: torch.Tensor, deq_weight: torch.Tensor) -> float:
    with torch.no_grad():
        y_fp = F.linear(activations, linear.weight.detach().float(), linear.bias.detach().float() if linear.bias is not None else None)
        y_q = F.linear(activations, deq_weight.detach().float(), linear.bias.detach().float() if linear.bias is not None else None)
        return float(F.mse_loss(y_q, y_fp).item())


def hidden_mse_scores(
    model: nn.Module,
    collector: ActivationCollector,
    quant_cfg: QuantConfig,
    method: str = "uniform",
) -> pd.DataFrame:
    rows = []
    for name, module in iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex):
        # Quantize once per layer, reuse across stages. Run on the module's
        # device for speed; each iteration's scratch is released before the next.
        if method == "turbo":
            pack = turboquant_vector_quantize(
                module.weight.detach(), bits=quant_cfg.base_bits,
                block_size=quant_cfg.block_size, rotation=quant_cfg.rotation,
            )
        else:
            pack = uniform_group_quantize(module.weight.detach(), bits=quant_cfg.w_bits, group_size=quant_cfg.group_size)
        deq = pack["deq"]
        if deq.device != module.weight.device:
            deq = deq.to(module.weight.device)
        for stage in range(quant_cfg.stage_count):
            act = collector.get_stage_tensor(name, stage)
            if act is None:
                continue
            score = layer_output_mse_score(module, act.to(module.weight.device), deq)
            rows.append({"layer": name, "stage": stage, "score": score})
        del pack, deq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("No sensitivity rows collected")
    return pd.DataFrame(rows)


def masked_token_kl_score(
    model: nn.Module,
    layer_name: str,
    quantized_module: nn.Module,
    batches: Sequence[Mapping[str, torch.Tensor]],
) -> float:
    parent, leaf = get_parent_module(model, layer_name)
    original = getattr(parent, leaf)
    replace_module(model, layer_name, quantized_module)
    try:
        total_kl = 0.0
        total_count = 0
        for batch in batches:
            with torch.no_grad():
                StageContext.current_stage = ratio_to_stage(float(batch["ratio"].item()), getattr(model, '_turboldm_stage_ratios', (0.85, 0.60, 0.35, 0.15)))
                logits_q = model_forward_logits(model, batch["input_ids"], attention_mask=batch.get("attention_mask"))
                replace_module(model, layer_name, original)
                logits_fp = model_forward_logits(model, batch["input_ids"], attention_mask=batch.get("attention_mask"))
                replace_module(model, layer_name, quantized_module)
                mask = batch["mask_index"]
                p = F.log_softmax(logits_fp.float(), dim=-1)
                q = F.log_softmax(logits_q.float(), dim=-1)
                kl = (p.exp() * (p - q)).sum(dim=-1)
                total_kl += float(kl[mask].sum().item())
                total_count += int(mask.sum().item())
        return total_kl / max(total_count, 1)
    finally:
        replace_module(model, layer_name, original)
        StageContext.reset()


def compute_stage_layer_scores(
    model: nn.Module,
    stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]],
    quant_cfg: QuantConfig,
) -> pd.DataFrame:
    module_names = [name for name, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)]
    if quant_cfg.sensitivity_metric in ("hidden_mse", "attn_mse", "act_max"):
        collector = collect_activation_stats(model, module_names, stage_batches)
        df = hidden_mse_scores(model, collector, quant_cfg, method="turbo" if quant_cfg.method == "turboldm" else "uniform")
        if quant_cfg.sensitivity_metric == "attn_mse":
            df["score"] = df["score"] * df["layer"].str.contains(r"(q_proj|k_proj|v_proj|o_proj|attn)", regex=True).astype(float).replace(0.0, 0.5)
        elif quant_cfg.sensitivity_metric == "act_max":
            stage_scale = df.groupby("stage")["score"].transform(lambda x: x / (x.max() + 1e-8))
            df["score"] = stage_scale
        return df
    if quant_cfg.sensitivity_metric == "masked_kl":
        rows = []
        for name, linear in iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex):
            pack = turboquant_vector_quantize(linear.weight.detach(), bits=quant_cfg.base_bits, block_size=quant_cfg.block_size, rotation=quant_cfg.rotation)
            qmod = TurboLDMLinear(linear, pack, None, active_stages=range(quant_cfg.stage_count))
            qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
            for stage, batches in stage_batches.items():
                score = masked_token_kl_score(model, name, qmod, batches[: max(1, quant_cfg.probe_max_examples // max(len(batches), 1))])
                rows.append({"layer": name, "stage": stage, "score": score})
        return pd.DataFrame(rows)
    raise KeyError(f"Unsupported sensitivity metric: {quant_cfg.sensitivity_metric}")


# ---------------------------------------------------------------------------
# Quantization baselines
# ---------------------------------------------------------------------------

def build_awq_importance(linear: nn.Linear, collector: ActivationCollector, layer_name: str) -> torch.Tensor:
    xs = []
    for stage in range(32):
        act = collector.get_stage_tensor(layer_name, stage)
        if act is not None:
            xs.append(act)
    if not xs:
        return torch.ones(linear.in_features, device=linear.weight.device)
    x = torch.cat(xs, dim=0).to(linear.weight.device)
    importance = x.abs().mean(dim=0).clamp_min(1e-6)
    return importance


def build_smoothquant_scale(linear: nn.Linear, collector: ActivationCollector, layer_name: str, alpha: float) -> torch.Tensor:
    xs = []
    for stage in range(32):
        act = collector.get_stage_tensor(layer_name, stage)
        if act is not None:
            xs.append(act)
    if not xs:
        return torch.ones(linear.in_features, device=linear.weight.device)
    x = torch.cat(xs, dim=0).to(linear.weight.device)
    amax = x.abs().amax(dim=0).clamp_min(1e-6)
    wmax = linear.weight.detach().abs().amax(dim=0).clamp_min(1e-6)
    scale = amax.pow(alpha) / wmax.pow(1.0 - alpha)
    return scale.clamp_min(1e-6)


def build_gptq_importance(linear: nn.Linear, collector: ActivationCollector, layer_name: str, damping: float) -> torch.Tensor:
    xs = []
    for stage in range(32):
        act = collector.get_stage_tensor(layer_name, stage)
        if act is not None:
            xs.append(act)
    if not xs:
        return torch.ones(linear.in_features, device=linear.weight.device)
    x = torch.cat(xs, dim=0).to(linear.weight.device)
    diag_h = x.pow(2).mean(dim=0) + damping
    return diag_h


def apply_uniform_quantization(model: nn.Module, quant_cfg: QuantConfig, method_name: str = "uniform") -> nn.Module:
    for name, linear in list(iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)):
        pack = uniform_group_quantize(linear.weight.detach(), bits=quant_cfg.w_bits, group_size=quant_cfg.group_size)
        qmod = UniformLinear(linear, pack, cache_dequantized=quant_cfg.cache_dequantized)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
    return model


@torch.no_grad()
def gptq_full_quantize_layer(
    weight: torch.Tensor,
    activations: torch.Tensor,
    bits: int,
    group_size: int = 128,
    damping: float = 0.01,
    block_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """Full-Hessian GPTQ on a single linear layer (Frantar et al. 2022).

    Performs iterative column-by-column quantisation with error compensation
    against the Cholesky-damped Hessian H = X^T X built from input activations.
    Returns the same packed dict as ``uniform_group_quantize`` so the caller
    can wrap a UniformLinear around it.

    Args:
        weight: (out_dim, in_dim) BF16/FP32 weight, accessed via .float().
        activations: (n_tokens, in_dim) sample of inputs to the layer.
        bits: target weight bits (3 or 4 in practice).
        group_size: per-row group size for the symmetric quantiser.
        damping: relative diagonal damping; H_diag *= (1 + damping).
        block_size: number of columns updated together inside the loop.
    """
    device = weight.device
    W = weight.detach().float().to(device).clone()
    out_dim, in_dim = W.shape
    X = activations.detach().float().to(device)
    if X.shape[-1] != in_dim:
        raise ValueError(f"activations dim {X.shape[-1]} != weight in_dim {in_dim}")
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    # Hessian H = X^T X / n; add diagonal damping; replace dead rows.
    n = max(X.shape[0], 1)
    H = (X.t() @ X) / n
    diag = torch.arange(in_dim, device=device)
    dead = (H[diag, diag] == 0)
    H[dead, dead] = 1.0
    W[:, dead] = 0
    damp = damping * H[diag, diag].mean().clamp_min(1e-8)
    H[diag, diag] += damp
    # Cholesky of inverse Hessian (upper triangular).
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    # Symmetric per-group quantiser (matches uniform_group_quantize).
    qmax = 2 ** (bits - 1) - 1
    Q = torch.zeros_like(W, dtype=torch.float32)
    pad = (group_size - in_dim % group_size) % group_size
    if pad:
        W = torch.cat([W, W.new_zeros(out_dim, pad)], dim=1)
        Hinv = torch.nn.functional.pad(Hinv, (0, pad, 0, pad))
        Hinv[in_dim:, in_dim:] = torch.eye(pad, device=device)
        in_dim_padded = in_dim + pad
        Q = torch.cat([Q, torch.zeros(out_dim, pad, device=device)], dim=1)
    else:
        in_dim_padded = in_dim
    n_groups = in_dim_padded // group_size
    scales = torch.zeros(out_dim, n_groups, device=device)
    Errs = torch.zeros_like(W)
    for i1 in range(0, in_dim_padded, block_size):
        i2 = min(i1 + block_size, in_dim_padded)
        W_block = W[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            col = i1 + j
            g = col // group_size
            if col % group_size == 0:
                start = col
                end = min(col + group_size, in_dim_padded)
                scale_g = W[:, start:end].abs().amax(dim=-1).clamp_min(1e-8) / qmax
                scales[:, g] = scale_g
            scale_g = scales[:, g]
            w = W_block[:, j]
            q = torch.round(w / scale_g).clamp(-qmax - 1, qmax)
            wq = q * scale_g
            Q[:, col] = q
            Errs[:, col] = (w - wq) / Hinv_block[j, j]
            # Compensate the rest of this block.
            if j + 1 < (i2 - i1):
                W_block[:, j + 1:] -= Errs[:, col].unsqueeze(1) * Hinv_block[j, j + 1:].unsqueeze(0)
        # Compensate later blocks once the current block is done.
        if i2 < in_dim_padded:
            W[:, i2:] -= Errs[:, i1:i2] @ Hinv[i1:i2, i2:]
    # Trim padding.
    Q = Q[:, :in_dim]
    deq = (Q.view(out_dim, n_groups, group_size if pad == 0 else group_size)[:, :in_dim // group_size + (1 if pad else 0)] * scales[:, :, None]).reshape(out_dim, -1)[:, :in_dim]
    return {
        "q": Q.to(torch.int8),
        "scales": scales,
        "pad": torch.tensor([pad]),
        "bits": torch.tensor([bits]),
        "shape": torch.tensor([out_dim, in_dim]),
        "deq": deq.to(weight.dtype),
    }


def apply_gptq_full(model: nn.Module, stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]], quant_cfg: QuantConfig) -> nn.Module:
    """Full-Hessian GPTQ over all selected linears."""
    names = [name for name, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)]
    collector = collect_activation_stats(model, names, stage_batches)
    for name, linear in list(iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)):
        # Concatenate activations across stages for this layer.
        chunks = []
        for stage in range(quant_cfg.stage_count):
            t = collector.get_stage_tensor(name, stage)
            if t is not None:
                chunks.append(t)
        if not chunks:
            # Fall back to a small random probe if no activations were captured.
            chunks = [torch.randn(64, linear.in_features, device=linear.weight.device, dtype=torch.float32)]
        acts = torch.cat(chunks, dim=0)
        try:
            pack = gptq_full_quantize_layer(
                linear.weight.detach(), acts, bits=quant_cfg.w_bits,
                group_size=quant_cfg.group_size, damping=quant_cfg.damping,
            )
        except Exception:
            # If Cholesky fails (rank-deficient activations) fall back to plain group-quant.
            pack = uniform_group_quantize(linear.weight.detach(), bits=quant_cfg.w_bits, group_size=quant_cfg.group_size)
        qmod = UniformLinear(linear, pack, cache_dequantized=quant_cfg.cache_dequantized)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
    return model


def apply_gptq_fallback(model: nn.Module, stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]], quant_cfg: QuantConfig) -> nn.Module:
    names = [name for name, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)]
    collector = collect_activation_stats(model, names, stage_batches)
    for name, linear in list(iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)):
        importance = build_gptq_importance(linear, collector, name, damping=quant_cfg.damping)
        pack = uniform_group_quantize(linear.weight.detach(), bits=quant_cfg.w_bits, group_size=quant_cfg.group_size, importance=importance)
        qmod = UniformLinear(linear, pack, cache_dequantized=quant_cfg.cache_dequantized)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
    return model


def apply_awq_fallback(model: nn.Module, stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]], quant_cfg: QuantConfig) -> nn.Module:
    names = [name for name, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)]
    collector = collect_activation_stats(model, names, stage_batches)
    for name, linear in list(iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)):
        importance = build_awq_importance(linear, collector, name)
        scale = importance.pow(quant_cfg.awq_alpha)
        quant_device = torch.device("cpu") if linear.out_features > 50000 else linear.weight.device
        scale_q = scale.to(quant_device)
        scaled_weight = linear.weight.detach().to(quant_device) * scale_q.view(1, -1)
        pack = uniform_group_quantize(scaled_weight, bits=quant_cfg.w_bits, group_size=quant_cfg.group_size, device=quant_device)
        # Undo scaling in the dequantized reference weight.
        pack["deq"] = pack["deq"] / scale_q.view(1, -1).to(pack["deq"].device)
        del scaled_weight, scale_q
        qmod = UniformLinear(linear, pack, cache_dequantized=quant_cfg.cache_dequantized)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
        del pack, qmod, importance, scale
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model


def apply_smoothquant_fallback(model: nn.Module, stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]], quant_cfg: QuantConfig) -> nn.Module:
    names = [name for name, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)]
    collector = collect_activation_stats(model, names, stage_batches)
    for name, linear in list(iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)):
        scale = build_smoothquant_scale(linear, collector, name, alpha=quant_cfg.smoothquant_alpha)
        scaled_weight = linear.weight.detach() * scale.view(1, -1)
        pack = uniform_group_quantize(scaled_weight, bits=quant_cfg.w_bits, group_size=quant_cfg.group_size)
        qmod = UniformLinear(linear, pack, input_scale=scale, cache_dequantized=quant_cfg.cache_dequantized)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
    return model


def apply_spqr_fallback(model: nn.Module, quant_cfg: QuantConfig) -> nn.Module:
    # Iterate by name, not by a materialized list — see comment in
    # `apply_turboldm` for why this matters on 8B-class models.
    layer_names = [n for n, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex,
                                                     exclude_regex=quant_cfg.exclude_regex)]
    for name in layer_names:
        parent, leaf = get_parent_module(model, name)
        linear = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]  # type: ignore[index]
        pack = uniform_group_quantize(linear.weight.detach(), bits=quant_cfg.w_bits, group_size=quant_cfg.group_size)
        residual = linear.weight.detach() - pack["deq"].to(linear.weight.device)
        flat = residual.abs().flatten()
        keep = max(1, int(flat.numel() * quant_cfg.outlier_fraction))
        threshold = torch.topk(flat, keep).values.min()
        sparse = torch.where(residual.abs() >= threshold, residual, torch.zeros_like(residual))
        qmod = SpQRLinear(linear, pack, sparse_residual=sparse, cache_dequantized=quant_cfg.cache_dequantized)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
        del pack, residual, flat, sparse, linear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model


def choose_qdiffusion_static_layer_bits(scores: pd.DataFrame, quant_cfg: QuantConfig) -> Dict[str, int]:
    avg_scores = scores.groupby("layer")["score"].mean().sort_values(ascending=False)
    topk = max(1, int(len(avg_scores) * quant_cfg.static_high_precision_fraction))
    high_layers = set(avg_scores.head(topk).index.tolist())
    out = {}
    for layer in avg_scores.index:
        out[layer] = max(quant_cfg.w_bits + 1, quant_cfg.w_bits) if layer in high_layers else quant_cfg.w_bits
    return out


def apply_qdiffusion_text(model: nn.Module, stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]], quant_cfg: QuantConfig) -> nn.Module:
    scores = compute_stage_layer_scores(model, stage_batches, quant_cfg)
    layer_bits = choose_qdiffusion_static_layer_bits(scores, quant_cfg)
    model._turboldm_stage_ratios = tuple(quant_cfg.stage_ratios)
    for name, linear in list(iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex)):
        bits = min(layer_bits.get(name, quant_cfg.w_bits), 8)
        pack = uniform_group_quantize(linear.weight.detach(), bits=bits, group_size=quant_cfg.group_size)
        qmod = UniformLinear(linear, pack, cache_dequantized=quant_cfg.cache_dequantized)
        qmod = qmod.to(linear.weight.device, dtype=linear.weight.dtype)
        replace_module(model, name, qmod)
    return model


def build_turboldm_schedule(
    scores: pd.DataFrame,
    quant_cfg: QuantConfig,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[int, float]]]:
    mode = getattr(quant_cfg, "schedule_mode", "stage")
    flat = scores.sort_values("score", ascending=False).reset_index(drop=True)
    k = max(1, int(len(flat) * quant_cfg.residual_top_fraction))
    layer_to_stages: Dict[str, List[int]] = defaultdict(list)
    entropy_thresholds: Dict[str, Dict[int, float]] = defaultdict(dict)
    if mode == "none":
        return layer_to_stages, entropy_thresholds
    if mode == "random":
        import numpy as _np
        rng = _np.random.default_rng(quant_cfg.stage_count * 101 + k)
        idx = rng.permutation(len(flat))[:k]
        chosen = flat.iloc[idx]
    elif mode == "static_layer":
        # Pick a fraction of *layers* (regardless of stage) and activate every
        # stage on them. The fraction is chosen so the total layer-stage count
        # matches ``k`` for a fair matched-budget comparison.
        S = int(quant_cfg.stage_count)
        layer_budget = max(1, k // max(1, S))
        layer_scores = scores.groupby("layer")["score"].max().sort_values(ascending=False)
        picked_layers = list(layer_scores.head(layer_budget).index)
        for layer in picked_layers:
            for stage in range(S):
                layer_to_stages[str(layer)].append(int(stage))
        return layer_to_stages, entropy_thresholds
    elif mode == "layer_once":
        # Knapsack in the "layer-once" cost model (reviewer Phase 4): charge
        # each layer once if *any* stage activates it. We rank layers by their
        # max-over-stage score and keep the top layer_budget that fit in the
        # matched-stage-pair budget k.
        S = int(quant_cfg.stage_count)
        layer_budget = max(1, k // max(1, S))
        layer_scores = scores.groupby("layer")["score"].max().sort_values(ascending=False)
        for layer in layer_scores.head(layer_budget).index:
            layer_to_stages[str(layer)].extend(range(S))
        return layer_to_stages, entropy_thresholds
    else:
        # "stage" (default) and "stage_entropy" both use the sensitivity-ranked
        # (layer, stage) head-k selection.
        chosen = flat.head(k)
    for _, row in chosen.iterrows():
        layer = str(row["layer"])
        stage = int(row["stage"])
        layer_to_stages[layer].append(stage)
    if quant_cfg.entropy_gating or mode == "stage_entropy":
        stage_threshold = scores.groupby("stage")["score"].quantile(quant_cfg.entropy_quantile).to_dict()
        for layer, stages in layer_to_stages.items():
            for stage in stages:
                entropy_thresholds[layer][stage] = float(stage_threshold.get(stage, 0.0))
    return layer_to_stages, entropy_thresholds


def apply_turboldm(model: nn.Module, stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]], quant_cfg: QuantConfig) -> nn.Module:
    model._turboldm_stage_ratios = tuple(quant_cfg.stage_ratios)
    if quant_cfg.compute_stage_scores:
        scores = compute_stage_layer_scores(model, stage_batches, quant_cfg)
    else:
        rows = []
        for name, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex, exclude_regex=quant_cfg.exclude_regex):
            for stage in range(quant_cfg.stage_count):
                rows.append({"layer": name, "stage": stage, "score": float(stage + 1)})
        scores = pd.DataFrame(rows)
    # Free the GPU scratch from the scoring pass before we start allocating quant buffers.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    layer_to_stages, entropy_thresholds = build_turboldm_schedule(scores, quant_cfg)
    # Iterate over names only. Materializing a `list(iter_named_linears(...))`
    # eagerly would keep a reference to every original Linear module, which
    # (for an 8B model) doubles GPU memory and guarantees an OOM halfway
    # through the replacement loop.
    layer_names = [n for n, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex,
                                                     exclude_regex=quant_cfg.exclude_regex)]
    for name in layer_names:
        parent, leaf = get_parent_module(model, name)
        linear = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]  # type: ignore[index]
        original_device = linear.weight.device
        original_dtype = linear.weight.dtype
        base_pack = turboquant_vector_quantize(
            linear.weight.detach(),
            bits=quant_cfg.base_bits,
            block_size=quant_cfg.block_size,
            rotation=quant_cfg.rotation,
        )
        residual_pack = None
        if name in layer_to_stages:
            residual = linear.weight.detach() - base_pack["deq"]
            residual_pack = turboquant_vector_quantize(
                residual,
                bits=max(quant_cfg.residual_bits, 1),
                block_size=quant_cfg.block_size,
                rotation=quant_cfg.rotation,
            )
            if quant_cfg.inner_product_attention_residual and re.search(r"(q_proj|k_proj|v_proj|o_proj|attn)", name):
                qjl_pack = qjl_residual_sign_quantize(residual, block_size=quant_cfg.block_size)
                residual_pack["deq"] = residual_pack["deq"] + 0.5 * qjl_pack["deq"]
                residual_pack["qjl_sign_bits"] = qjl_pack["sign_bits"]
                residual_pack["qjl_scale"] = qjl_pack["scale"]
            del residual
        qmod = TurboLDMLinear(
            linear,
            base_pack=base_pack,
            residual_pack=residual_pack,
            active_stages=layer_to_stages.get(name, []),
            entropy_thresholds=entropy_thresholds.get(name),
            cache_dequantized=quant_cfg.cache_dequantized,
        )
        qmod = qmod.to(original_device, dtype=original_dtype)
        replace_module(model, name, qmod)
        del base_pack, residual_pack, linear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model


def apply_lazy_turboldm(
    model: nn.Module,
    stage_batches: Mapping[int, Sequence[Mapping[str, torch.Tensor]]],
    quant_cfg: QuantConfig,
) -> nn.Module:
    """Like ``apply_turboldm`` but uses :class:`LazyTurboLDMLinear` — compact
    storage, on-the-fly weight reconstruction. Intended for models where the
    dequantize-and-cache convention would exceed GPU memory (e.g. 16B-class
    MoE within 32 GB VRAM).
    """
    model._turboldm_stage_ratios = tuple(quant_cfg.stage_ratios)
    if quant_cfg.compute_stage_scores:
        scores = compute_stage_layer_scores(model, stage_batches, quant_cfg)
    else:
        rows = [{"layer": n, "stage": s, "score": float(s + 1)}
                for n, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex,
                                                exclude_regex=quant_cfg.exclude_regex)
                for s in range(quant_cfg.stage_count)]
        scores = pd.DataFrame(rows)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    layer_to_stages, entropy_thresholds = build_turboldm_schedule(scores, quant_cfg)
    layer_names = [n for n, _ in iter_named_linears(model, include_regex=quant_cfg.include_regex,
                                                     exclude_regex=quant_cfg.exclude_regex)]
    for name in layer_names:
        parent, leaf = get_parent_module(model, name)
        linear = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        original_device = linear.weight.device
        # Do the quantization on CPU to keep transient memory low (we're aimed
        # at memory-tight 16B use cases).
        base_pack = turboquant_vector_quantize(
            linear.weight.detach(), bits=quant_cfg.base_bits,
            block_size=quant_cfg.block_size, rotation=quant_cfg.rotation,
            device=torch.device("cpu"),
        )
        residual_pack = None
        if name in layer_to_stages:
            residual = linear.weight.detach().to("cpu") - base_pack["deq"]
            residual_pack = turboquant_vector_quantize(
                residual, bits=max(quant_cfg.residual_bits, 1),
                block_size=quant_cfg.block_size, rotation=quant_cfg.rotation,
                device=torch.device("cpu"),
            )
            del residual
        # Drop the dequantized BF16 tensors from the packs so they don't get
        # moved to GPU with the module (LazyTurboLDMLinear does not need
        # them at construction).
        base_pack = {k: v for k, v in base_pack.items() if k != "deq"}
        if residual_pack is not None:
            residual_pack = {k: v for k, v in residual_pack.items() if k != "deq"}
        qmod = LazyTurboLDMLinear(
            linear, base_pack=base_pack, residual_pack=residual_pack,
            active_stages=layer_to_stages.get(name, []),
            entropy_thresholds=entropy_thresholds.get(name),
        )
        qmod = qmod.to(original_device)
        replace_module(model, name, qmod)
        del base_pack, residual_pack, linear
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return model


def try_bitsandbytes_int8(model_id: str, spec: ModelSpec, device_map: str = "auto") -> Optional[Tuple[nn.Module, Any]]:
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=spec.trust_remote_code)
        cls = AutoModelForCausalLM if spec.model_cls == "AutoModelForCausalLM" else AutoModel
        model = cls.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=spec.trust_remote_code, device_map=device_map)
        model.eval()
        return model, tok
    except Exception as exc:
        print(f"bitsandbytes path failed, using fallback later: {exc}")
        return None


def quantize_model_inplace(
    model: nn.Module,
    tokenizer: Any,
    spec: ModelSpec,
    quant_cfg: QuantConfig,
    run_cfg: RunConfig,
    stage_batches: Optional[Mapping[int, Sequence[Mapping[str, torch.Tensor]]]] = None,
) -> nn.Module:
    method = quant_cfg.method.lower()
    if method in ("bf16", "fp16", "fp32", "baseline", "none"):
        return model
    if method in ("bnb_int8", "llm_int8"):
        warnings.warn("For exact LLM.int8 kernels, load with bitsandbytes before calling this helper. Falling back to uniform int8.")
        qcfg = copy.deepcopy(quant_cfg)
        qcfg.w_bits = 8
        return apply_uniform_quantization(model, qcfg)
    if method == "gptq":
        if stage_batches is None:
            raise ValueError("stage_batches required for GPTQ fallback")
        return apply_gptq_fallback(model, stage_batches, quant_cfg)
    if method == "gptq_full":
        if stage_batches is None:
            raise ValueError("stage_batches required for full GPTQ")
        return apply_gptq_full(model, stage_batches, quant_cfg)
    if method == "awq":
        if stage_batches is None:
            raise ValueError("stage_batches required for AWQ fallback")
        return apply_awq_fallback(model, stage_batches, quant_cfg)
    if method == "smoothquant":
        if stage_batches is None:
            raise ValueError("stage_batches required for SmoothQuant fallback")
        return apply_smoothquant_fallback(model, stage_batches, quant_cfg)
    if method == "spqr":
        return apply_spqr_fallback(model, quant_cfg)
    if method == "qdiffusion_text":
        if stage_batches is None:
            raise ValueError("stage_batches required for QDiffusion-Text baseline")
        return apply_qdiffusion_text(model, stage_batches, quant_cfg)
    if method == "uniform":
        return apply_uniform_quantization(model, quant_cfg)
    if method == "turboldm":
        if stage_batches is None:
            raise ValueError("stage_batches required for TurboLDM")
        return apply_turboldm(model, stage_batches, quant_cfg)
    if method == "turboldm_lazy":
        if stage_batches is None:
            raise ValueError("stage_batches required for TurboLDM-lazy")
        return apply_lazy_turboldm(model, stage_batches, quant_cfg)
    raise KeyError(f"Unknown quantization method: {quant_cfg.method}")


# ---------------------------------------------------------------------------
# Memory / bits accounting
# ---------------------------------------------------------------------------

def quantized_module_storage_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinearBase):
            total += module.quant_storage_bytes()
    return total


def avg_bits_per_param(model: nn.Module) -> float:
    total_bits = 0.0
    total_params = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinearBase):
            shape = module.quant_metadata.get("shape")
            if not isinstance(shape, torch.Tensor):
                continue
            params = int(shape[0].item()) * int(shape[1].item())
            total_params += params
            total_bits += params * module.avg_bits_per_param()
    if total_params == 0:
        total = sum(p.numel() * p.element_size() for p in model.parameters())
        params = sum(p.numel() for p in model.parameters())
        return 8.0 * total / max(params, 1)
    return total_bits / max(total_params, 1)


def model_parameter_storage_bytes(model: nn.Module) -> int:
    qbytes = quantized_module_storage_bytes(model)
    if qbytes > 0:
        dense_rest = 0
        quant_params = set()
        for module in model.modules():
            if isinstance(module, QuantizedLinearBase):
                shape = module.quant_metadata.get("shape")
                if isinstance(shape, torch.Tensor):
                    quant_params.add(id(module))
        # conservative fallback: use quantized bytes if any quant modules exist
        return qbytes
    return sum(p.numel() * p.element_size() for p in model.parameters())


# ---------------------------------------------------------------------------
# Benchmark evaluators
# ---------------------------------------------------------------------------

def batched(iterable: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i+batch_size]


def maybe_reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def current_peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 3))


def evaluate_mcq_dataset(
    model: nn.Module,
    tokenizer: Any,
    spec: ModelSpec,
    dataset_name: str,
    gen_cfg: GenerateConfig,
    max_examples: Optional[int] = None,
    device: str = "cuda",
) -> ResultRecord:
    ds = load_hf_dataset(dataset_name, max_examples=max_examples)
    prompts = []
    labels = []
    for ex in ds:
        prompt, label = format_mcq_example(dataset_name, ex)
        prompts.append(prompt)
        labels.append(label)
    maybe_reset_cuda_peak()
    t0 = time.perf_counter()
    preds: List[str] = []
    for batch_prompts in batched(prompts, gen_cfg.batch_size):
        out = generate_text_batch(model, tokenizer, spec, list(batch_prompts), gen_cfg, device=device)
        preds.extend(out)
    elapsed = time.perf_counter() - t0
    correct_flags: List[int] = []
    for pred, gold in zip(preds, labels):
        n_choices = 4 if dataset_name != "piqa" else 2
        correct_flags.append(1 if parse_choice_answer(pred, num_choices=n_choices) == gold else 0)
    correct = sum(correct_flags)
    score = correct / max(len(labels), 1)
    return ResultRecord(
        model=spec.key,
        baseline="",
        dataset=dataset_name,
        split=DATASET_SPECS[dataset_name]["split"],
        num_examples=len(labels),
        score_name="accuracy",
        score_value=score,
        elapsed_sec=elapsed,
        peak_mem_gb=current_peak_mem_gb(),
        param_storage_gb=0.0,
        avg_bits_per_param=0.0,
        notes="Generative exact-match MCQ evaluation",
        extra={"predictions": preds[:10], "labels": labels[:10],
               "correct_per_example": correct_flags},
    )


def evaluate_gsm8k(
    model: nn.Module,
    tokenizer: Any,
    spec: ModelSpec,
    gen_cfg: GenerateConfig,
    max_examples: Optional[int] = None,
    device: str = "cuda",
) -> ResultRecord:
    ds = load_hf_dataset("gsm8k", max_examples=max_examples)
    prompts = []
    labels = []
    for ex in ds:
        prompt, answer = format_gsm8k_example(ex)
        prompts.append(prompt)
        labels.append(normalize_number_text(answer))
    maybe_reset_cuda_peak()
    t0 = time.perf_counter()
    preds = []
    for batch_prompts in batched(prompts, gen_cfg.batch_size):
        out = generate_text_batch(model, tokenizer, spec, list(batch_prompts), gen_cfg, device=device)
        preds.extend(out)
    elapsed = time.perf_counter() - t0
    correct_flags: List[int] = []
    for pred, gold in zip(preds, labels):
        correct_flags.append(1 if normalize_number_text(pred) == gold else 0)
    correct = sum(correct_flags)
    score = correct / max(len(labels), 1)
    return ResultRecord(
        model=spec.key,
        baseline="",
        dataset="gsm8k",
        split=DATASET_SPECS["gsm8k"]["split"],
        num_examples=len(labels),
        score_name="exact_match",
        score_value=score,
        elapsed_sec=elapsed,
        peak_mem_gb=current_peak_mem_gb(),
        param_storage_gb=0.0,
        avg_bits_per_param=0.0,
        notes="Final-number exact match",
        extra={"predictions": preds[:10], "labels": labels[:10],
               "correct_per_example": correct_flags},
    )


def _run_python_snippet_with_tests(code: str, tests: Sequence[str], timeout: int = 5) -> bool:
    import tempfile
    import textwrap

    payload = code + "\n\n" + "\n".join(tests) + "\nprint('__TURBOLDM_PASS__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(payload)
        tmp_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return "__TURBOLDM_PASS__" in proc.stdout and proc.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def clean_code_completion(text: str) -> str:
    fence_match = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.S)
    if fence_match:
        return fence_match.group(1).strip() + "\n"
    return text.strip() + "\n"


def evaluate_humaneval(
    model: nn.Module,
    tokenizer: Any,
    spec: ModelSpec,
    gen_cfg: GenerateConfig,
    max_examples: Optional[int] = None,
    device: str = "cuda",
) -> ResultRecord:
    ds = load_hf_dataset("humaneval", max_examples=max_examples)
    prompts = []
    metas = []
    for ex in ds:
        prompt, meta = format_humaneval_example(ex)
        prompts.append(prompt)
        metas.append(meta)
    maybe_reset_cuda_peak()
    t0 = time.perf_counter()
    preds = []
    for batch_prompts in batched(prompts, gen_cfg.batch_size):
        out = generate_text_batch(model, tokenizer, spec, list(batch_prompts), gen_cfg, device=device)
        preds.extend(out)
    elapsed = time.perf_counter() - t0
    passed = 0
    for pred, meta in zip(preds, metas):
        code = clean_code_completion(pred)
        if _run_python_snippet_with_tests(code, [meta["test"]]):
            passed += 1
    score = passed / max(len(metas), 1)
    return ResultRecord(
        model=spec.key,
        baseline="",
        dataset="humaneval",
        split=DATASET_SPECS["humaneval"]["split"],
        num_examples=len(metas),
        score_name="pass@1",
        score_value=score,
        elapsed_sec=elapsed,
        peak_mem_gb=current_peak_mem_gb(),
        param_storage_gb=0.0,
        avg_bits_per_param=0.0,
        notes="Single-sample HumanEval execution",
        extra={"predictions": preds[:3]},
    )


def evaluate_mbpp(
    model: nn.Module,
    tokenizer: Any,
    spec: ModelSpec,
    gen_cfg: GenerateConfig,
    max_examples: Optional[int] = None,
    device: str = "cuda",
) -> ResultRecord:
    ds = load_hf_dataset("mbpp", max_examples=max_examples)
    prompts = []
    metas = []
    for ex in ds:
        prompt, meta = format_mbpp_example(ex)
        prompts.append(prompt)
        metas.append(meta)
    maybe_reset_cuda_peak()
    t0 = time.perf_counter()
    preds = []
    for batch_prompts in batched(prompts, gen_cfg.batch_size):
        out = generate_text_batch(model, tokenizer, spec, list(batch_prompts), gen_cfg, device=device)
        preds.extend(out)
    elapsed = time.perf_counter() - t0
    passed = 0
    for pred, meta in zip(preds, metas):
        code = clean_code_completion(pred)
        if _run_python_snippet_with_tests(code, meta["test_list"]):
            passed += 1
    score = passed / max(len(metas), 1)
    return ResultRecord(
        model=spec.key,
        baseline="",
        dataset="mbpp",
        split=DATASET_SPECS["mbpp"]["split"],
        num_examples=len(metas),
        score_name="pass@1",
        score_value=score,
        elapsed_sec=elapsed,
        peak_mem_gb=current_peak_mem_gb(),
        param_storage_gb=0.0,
        avg_bits_per_param=0.0,
        notes="Single-sample MBPP execution",
        extra={"predictions": preds[:3]},
    )


def evaluate_dataset(
    model: nn.Module,
    tokenizer: Any,
    spec: ModelSpec,
    dataset_name: str,
    gen_cfg: GenerateConfig,
    max_examples: Optional[int],
    device: str = "cuda",
) -> ResultRecord:
    if dataset_name in ("mmlu", "arc_challenge", "hellaswag", "piqa"):
        return evaluate_mcq_dataset(model, tokenizer, spec, dataset_name, gen_cfg, max_examples=max_examples, device=device)
    if dataset_name == "gsm8k":
        return evaluate_gsm8k(model, tokenizer, spec, gen_cfg, max_examples=max_examples, device=device)
    if dataset_name == "humaneval":
        return evaluate_humaneval(model, tokenizer, spec, gen_cfg, max_examples=max_examples, device=device)
    if dataset_name == "mbpp":
        return evaluate_mbpp(model, tokenizer, spec, gen_cfg, max_examples=max_examples, device=device)
    raise KeyError(dataset_name)


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def default_generate_config_for_model(spec: ModelSpec, dataset_name: str) -> GenerateConfig:
    gen_length = spec.default_gen_length
    stop_strings: Tuple[str, ...] = tuple()
    if dataset_name in ("mmlu", "arc_challenge", "hellaswag", "piqa"):
        gen_length = 8
        stop_strings = ("\n",)
    elif dataset_name == "gsm8k":
        gen_length = max(gen_length, 128)
    elif dataset_name in ("humaneval", "mbpp"):
        gen_length = max(gen_length, 256)
        stop_strings = ("\n\n\n",)
    return GenerateConfig(
        steps=spec.default_steps,
        gen_length=gen_length,
        block_length=spec.default_block_length,
        temperature=0.0,
        batch_size=1 if dataset_name in ("gsm8k", "humaneval", "mbpp") else 2,
        stop_strings=stop_strings,
    )


def prepare_stage_batches_for_model(
    model: nn.Module,
    tokenizer: Any,
    spec: ModelSpec,
    quant_cfg: QuantConfig,
    run_cfg: RunConfig,
) -> Dict[int, List[Dict[str, torch.Tensor]]]:
    texts = load_calibration_texts(run_cfg, max_texts=quant_cfg.calibration_max_examples)
    return build_stage_calibration_batches(
        tokenizer=tokenizer,
        texts=texts,
        spec=spec,
        quant_cfg=quant_cfg,
        run_cfg=run_cfg,
        device=next(model.parameters()).device.type if next(model.parameters()).is_cuda else "cpu",
    )


def run_single_experiment(
    spec: ModelSpec,
    quant_cfg: QuantConfig,
    dataset_name: str,
    run_cfg: RunConfig,
    override_gen_cfg: Optional[GenerateConfig] = None,
    max_examples: Optional[int] = None,
    save_jsonl: Optional[str] = None,
    preloaded_model: Optional[nn.Module] = None,
    preloaded_tokenizer: Optional[Any] = None,
) -> ResultRecord:
    seed_everything(run_cfg.seed)
    set_matmul_precision()

    model = preloaded_model
    tokenizer = preloaded_tokenizer
    if model is None or tokenizer is None:
        if quant_cfg.method in ("bnb_int8", "llm_int8") and quant_cfg.external_first:
            maybe = try_bitsandbytes_int8(spec.model_id, spec)
            if maybe is not None:
                model, tokenizer = maybe
            else:
                model, tokenizer = load_model_and_tokenizer(spec, device=run_cfg.device, dtype=run_cfg.dtype)
        else:
            model, tokenizer = load_model_and_tokenizer(spec, device=run_cfg.device, dtype=run_cfg.dtype)

    stage_batches = None
    if quant_cfg.method not in ("bf16", "fp16", "fp32", "baseline", "none"):
        stage_batches = prepare_stage_batches_for_model(model, tokenizer, spec, quant_cfg, run_cfg)
        model = quantize_model_inplace(model, tokenizer, spec, quant_cfg, run_cfg, stage_batches=stage_batches)

    gen_cfg = override_gen_cfg or default_generate_config_for_model(spec, dataset_name)
    result = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        spec=spec,
        dataset_name=dataset_name,
        gen_cfg=gen_cfg,
        max_examples=max_examples,
        device=run_cfg.device,
    )
    result.baseline = quant_cfg.method
    result.param_storage_gb = model_parameter_storage_bytes(model) / (1024 ** 3)
    result.avg_bits_per_param = avg_bits_per_param(model)
    result.extra.update({
        "quant_cfg": asdict(quant_cfg),
        "gen_cfg": asdict(gen_cfg),
    })

    if save_jsonl is not None:
        path = Path(save_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict()) + "\n")

    # release memory
    try:
        del model
        del tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def run_matrix(
    model_keys: Sequence[str],
    quant_cfgs: Sequence[QuantConfig],
    datasets: Sequence[str],
    run_cfg: RunConfig,
    save_jsonl: Optional[str] = None,
) -> pd.DataFrame:
    records = []
    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        for qcfg in quant_cfgs:
            for dataset_name in datasets:
                max_examples = run_cfg.max_examples_per_dataset.get(dataset_name)
                print(f"\n=== model={model_key} baseline={qcfg.method} dataset={dataset_name} ===")
                rec = run_single_experiment(
                    spec=spec,
                    quant_cfg=qcfg,
                    dataset_name=dataset_name,
                    run_cfg=run_cfg,
                    max_examples=max_examples,
                    save_jsonl=save_jsonl,
                )
                records.append(rec.to_dict())
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Paper-aligned experiment presets
# ---------------------------------------------------------------------------

def baseline_presets() -> List[QuantConfig]:
    return [
        QuantConfig(method="bf16", w_bits=16),
        QuantConfig(method="bnb_int8", w_bits=8),
        QuantConfig(method="gptq", w_bits=4, group_size=128),
        QuantConfig(method="awq", w_bits=4, group_size=128),
        QuantConfig(method="smoothquant", w_bits=8, group_size=128),
        QuantConfig(method="spqr", w_bits=4, group_size=128, outlier_fraction=0.001),
        QuantConfig(method="qdiffusion_text", w_bits=3, group_size=128, stage_count=4, sensitivity_metric="hidden_mse"),
        QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=128, stage_count=4, sensitivity_metric="hidden_mse"),
    ]


def main_paper_models() -> List[str]:
    return ["llada_8b_instruct", "dream_7b_instruct"]


def scaling_models() -> List[str]:
    return ["mdlm_owt", "llada_8b_instruct", "dream_7b_instruct", "llada_moe_7b_instruct", "llada2_mini", "llada2_mini_cap", "llada2_flash", "llada2_flash_cap"]


def main_datasets() -> List[str]:
    return ["mmlu", "arc_challenge", "hellaswag", "piqa", "gsm8k"]


def code_datasets() -> List[str]:
    return ["humaneval", "mbpp"]


def stage_vs_pooled_ablation() -> List[QuantConfig]:
    return [
        QuantConfig(method="turboldm", base_bits=3, residual_bits=1, stage_count=4, sensitivity_metric="hidden_mse"),
        QuantConfig(method="qdiffusion_text", w_bits=4, stage_count=1, stage_ratios=(0.50,), sensitivity_metric="hidden_mse"),
    ]


def residual_schedule_ablation() -> List[QuantConfig]:
    return [
        QuantConfig(method="turboldm", base_bits=3, residual_bits=0, residual_top_fraction=0.0, entropy_gating=False),
        QuantConfig(method="qdiffusion_text", w_bits=4, static_high_precision_fraction=0.20),
        QuantConfig(method="turboldm", base_bits=3, residual_bits=1, residual_top_fraction=0.15, entropy_gating=False),
        QuantConfig(method="turboldm", base_bits=3, residual_bits=1, residual_top_fraction=0.15, entropy_gating=True),
    ]


def sensitivity_ablation() -> List[QuantConfig]:
    return [
        QuantConfig(method="turboldm", base_bits=3, residual_bits=1, sensitivity_metric="masked_kl"),
        QuantConfig(method="turboldm", base_bits=3, residual_bits=1, sensitivity_metric="hidden_mse"),
        QuantConfig(method="turboldm", base_bits=3, residual_bits=1, sensitivity_metric="attn_mse"),
        QuantConfig(method="turboldm", base_bits=3, residual_bits=1, sensitivity_metric="act_max"),
    ]


def block_rotation_ablation() -> List[QuantConfig]:
    out = []
    for block in (64, 128):
        out.append(QuantConfig(method="turboldm", base_bits=3, residual_bits=1, block_size=block, rotation="hadamard"))
    return out


def decoder_step_ablation(spec: ModelSpec) -> List[GenerateConfig]:
    base = default_generate_config_for_model(spec, "gsm8k")
    out = []
    for steps in (spec.default_steps, max(spec.default_steps // 2, 8), max(spec.default_steps // 4, 4)):
        cfg = copy.deepcopy(base)
        cfg.steps = steps
        out.append(cfg)
    return out


# ---------------------------------------------------------------------------
# Results aggregation and table helpers
# ---------------------------------------------------------------------------

def load_results_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if "score_value" in df.columns:
        score_col = "score_value"
    else:
        score_col = "score"
    cols = ["model", "baseline", "dataset", score_col, "elapsed_sec", "peak_mem_gb", "param_storage_gb", "avg_bits_per_param"]
    return df[cols].copy()


def pivot_main_table(df: pd.DataFrame) -> pd.DataFrame:
    score_df = df.pivot_table(index=["model", "baseline"], columns="dataset", values="score_value", aggfunc="mean")
    mem_df = df.groupby(["model", "baseline"])[["param_storage_gb", "peak_mem_gb", "avg_bits_per_param", "elapsed_sec"]].mean()
    out = mem_df.join(score_df)
    return out.reset_index()


def latex_table_from_df(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_latex(index=False, float_format=(lambda x: format(x, floatfmt)))


def plot_pareto(df: pd.DataFrame, dataset: str = "mmlu", ax: Optional[Any] = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    subset = df[df["dataset"] == dataset]
    for _, row in subset.iterrows():
        ax.scatter(row["param_storage_gb"], row["score_value"], s=60)
        ax.annotate(f"{row['model']} / {row['baseline']}", (row["param_storage_gb"], row["score_value"]), fontsize=8)
    ax.set_xlabel("Parameter storage (GB)")
    ax.set_ylabel(f"{dataset} score")
    ax.set_title(f"Pareto frontier on {dataset}")
    return ax


def stage_heatmap(scores: pd.DataFrame, top_k_layers: int = 20) -> pd.DataFrame:
    piv = scores.pivot_table(index="layer", columns="stage", values="score", aggfunc="mean").fillna(0.0)
    piv["mean_score"] = piv.mean(axis=1)
    piv = piv.sort_values("mean_score", ascending=False).head(top_k_layers).drop(columns="mean_score")
    return piv


# ---------------------------------------------------------------------------
# Optional external baseline hooks
# ---------------------------------------------------------------------------

def external_baseline_notes() -> pd.DataFrame:
    rows = [
        {"baseline": "AutoGPTQ", "repo": OFFICIAL_REPOS["autogptq"], "note": "Notebook uses a generic diagonal-Hessian GPTQ fallback when trust_remote_code models are unsupported."},
        {"baseline": "AutoAWQ", "repo": OFFICIAL_REPOS["autoawq"], "note": "Notebook uses an activation-aware scaling fallback when AutoAWQ cannot wrap diffusion architectures."},
        {"baseline": "AQLM", "repo": OFFICIAL_REPOS["aqlm"], "note": "Optional extension. Integrate after exporting HF checkpoints for a compatible architecture."},
        {"baseline": "QuIP#", "repo": OFFICIAL_REPOS["quip_sharp"], "note": "Optional extension. The notebook focuses on the main paper baselines first."},
        {"baseline": "KIVI", "repo": OFFICIAL_REPOS["kivi"], "note": "Optional cache-quantization extension; most diffusion generation here is full-sequence masked decoding."},
        {"baseline": "KVQuant", "repo": OFFICIAL_REPOS["kvquant"], "note": "Optional cache-quantization extension; not part of the main weight PTQ matrix."},
        {"baseline": "QJL", "repo": OFFICIAL_REPOS["qjl"], "note": "Used here only for an inner-product-aware residual correction."},
    ]
    return pd.DataFrame(rows)


__all__ = [
    "OFFICIAL_REPOS",
    "HF_MODELS",
    "DATASET_SPECS",
    "ModelSpec",
    "MODEL_SPECS",
    "GenerateConfig",
    "QuantConfig",
    "RunConfig",
    "ResultRecord",
    "seed_everything",
    "sync_official_repos",
    "load_model_and_tokenizer",
    "infer_mask_token_id",
    "load_hf_dataset",
    "load_calibration_texts",
    "build_stage_calibration_batches",
    "generate_text_batch",
    "compute_stage_layer_scores",
    "quantize_model_inplace",
    "run_single_experiment",
    "run_matrix",
    "baseline_presets",
    "main_paper_models",
    "scaling_models",
    "main_datasets",
    "code_datasets",
    "stage_vs_pooled_ablation",
    "residual_schedule_ablation",
    "sensitivity_ablation",
    "block_rotation_ablation",
    "decoder_step_ablation",
    "load_results_jsonl",
    "summarize_results",
    "pivot_main_table",
    "latex_table_from_df",
    "plot_pareto",
    "stage_heatmap",
    "external_baseline_notes",
]
