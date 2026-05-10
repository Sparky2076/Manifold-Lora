from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelLoadConfig:
    model_name: str
    trust_remote_code: bool = False
    torch_dtype: str | None = None


def _dtype_from_name(name: str | None):
    if name is None:
        return None
    n = str(name).lower()
    if n in ("float16", "fp16", "half"):
        return torch.float16
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    if n in ("float32", "fp32"):
        return torch.float32
    return None


def load_model_and_tokenizer(cfg: ModelLoadConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=cfg.trust_remote_code, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=_dtype_from_name(cfg.torch_dtype),
    )
    return model, tokenizer
