# models.py — 序列分类模型加载（distilbert 子包，独立于 deepseek）
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


@dataclass
class ModelLoadConfig:
    model_name: str = "distilbert/distilbert-base-uncased"
    num_labels: int = 2
    trust_remote_code: bool = False
    torch_dtype: Optional[str] = None
    device_map: Optional[str] = None
    use_fast_tokenizer: bool = True


def _parse_dtype(torch_dtype: Optional[str]):
    if torch_dtype is None:
        return None
    s = torch_dtype.lower()
    if s == "float16" or s == "fp16":
        return torch.float16
    if s == "bfloat16" or s == "bf16":
        return torch.bfloat16
    if s == "float32" or s == "fp32":
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {torch_dtype}")


def load_model_and_tokenizer(cfg: ModelLoadConfig) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """
    Load HuggingFace model + tokenizer for sequence classification.
    """
    dtype = _parse_dtype(cfg.torch_dtype)

    hf_config = AutoConfig.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    if hasattr(hf_config, "num_labels"):
        hf_config.num_labels = cfg.num_labels
    else:
        setattr(hf_config, "num_labels", cfg.num_labels)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=cfg.use_fast_tokenizer,
            trust_remote_code=cfg.trust_remote_code,
        )
    except (TypeError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            use_fast=False,
            trust_remote_code=cfg.trust_remote_code,
        )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": ""})

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=hf_config,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
        device_map=cfg.device_map,
    )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
