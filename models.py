# models.py
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
    torch_dtype: Optional[str] = None  # "float16", "bfloat16", or None
    device_map: Optional[str] = None   # "auto" or None


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


def load_model_and_tokenizer(cfg: ModelLoadConfig):
    """
    Load HF model + tokenizer.
    Supports:
      - distilbert-base-uncased (and other BERT-like)
      - deepseek 1.5b variants (pass correct repo id via --model_name)

    For large models, you may want:
      --device_map auto --torch_dtype bfloat16 --trust_remote_code
    """
    dtype = _parse_dtype(cfg.torch_dtype)

    hf_config = AutoConfig.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    hf_config.num_labels = cfg.num_labels

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=True,
        trust_remote_code=cfg.trust_remote_code,
    )

    # Some tokenizers may not have pad token (common for decoder-only).
    if tokenizer.pad_token is None:
        # A safe default: use eos_token as pad_token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # As last resort
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=hf_config,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
        device_map=cfg.device_map,
    )

    # If we added tokens, resize embeddings.
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer