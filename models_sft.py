# models_sft.py — DeepSeek / 因果 LM 的 SFT 加载（独立于 models.py 的分类加载）
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class CausalLMConfig:
    model_name: str
    trust_remote_code: bool = False
    torch_dtype: Optional[str] = None  # "float16", "bfloat16", "float32"
    device_map: Optional[str] = None
    use_fast_tokenizer: bool = True


def _parse_dtype(torch_dtype: Optional[str]):
    if torch_dtype is None:
        return None
    s = torch_dtype.lower()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {torch_dtype}")


def load_causal_lm_and_tokenizer(cfg: CausalLMConfig) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """
    加载 HuggingFace CausalLM + Tokenizer，用于指令微调（SFT）。
    与 models.py 中的 SequenceClassification 路径分离，避免改动原分类逻辑。
    """
    dtype = _parse_dtype(cfg.torch_dtype)

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

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
        device_map=cfg.device_map,
    )

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
