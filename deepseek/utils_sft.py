# utils_sft.py — 小体量指令数据集 + SFT 标签构造（独立于根目录 utils.py 的 GLUE 分类）
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


@dataclass
class SFTDataConfig:
    """Hub 小指令集预设见 deepseek/README.md 与 docs/DEEPSEEK_FINETUNE_PLAN.md"""

    dataset_id: str = "HuggingFaceH4/testing_alpaca_small"
    dataset_config: Optional[str] = None
    split_train: str = "train"
    val_ratio: float = 0.1
    max_samples: Optional[int] = None
    max_length: int = 512
    num_workers: int = 0
    seed: int = 42


SFT_DATASET_PRESETS: Dict[str, Tuple[str, Optional[str], Optional[str]]] = {
    "testing_alpaca_small": ("HuggingFaceH4/testing_alpaca_small", None, None),
    "alpaca_gpt4_500": ("levulinh/alpaca-gpt4-500", None, None),
    "alpaca_train_500": ("tatsu-lab/alpaca", None, "train[:500]"),
    "alpaca_train_1k": ("tatsu-lab/alpaca", None, "train[:1000]"),
}


def _alpaca_prompt_response(example: Dict[str, Any]) -> Tuple[str, str]:
    inst = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    out = (example.get("output") or example.get("response") or "").strip()
    if inp:
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
    return prompt, out


def _dolly_prompt_response(example: Dict[str, Any]) -> Tuple[str, str]:
    inst = (example.get("instruction") or "").strip()
    ctx = (example.get("context") or "").strip()
    resp = (example.get("response") or "").strip()
    if ctx:
        prompt = f"### Instruction:\n{inst}\n\n### Context:\n{ctx}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
    return prompt, resp


def _detect_format(ds_sample: Dict[str, Any]) -> str:
    keys = set(ds_sample.keys())
    if "instruction" in keys and ("output" in keys or "response" in keys):
        return "alpaca_like"
    if "instruction" in keys and "context" in keys and "response" in keys:
        return "dolly"
    if "text" in keys and len(keys) <= 4:
        return "text_only"
    return "alpaca_like"


def _encode_sft(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Dict[str, List[int]]:
    p_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        r_ids = r_ids + [tokenizer.eos_token_id]

    input_ids = p_ids + r_ids
    labels = [-100] * len(p_ids) + r_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "labels": labels}


def build_sft_dataloaders(
    tokenizer,
    cfg: SFTDataConfig,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    split = cfg.split_train
    if cfg.dataset_config:
        ds = load_dataset(
            cfg.dataset_id,
            cfg.dataset_config,
            split=split,
            trust_remote_code=True,
        )
    else:
        ds = load_dataset(
            cfg.dataset_id,
            split=split,
            trust_remote_code=True,
        )

    if cfg.max_samples is not None and cfg.max_samples > 0:
        n = min(cfg.max_samples, len(ds))
        ds = ds.select(range(n))

    sample = ds[0]
    fmt = _detect_format(sample)

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        if fmt == "dolly":
            prompt, response = _dolly_prompt_response(example)
        elif fmt == "text_only" and example.get("text"):
            full = example["text"].strip()
            mid = len(full) // 2
            prompt, response = full[:mid], full[mid:]
        else:
            prompt, response = _alpaca_prompt_response(example)
        enc = _encode_sft(tokenizer, prompt, response, cfg.max_length)
        return enc

    ds = ds.map(preprocess, remove_columns=ds.column_names, desc="SFT tokenize")

    n = len(ds)
    if n <= 1:
        train_ds, eval_ds = ds, ds
    else:
        test_n = max(1, min(n - 1, ceil(n * cfg.val_ratio)))
        split_ds = ds.train_test_split(test_size=test_n, seed=cfg.seed)
        train_ds = split_ds["train"]
        eval_ds = split_ds["test"]

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for x in batch:
            ids = x["input_ids"]
            lab = x["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(lab + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader
