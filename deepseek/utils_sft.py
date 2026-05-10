from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


def _resolve_preset(preset: str) -> tuple[str, str]:
    p = (preset or "").strip().lower()
    if p in ("alpaca_train_1k", "alpaca_1k"):
        return "tatsu-lab/alpaca", "train[:1000]"
    if p in ("alpaca_train_500", "alpaca_500"):
        return "tatsu-lab/alpaca", "train[:500]"
    if p in ("testing_alpaca_small", "alpaca_small"):
        return "HuggingFaceH4/testing_alpaca_small", "train"
    raise ValueError(f"Unknown sft preset: {preset}")


def _normalize_columns(ds: Dataset) -> Dataset:
    cols = set(ds.column_names)

    def _pick(row, keys):
        for k in keys:
            if k in row and row[k] is not None:
                return str(row[k]).strip()
        return ""

    def _map(row):
        inst = _pick(row, ["instruction", "prompt", "question"])
        inp = _pick(row, ["input", "context"])
        out = _pick(row, ["output", "response", "answer", "completion"])
        if not inst and "text" in row:
            text = str(row.get("text", "")).strip()
            return {"text": text}
        parts = []
        if inst:
            parts.append(f"### Instruction:\n{inst}")
        if inp:
            parts.append(f"### Input:\n{inp}")
        if out:
            parts.append(f"### Response:\n{out}")
        return {"text": "\n\n".join(parts).strip()}

    if "text" in cols:
        return ds.map(lambda r: {"text": str(r.get("text", "")).strip()}, remove_columns=list(cols))
    return ds.map(_map, remove_columns=list(cols))


def load_sft_dataset(dataset_name: str = "", split: str = "", preset: str = "alpaca_train_1k") -> Dataset:
    if dataset_name.strip():
        name, sp = dataset_name.strip(), (split.strip() or "train")
    else:
        name, sp = _resolve_preset(preset)
    ds = load_dataset(name, split=sp)
    return _normalize_columns(ds)


def split_train_val(ds: Dataset, val_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if val_ratio <= 0:
        return ds, ds.select(range(min(len(ds), 1)))
    n = len(ds)
    val_n = max(1, int(math.ceil(n * val_ratio)))
    out = ds.train_test_split(test_size=val_n, seed=seed, shuffle=True)
    return out["train"], out["test"]


@dataclass
class SFTDataConfig:
    max_length: int = 512
    batch_size: int = 2
    num_workers: int = 0


class CausalCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        texts = [s["text"] for s in samples]
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc


def build_dataloaders(tokenizer, train_ds: Dataset, val_ds: Dataset, cfg: SFTDataConfig):
    collate = CausalCollator(tokenizer, cfg.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
    )
    return train_loader, val_loader
