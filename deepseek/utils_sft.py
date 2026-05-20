from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

SFTFormat = Literal["chat", "alpaca"]


def _resolve_preset(preset: str) -> tuple[str, str]:
    p = (preset or "").strip().lower()
    if p in ("alpaca_train_1k", "alpaca_1k"):
        return "tatsu-lab/alpaca", "train[:1000]"
    if p in ("alpaca_train_500", "alpaca_500"):
        return "tatsu-lab/alpaca", "train[:500]"
    if p in ("alpaca_train_full", "alpaca_full", "alpaca_52k"):
        return "tatsu-lab/alpaca", "train"
    if p in ("testing_alpaca_small", "alpaca_small"):
        return "HuggingFaceH4/testing_alpaca_small", "train"
    raise ValueError(f"Unknown sft preset: {preset}")


def _pick_nonempty(row: dict, keys: list[str]) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            s = str(row[k]).strip()
            if s:
                return s
    return ""


def alpaca_row_to_user_content(instruction: str, input_text: str) -> str:
    parts: list[str] = []
    if instruction:
        parts.append(instruction)
    if input_text:
        parts.append(input_text)
    return "\n".join(parts).strip()


def alpaca_legacy_prompt_and_full(instruction: str, input_text: str, output: str) -> tuple[str, str]:
    parts: list[str] = []
    if instruction:
        parts.append(f"### Instruction:\n{instruction}")
    if input_text:
        parts.append(f"### Input:\n{input_text}")
    head = "\n\n".join(parts).strip()
    out = output.strip()
    if head:
        prompt = f"{head}\n\n### Response:\n"
    else:
        prompt = "### Response:\n"
    return prompt, f"{prompt}{out}"


def _normalize_dataset_rows(ds: Dataset) -> Dataset:
    cols = list(ds.column_names)

    def _map(row):
        if "text" in cols:
            t = str(row.get("text", "")).strip()
            if t:
                return {
                    "kind": "text",
                    "text": t,
                    "instruction": "",
                    "input": "",
                    "output": "",
                }
        inst = _pick_nonempty(row, ["instruction", "prompt", "question"])
        inp = _pick_nonempty(row, ["input", "context"])
        out = _pick_nonempty(row, ["output", "response", "answer", "completion"])
        if not out:
            return {"kind": "skip", "text": "", "instruction": "", "input": "", "output": ""}
        return {
            "kind": "structured",
            "text": "",
            "instruction": inst,
            "input": inp,
            "output": out,
        }

    out = ds.map(_map, remove_columns=cols)
    return out.filter(lambda r: r["kind"] != "skip")


def load_sft_dataset(dataset_name: str = "", split: str = "", preset: str = "alpaca_train_1k") -> Dataset:
    if dataset_name.strip():
        name, sp = dataset_name.strip(), (split.strip() or "train")
    else:
        name, sp = _resolve_preset(preset)
    ds = load_dataset(name, split=sp)
    return _normalize_dataset_rows(ds)


def split_train_val(ds: Dataset, val_ratio: float, seed: int) -> tuple[Dataset, Dataset]:
    if val_ratio <= 0:
        return ds, ds.select(range(min(len(ds), 1)))
    n = len(ds)
    val_n = max(1, int(math.ceil(n * val_ratio)))
    out = ds.train_test_split(test_size=val_n, seed=seed, shuffle=True)
    return out["train"], out["test"]


def _common_prompt_len(prompt_ids: list[int], full_ids: list[int]) -> int:
    if len(full_ids) >= len(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    n = min(len(prompt_ids), len(full_ids))
    i = 0
    while i < n and prompt_ids[i] == full_ids[i]:
        i += 1
    return i


def _truncate_from_left(
    input_ids: list[int],
    labels: list[int],
    prompt_len: int,
    max_length: int,
) -> tuple[list[int], list[int], int]:
    if len(input_ids) <= max_length:
        return input_ids, labels, prompt_len
    drop = len(input_ids) - max_length
    input_ids = input_ids[drop:]
    labels = labels[drop:]
    prompt_len = max(0, prompt_len - drop)
    return input_ids, labels, prompt_len


@dataclass
class SFTDataConfig:
    max_length: int = 512
    batch_size: int = 2
    num_workers: int = 0


class MaskedSFTCollator:
    """Prompt masking for instruction tuning; chat template or legacy Alpaca strings."""

    def __init__(self, tokenizer, max_length: int, sft_format: SFTFormat):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sft_format = sft_format
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _encode_structured_chat(self, row: dict) -> tuple[list[int], list[int]]:
        user_content = alpaca_row_to_user_content(row["instruction"], row["input"]) or "[empty]"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["output"].strip()},
        ]
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                "Tokenizer has no chat_template; use --sft_format alpaca or a chat-capable tokenizer."
            )
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_tensors=None
        )
        pl = _common_prompt_len(list(prompt_ids), list(full_ids))
        labels = list(full_ids)
        for i in range(pl):
            labels[i] = -100
        ids, labels2, _ = _truncate_from_left(list(full_ids), labels, pl, self.max_length)
        return ids, labels2

    def _encode_structured_alpaca(self, row: dict) -> tuple[list[int], list[int]]:
        prompt_txt, full_txt = alpaca_legacy_prompt_and_full(
            row["instruction"], row["input"], row["output"]
        )
        prompt_ids = self.tokenizer(prompt_txt, add_special_tokens=True, truncation=False)["input_ids"]
        full_ids = self.tokenizer(full_txt, add_special_tokens=True, truncation=False)["input_ids"]
        pl = _common_prompt_len(list(prompt_ids), list(full_ids))
        labels = list(full_ids)
        for i in range(pl):
            labels[i] = -100
        ids, labels2, _ = _truncate_from_left(list(full_ids), labels, pl, self.max_length)
        return ids, labels2

    def _encode_text_only(self, row: dict) -> tuple[list[int], list[int]]:
        enc = self.tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return enc["input_ids"], list(enc["input_ids"])

    def _encode_one(self, row: dict) -> tuple[list[int], list[int]]:
        if row["kind"] == "text":
            return self._encode_text_only(row)
        if self.sft_format == "chat":
            return self._encode_structured_chat(row)
        return self._encode_structured_alpaca(row)

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        seqs: list[list[int]] = []
        labs: list[list[int]] = []
        for row in samples:
            ids, labels = self._encode_one(row)
            seqs.append(ids)
            labs.append(labels)
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        max_len = min(max(len(s) for s in seqs) if seqs else 1, self.max_length)

        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        labels_out: list[list[int]] = []
        for ids, lb in zip(seqs, labs):
            if len(ids) > max_len:
                ids, lb = ids[-max_len:], lb[-max_len:]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention.append([1] * len(ids) + [0] * pad_n)
            labels_out.append(lb + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels_out, dtype=torch.long),
        }


def build_dataloaders(
    tokenizer,
    train_ds: Dataset,
    val_ds: Dataset,
    cfg: SFTDataConfig,
    *,
    sft_format: SFTFormat = "chat",
):
    fmt: SFTFormat = "chat" if str(sft_format).lower() == "chat" else "alpaca"
    collate = MaskedSFTCollator(tokenizer, cfg.max_length, fmt)
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


class CausalCollator:
    """Legacy full-sequence CE on ``text`` field (kept for external callers)."""

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
