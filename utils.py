# utils.py
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import DataCollatorWithPadding


def build_dataloaders(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    text_field: str,
    max_length: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build training and evaluation dataloaders.

    Default NLP dataset: GLUE/SST2.
    Metric: accuracy.

    Notes:
      - Removes the raw text column (e.g., 'sentence') after tokenization; otherwise
        DataCollatorWithPadding will try to tensorize strings.
    """
    ds = load_dataset(dataset_name, dataset_config)

    # Cache original columns BEFORE mapping so we can remove them after tokenization.
    train_columns = list(ds["train"].column_names)

    def preprocess(examples):
        enc = tokenizer(
            examples[text_field],
            truncation=True,
            max_length=max_length,
        )
        if "label" in examples:
            enc["labels"] = examples["label"]
        return enc

    ds = ds.map(
        preprocess,
        batched=True,
        remove_columns=train_columns,
        desc="Tokenizing",
    )
    ds.set_format(type="torch")

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    train_loader = DataLoader(
        ds["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    eval_split = "validation" if "validation" in ds else ("test" if "test" in ds else None)
    if eval_split is None:
        raise ValueError("Dataset has no validation/test split available.")

    eval_loader = DataLoader(
        ds[eval_split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, eval_loader
