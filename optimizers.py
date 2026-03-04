# optimizers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass
class AdamWConfig:
    lr: float = 2e-4
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01


def get_optimizer(
    params: Iterable[torch.nn.Parameter],
    cfg: AdamWConfig,
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.AdamW optimizer.
    Kept as a single entry point for extensibility.
    """
    return torch.optim.AdamW(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )