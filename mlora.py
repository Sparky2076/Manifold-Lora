
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    """Configuration for multiplicative LoRA (mLoRA).

    This keeps the public API identical to lora.py so existing code can swap
    `from lora import ...` -> `from mlora import ...` without other changes.

    Notation:
        - Base (frozen) weight: W_*  (shape [out, in])
        - Low-rank factorization: Δ = B A  with rank r
        - Multiplicative adapter: W = W_* ⊙ W_H

    Practical parameterization (important for compatibility/stability):
        We set W_H := 1 + s * Δ, so that at initialization W_H ≈ 1 and hence
        W ≈ W_*. This avoids collapsing the base weights to ~0 at start.
  """
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    bias: str = "none"  # kept for extension
    target_modules: Optional[List[str]] = None
    attention_only: bool = True


class MLoRALinear(nn.Module):
    """A manual multiplicative-LoRA wrapper for nn.Linear.

    Base LoRA (additive) uses:
        W = W_* + s * (B A)
    Here we implement the multiplicative variant requested:
        W = W_* ⊙ W_H

    We use the stable parameterization:
        W_H = 1 + s * (B A)
    so that the wrapped layer starts identical to the base layer if B=0.

    Forward:
        y = x ( (W_* ⊙ (1 + s*BA))^T ) + b

    Notes:
      - The base linear parameters are frozen by default; only A,B train.
      - Dropout (if enabled) is applied to the adapter Δ entries.
  """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        device,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("MLoRALinear expects a nn.Linear as base module.")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0

        # Dropout on adapter entries (Δ). Identity if dropout=0.
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        # Prefer allocating LoRA params on the base module's device (important when using device_map).
        base_device = getattr(getattr(base, "weight", None), "device", None)
        if base_device is None or str(base_device) == "meta":
            base_device = device

        if self.r > 0:
            # A: (r, in_features), B: (out_features, r)
            self.lora_A = nn.Parameter(torch.zeros(self.r, self.in_features, device=base_device))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=base_device))

            # Initialization: A ~ Kaiming uniform, B = 0 (so Δ=0 and W_H=1 at init).
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # Freeze base by default; caller can override if desired.
        for p in self.base.parameters():
            p.requires_grad_(False)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling}"
        )

    def _adapter_gate(self) -> Optional[torch.Tensor]:
        """Compute W_H = 1 + scaling * (B @ A), optionally with dropout.

        Returns:
            gate of shape (out_features, in_features), or None if r==0.
        """
        if self.r <= 0:
            return None

        # (out, r) @ (r, in) -> (out, in)
        delta = self.lora_B @ self.lora_A
        delta = self.dropout(delta)
        gate = 1.0 + self.scaling * delta
        return gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r <= 0:
            return self.base(x)

        gate = self._adapter_gate()
        # Effective weight: W_eff = W_* ⊙ W_H
        # base.weight shape: (out, in)
        w_eff = self.base.weight * gate

        # Use F.linear for correct broadcasting and speed.
        return F.linear(x, w_eff, self.base.bias)


def _iter_named_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        yield name, module


def _looks_like_attention_module(module_name: str) -> bool:
    n = module_name.lower()
    patterns = [
        r"attn", r"attention", r"self_attn",
        r"\bq\b", r"\bk\b", r"\bv\b",
        r"q_proj", r"k_proj", r"v_proj", r"o_proj", r"out_proj",
        r"query", r"key", r"value",
        r"q_lin", r"k_lin", r"v_lin", r"out_lin",
        r"c_attn", r"c_proj",
    ]
    return any(re.search(p, n) for p in patterns)


def _match_target_modules(module_name: str, target_modules: Optional[List[str]]) -> bool:
    if not target_modules:
        return True
    ln = module_name.lower()
    for t in target_modules:
        if t.lower() in ln:
            return True
    return False


def apply_lora(
    device,
    model: nn.Module,
    cfg: LoRAConfig,
    verbose: bool = True,
) -> Dict[str, nn.Module]:
    """Replace selected nn.Linear modules with MLoRALinear using automatic scanning.

    This is API-compatible with lora.apply_lora, but swaps in MLoRALinear.

    Returns:
        dict mapping replaced module full-names -> new MLoRALinear modules.
    """
    replaced: Dict[str, nn.Module] = {}

    candidates: List[Tuple[nn.Module, str, str, nn.Module]] = []
    for full_name, module in _iter_named_modules(model):
        if full_name == "":
            continue
        if isinstance(module, nn.Linear):
            if cfg.attention_only and (not _looks_like_attention_module(full_name)):
                continue
            if not _match_target_modules(full_name, cfg.target_modules):
                continue

            parts = full_name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            child_name = parts[-1]
            candidates.append((parent, child_name, full_name, module))

    for parent, child_name, full_name, child in candidates:
        wrapped = MLoRALinear(
            base=child,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            device=device,
        )
        setattr(parent, child_name, wrapped)
        replaced[full_name] = wrapped

    if verbose:
        print(f"[mLoRA] Replaced {len(replaced)} Linear modules with MLoRALinear.")
        if len(replaced) > 0:
            sample = list(replaced.keys())[:20]
            print("[mLoRA] Sample replaced modules:")
            for s in sample:
                print("  -", s)

    return replaced


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze everything, then unfreeze only LoRA parameters (lora_A, lora_B)."""
    for p in model.parameters():
        p.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, MLoRALinear):
            for n, p in m.named_parameters():
                if n in ("lora_A", "lora_B") and p is not None:
                    p.requires_grad_(True)


def lora_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return only parameters that require grad."""
    return [p for p in model.parameters() if p.requires_grad]
