# lora.py
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
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    bias: str = "none"  # {"none","all","lora_only"} (kept for extension)
    target_modules: Optional[List[str]] = None
    # If True, will only apply LoRA to modules whose names match attention-like patterns.
    attention_only: bool = True


class LoRALinear(nn.Module):
    """
    A manual LoRA wrapper for nn.Linear:
        y = x W^T + b + scale * ( (dropout(x) A^T) B^T )
    where A: (r, in_features), B: (out_features, r)
    We keep the base Linear frozen by default, and train only A,B.
    """
    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float,
        device
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects a nn.Linear as base module.")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()
        self.device = device
        # LoRA parameters
        if self.r > 0:
            # DeepSeek 等模型通常以 bfloat16 形式加载权重；
            # 若基座权重为 bf16，则将 LoRA 参数也初始化为 bf16，否则使用 float32。
            base_dtype = getattr(self.base.weight, "dtype", torch.float32)
            if base_dtype == torch.bfloat16:
                lora_dtype = torch.bfloat16
            else:
                lora_dtype = torch.float32

            # A: (r, in_features), B: (out_features, r)
            self.lora_A = nn.Parameter(
                torch.zeros(self.r, self.in_features, device=self.device, dtype=lora_dtype)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(self.out_features, self.r, device=self.device, dtype=lora_dtype)
            )
            # Initialization: A ~ Kaiming uniform, B = 0 (common practice)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # Freeze base by default; caller can override if needed.
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path
        out = self.base(x)

        # LoRA path
        if self.r > 0:
            x_d = self.dropout(x)
            # Cast LoRA params to input dtype (fixes Half/Float mismatch under autocast).
            dtype = x_d.dtype
            lora_A = self.lora_A.to(dtype)
            lora_B = self.lora_B.to(dtype)
            # (batch, *, in) @ (in, r) -> (batch, *, r) then @ (r, out) -> (batch, *, out)
            # Using F.linear for efficiency: F.linear(input, weight, bias)
            # F.linear: input @ weight^T
            z = F.linear(x_d, lora_A)              # weight: (r, in) => out dim r
            lora_out = F.linear(z, lora_B)         # weight: (out, r) => out dim out
            out = out + self.scaling * lora_out

        return out


def _iter_named_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    # Safer wrapper: yields (name, module) for all modules including nested.
    for name, module in model.named_modules():
        yield name, module


def _looks_like_attention_module(module_name: str) -> bool:
    """
    Heuristic filter for attention-related module names across architectures:
    - BERT-like: attention, self_attn, q_lin/k_lin/v_lin/out_lin
    - GPT-like: attn, c_attn, q_proj/k_proj/v_proj/o_proj
    - Generic: query/key/value, q_proj/k_proj/v_proj, out_proj
    """
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
    """
    If target_modules is provided, match by substring or regex-like patterns.
    We keep it simple: treat each entry as a substring.
    (You can extend to regex easily.)
    """
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
    """
    Replace selected nn.Linear modules with LoRALinear, using automatic scanning.

    Key requirement:
    - Automatically scan attention layers and match target_modules.

    Returns:
        A dict mapping replaced module full-names -> new LoRALinear modules.
    """
    replaced: Dict[str, nn.Module] = {}

    # Collect (parent_module, child_name, full_name, child_module)
    candidates: List[Tuple[nn.Module, str, str, nn.Module]] = []
    for full_name, module in _iter_named_modules(model):
        if full_name == "":
            continue
        # We need the parent to perform replacement. We'll find it by splitting.
        if isinstance(module, nn.Linear):
            if cfg.attention_only and (not _looks_like_attention_module(full_name)):
                continue
            if not _match_target_modules(full_name, cfg.target_modules):
                continue

            # Locate parent
            parts = full_name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            child_name = parts[-1]
            candidates.append((parent, child_name, full_name, module))

    # Replace
    for parent, child_name, full_name, child in candidates:
        wrapped = LoRALinear(
            base=child,
            r=cfg.r,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            device=device
        )
        setattr(parent, child_name, wrapped)
        replaced[full_name] = wrapped

    if verbose:
        print(f"[LoRA] Replaced {len(replaced)} Linear modules with LoRALinear.")
        if len(replaced) > 0:
            # Print a few examples for sanity
            sample = list(replaced.keys())[:20]
            print("[LoRA] Sample replaced modules:")
            for s in sample:
                print("  -", s)

    return replaced


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """
    Freeze everything, then unfreeze only LoRA parameters (lora_A, lora_B).
    """
    for p in model.parameters():
        p.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, LoRALinear):
            for n, p in m.named_parameters():
                if n in ("lora_A", "lora_B") and p is not None:
                    p.requires_grad_(True)


def lora_trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Convenience helper: return only parameters that require grad.
    """
    return [p for p in model.parameters() if p.requires_grad]