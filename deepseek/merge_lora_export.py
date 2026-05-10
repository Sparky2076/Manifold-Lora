"""Load DeepSeek SFT LoRA snapshot + run_meta, merge into base weights, save HF ``model_merged_hf``."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

SFT_LORA_STATE = "sft_lora_state.pt"
LEGACY_LORA_ADAPTER = "lora_adapter.pt"


def merge_loralinear_inplace(model: nn.Module) -> None:
    from lora import LoRALinear

    for full_name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        parts = full_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        child_name = parts[-1]
        W = module.base.weight.data.clone()
        if module.r and module.r > 0:
            B = module.lora_B.data
            A = module.lora_A.data
            W = W + module.scaling * (B @ A)
        new_lin = nn.Linear(module.in_features, module.out_features, bias=module.base.bias is not None)
        with torch.no_grad():
            new_lin.weight.copy_(W)
            if module.base.bias is not None:
                new_lin.bias.copy_(module.base.bias.data)
        setattr(parent, child_name, new_lin)


def resolve_lora_state_path(metrics_dir: Path) -> Path | None:
    for name in (SFT_LORA_STATE, LEGACY_LORA_ADAPTER):
        p = metrics_dir / name
        if p.is_file():
            return p
    return None


def read_run_meta(metrics_dir: Path) -> dict:
    p = metrics_dir / "run_meta.json"
    return json.loads(p.read_text(encoding="utf-8"))


def merge_and_export_hf(
    metrics_dir: Path,
    merged_hf_dir: Path | None = None,
    *,
    model_name: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: str | None = None,
) -> Path:
    """Merge LoRA into base and ``save_pretrained`` to *merged_hf_dir*. Returns resolved merged dir."""
    metrics_dir = metrics_dir.resolve()
    meta_path = metrics_dir / "run_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    lora_path = resolve_lora_state_path(metrics_dir)
    if lora_path is None:
        raise FileNotFoundError(
            f"No {SFT_LORA_STATE} or {LEGACY_LORA_ADAPTER} under {metrics_dir}",
        )

    meta = read_run_meta(metrics_dir)
    if meta.get("lora_type", "default") != "default":
        raise ValueError("merge_and_export_hf only supports lora_type=default (additive LoRA)")

    from deepseek.models_sft import ModelLoadConfig, load_model_and_tokenizer

    mn = model_name or meta.get("model_name") or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(mn, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
    )
    model.to(device)

    from lora import LoRAConfig, apply_lora, mark_only_lora_as_trainable

    _default_targets = "q_proj,k_proj,v_proj,o_proj,out_proj,q_lin,k_lin,v_lin,out_lin,c_attn,c_proj"
    _raw_targets = meta.get("lora_targets") or _default_targets
    targets = [t.strip() for t in str(_raw_targets).split(",") if t.strip()]
    lora_cfg = LoRAConfig(
        r=int(meta.get("lora_r", 16)),
        alpha=float(meta.get("lora_alpha", 32.0)),
        dropout=float(meta.get("lora_dropout", 0.05)),
        target_modules=targets,
        attention_only=True,
    )
    apply_lora(device, model, lora_cfg, verbose=False)
    mark_only_lora_as_trainable(model)

    ad = torch.load(lora_path, map_location=device)
    model.load_state_dict(ad, strict=False)
    model.eval()
    merge_loralinear_inplace(model)

    out_dir = (merged_hf_dir or (metrics_dir / "model_merged_hf")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def merged_hf_ready(merged_dir: Path) -> bool:
    return (merged_dir / "config.json").is_file()
