"""Merge LoRA weights from an SFT run and run BBH via lm-eval (EleutherAI harness).

Expects ``run_meta.json`` and ``lora_adapter.pt`` under ``--metrics_dir`` (written by
``deepseek.main_sft``). Only ``lora_type=default`` (additive LoRA) is supported for merge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _merge_loralinear_inplace(model: nn.Module) -> None:
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


def _bbh_scores_from_lm_eval(results: dict) -> tuple[float, dict[str, float]]:
    """Return (mean_acc, per_subtask_acc) from lm_eval ``results`` dict."""
    rmap = results.get("results") or {}
    sub_tasks = {
        k: v
        for k, v in rmap.items()
        if k.startswith("bbh_") and isinstance(v, dict) and "acc,none" in v
    }
    accs: dict[str, float] = {}
    if sub_tasks:
        for k, v in sub_tasks.items():
            try:
                accs[k] = float(v["acc,none"])
            except (TypeError, ValueError):
                continue
    if accs:
        return sum(accs.values()) / len(accs), accs
    v = rmap.get("bbh")
    if isinstance(v, dict) and "acc,none" in v:
        try:
            x = float(v["acc,none"])
            return x, {"bbh": x}
        except (TypeError, ValueError):
            pass
    return float("nan"), {}


def main() -> int:
    p = argparse.ArgumentParser(description="BBH eval for a DeepSeek LoRA SFT run (lm-eval harness).")
    p.add_argument("--metrics_dir", type=Path, required=True, help="Run directory with run_meta.json + lora_adapter.pt")
    p.add_argument("--output_json", type=Path, default=None, help="Default: <metrics_dir>/bbh_eval.json")
    p.add_argument("--model_name", type=str, default=None, help="Override base HF id (else from run_meta or default)")
    p.add_argument("--merged_hf_dir", type=Path, default=None, help="Merged HF export dir (default: <metrics_dir>/model_merged_hf)")
    p.add_argument("--tasks", type=str, default="bbh", help="Comma-separated lm-eval task names")
    p.add_argument("--num_fewshot", type=int, default=3)
    p.add_argument("--batch_size", type=str, default="auto")
    p.add_argument("--limit", type=int, default=0, help="If >0, passed to lm_eval as doc limit per task (smoke tests).")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--dry_merge_only", action="store_true", help="Only merge+save HF weights; skip lm_eval.")
    args = p.parse_args()

    metrics_dir = args.metrics_dir.resolve()
    meta_path = metrics_dir / "run_meta.json"
    adapter_path = metrics_dir / "lora_adapter.pt"
    if not meta_path.is_file():
        print(f"[eval_bbh] missing {meta_path}", file=sys.stderr)
        return 2
    if not adapter_path.is_file():
        print(f"[eval_bbh] missing {adapter_path} (re-run SFT with updated main_sft)", file=sys.stderr)
        return 2

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("lora_type", "default") != "default":
        print("[eval_bbh] only lora_type=default is supported for merge+BBH in this script.", file=sys.stderr)
        return 2

    try:
        import lm_eval
    except ImportError:
        print("[eval_bbh] pip install lm-eval lm_eval[hf] (and torch/transformers) on the eval node.", file=sys.stderr)
        return 2

    from deepseek.models_sft import ModelLoadConfig, load_model_and_tokenizer

    model_name = args.model_name or meta.get("model_name") or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(model_name, trust_remote_code=args.trust_remote_code, torch_dtype=args.torch_dtype)
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

    ad = torch.load(adapter_path, map_location=device)
    model.load_state_dict(ad, strict=False)
    model.eval()
    _merge_loralinear_inplace(model)

    merged_dir = (args.merged_hf_dir or (metrics_dir / "model_merged_hf")).resolve()
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    out_json = (args.output_json or (metrics_dir / "bbh_eval.json")).resolve()
    if args.dry_merge_only:
        payload = {"bbh_mean_acc": None, "note": "dry_merge_only", "merged_hf_dir": str(merged_dir)}
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[eval_bbh] wrote {out_json}")
        return 0

    simple_evaluate = getattr(lm_eval, "simple_evaluate", None)
    if simple_evaluate is None:
        from lm_eval import evaluator as _ev

        simple_evaluate = _ev.simple_evaluate

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    merged_s = merged_dir.as_posix()
    model_args = f"pretrained={merged_s},trust_remote_code={str(bool(args.trust_remote_code)).lower()}"
    kwargs = dict(
        model="hf",
        model_args=model_args,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
    )
    if args.limit and args.limit > 0:
        kwargs["limit"] = args.limit
    if torch.cuda.is_available():
        kwargs["device"] = "cuda"

    print(f"[eval_bbh] lm_eval tasks={task_list} model_args={model_args}", flush=True)
    results = simple_evaluate(**kwargs)
    mean_acc, task_acc = _bbh_scores_from_lm_eval(results)

    payload = {
        "metrics_dir": str(metrics_dir),
        "model_name": model_name,
        "merged_hf_dir": str(merged_dir),
        "tasks": task_list,
        "bbh_mean_acc": mean_acc,
        "bbh_task_acc": task_acc,
        "lm_eval_versions": results.get("versions"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval_bbh] bbh_mean_acc={mean_acc:.6f} wrote {out_json}")
    return 0 if mean_acc == mean_acc else 3


if __name__ == "__main__":
    raise SystemExit(main())
