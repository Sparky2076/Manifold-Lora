from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from deepseek.models_sft import ModelLoadConfig, load_model_and_tokenizer
from deepseek.utils_sft import SFTDataConfig, build_dataloaders, load_sft_dataset, split_train_val
from optimizers import AdamWConfig, get_optimizer


def _append_row(path: Path, row: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            if path.name == "train_sft.csv":
                w.writerow(["iteration", "train_loss", "train_perplexity"])
            else:
                w.writerow(["iteration", "eval_loss", "eval_perplexity"])
        w.writerow(row)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        valid = (batch["labels"] != -100).sum().item()
        total_loss += float(out.loss.item()) * max(valid, 1)
        total_tokens += max(valid, 1)
    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg, 20))
    return avg, ppl


def main():
    p = argparse.ArgumentParser(description="DeepSeek SFT (LoRA/mLoRA)")
    p.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--sft_preset", type=str, default="alpaca_train_1k")
    p.add_argument("--sft_dataset", type=str, default="")
    p.add_argument("--sft_split", type=str, default="")
    p.add_argument("--sft_val_ratio", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)

    p.add_argument("--lora_type", type=str, default="default", choices=["default", "mlora"])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_attention_only", action="store_true")
    p.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,out_proj,q_lin,k_lin,v_lin,out_lin,c_attn,c_proj",
    )

    p.add_argument("--metrics_dir", type=str, default="deepseek/results")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_and_tokenizer(
        ModelLoadConfig(args.model_name, trust_remote_code=args.trust_remote_code, torch_dtype=args.torch_dtype)
    )
    model.to(device)

    ds = load_sft_dataset(args.sft_dataset, args.sft_split, args.sft_preset)
    train_ds, val_ds = split_train_val(ds, args.sft_val_ratio, args.seed)
    train_loader, val_loader = build_dataloaders(
        tokenizer, train_ds, val_ds, SFTDataConfig(args.max_length, args.batch_size, args.num_workers)
    )

    if args.lora_type == "default":
        from lora import LoRAConfig, apply_lora, lora_trainable_parameters, mark_only_lora_as_trainable
    else:
        from mlora import LoRAConfig, apply_lora, lora_trainable_parameters, mark_only_lora_as_trainable

    lora_cfg = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        attention_only=args.lora_attention_only,
    )
    apply_lora(device, model, lora_cfg, verbose=True)
    mark_only_lora_as_trainable(model)
    trainable = lora_trainable_parameters(model)

    optimizer = get_optimizer(
        trainable,
        AdamWConfig(lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay),
    )
    warmup_steps = int(args.max_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, args.max_steps)
    scaler = GradScaler(enabled=(torch.cuda.is_available() and str(args.torch_dtype).lower() in ("fp16", "float16")))

    metrics_dir = Path(args.metrics_dir)
    train_csv = metrics_dir / "train_sft.csv"
    test_csv = metrics_dir / "test_sft.csv"

    step = 0
    running_loss = 0.0
    model.train()
    pbar = tqdm(total=args.max_steps, desc="[SFT Train]", dynamic_ncols=True)
    data_iter = iter(train_loader)
    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(enabled=scaler.is_enabled()):
                out = model(**batch)
                loss = out.loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.item()) * args.grad_accum_steps

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        step += 1
        pbar.update(1)

        if step % args.log_every == 0 or step == 1:
            avg = running_loss / max(step, 1)
            _append_row(train_csv, [step, f"{avg:.6f}", f"{math.exp(min(avg, 20)):.6f}"])
            pbar.set_postfix({"loss": f"{avg:.4f}"})

        if step % args.eval_every == 0 or step == args.max_steps:
            eval_loss, eval_ppl = evaluate(model, val_loader, device)
            _append_row(test_csv, [step, f"{eval_loss:.6f}", f"{eval_ppl:.6f}"])

    pbar.close()
    print(f"[done] max_steps={args.max_steps} metrics_dir={metrics_dir}")


if __name__ == "__main__":
    main()
