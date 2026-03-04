# main.py
from __future__ import annotations
import argparse
import math
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torchvision
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from models import ModelLoadConfig, load_model_and_tokenizer
from optimizers import AdamWConfig, get_optimizer
from utils import build_dataloaders


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_device_map: bool,
    desc: str = "[Test]"
) -> Tuple[float, float]:
    """Run evaluation with a dedicated progress bar.

    Returns:
        (avg_loss, accuracy) where accuracy is in [0,1].
    """
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=True)
    for batch in pbar:
        if not use_device_map:
            batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        # Ensure labels are on the same device as logits (important when using device_map).
        if labels.device != logits.device:
            labels = labels.to(logits.device)

        preds = torch.argmax(logits, dim=-1)
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.numel()
        correct += batch_correct
        total += batch_total
        loss_sum += loss.item() * labels.size(0)

        avg_loss = loss_sum / max(total, 1)
        acc = correct / max(total, 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}"})

    pbar.close()
    return (loss_sum / max(total, 1)), (correct / max(total, 1))


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    args,
    device: torch.device,
    use_device_map: bool,
    trainable_params,
) -> float:
    """Train for one epoch with its own progress bar. Returns average loss."""
    model.train()

    running_loss = 0.0
    seen = 0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"[Train][Epoch {epoch}/{args.epochs}]", dynamic_ncols=True, leave=True)
    for step, batch in enumerate(pbar, start=1):
        if not use_device_map:
            batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(
            enabled=(args.torch_dtype in ("float16", "fp16", "bfloat16", "bf16")) and torch.cuda.is_available(),
            dtype=(torch.bfloat16 if args.torch_dtype in ("bfloat16", "bf16") else torch.float16),
        ):
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss / args.grad_accum_steps

        labels = batch["labels"]
        # For device_map, labels may stay on CPU; align to logits device for accuracy computation.
        if labels.device != logits.device:
            labels = labels.to(logits.device)
        preds = torch.argmax(logits, dim=-1)
        batch_correct = (preds == labels).sum().item()
        batch_total = labels.numel()
        correct += batch_correct
        total += batch_total

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        bs = batch["labels"].size(0)
        running_loss += loss.item() * args.grad_accum_steps * bs
        seen += bs

        if step % args.grad_accum_steps == 0:
            if args.max_grad_norm and args.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = running_loss / max(seen, 1)
        acc = correct / max(total, 1)
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    pbar.close()
    return running_loss / max(seen, 1)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    args,
    device: torch.device,
    use_device_map: bool,
    trainable_params,
) -> Tuple[float, int]:
    """Full training loop. Evaluates once at the end of each epoch.

    Returns:
        (best_acc, best_epoch) where best_acc is in [0,1].
    """
    best_acc = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        # ---- Train (progress bar #1) ----
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            args=args,
            device=device,
            use_device_map=use_device_map,
            trainable_params=trainable_params,
        )

        # ---- Test once per epoch (progress bar #2) ----
        eval_loss, eval_acc = evaluate(
            model=model,
            dataloader=eval_loader,
            device=device,
            use_device_map=use_device_map,
            desc=f"[Test][Epoch {epoch}/{args.epochs}]",
        )

        if eval_acc > best_acc:
            best_acc = eval_acc
            best_epoch = epoch

        print(
            f"[Epoch {epoch} Ends] train_loss={train_loss:.4f} | "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} | "
            f"best_acc={best_acc:.4f} (from the best epoch: {best_epoch})"
        )

    return best_acc, best_epoch


def main():
    torchvision.disable_beta_transforms_warning()
    parser = argparse.ArgumentParser()

    # Model / dataset
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config", type=str, default="sst2")
    parser.add_argument("--text_field", type=str, default="sentence")
    parser.add_argument("--num_labels", type=int, default=2)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)  # (not used separately here, kept for extension)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Mixed precision / device map
    parser.add_argument("--torch_dtype", type=str, default=None, choices=[None, "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--device_map", type=str, default=None)  # e.g., "auto"

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_attention_only", action="store_true")
    parser.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,out_proj,q_lin,k_lin,v_lin,out_lin,c_attn,c_proj")
    parser.add_argument("--lora_type", type=str, default="default", choices=["default", "mlora"])

    # Logging
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    # Device handling:
    # If device_map is used (accelerate-style), model may be sharded; in that case we do not .to(device).
    use_device_map = args.device_map is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model/tokenizer
    model_cfg = ModelLoadConfig(
        model_name=args.model_name,
        num_labels=args.num_labels,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    if not use_device_map:
        model.to(device)

    # Data
    train_loader, eval_loader = build_dataloaders(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_field=args.text_field,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    if args.lora_type == "default":
        from lora import LoRAConfig, apply_lora, mark_only_lora_as_trainable, lora_trainable_parameters
        print("[Info] Using default LoRA implementation from lora.py")
    elif args.lora_type == "mlora":
        from mlora import LoRAConfig, apply_lora, mark_only_lora_as_trainable, lora_trainable_parameters
        print("[Info] Using mLoRA implementation from mlora.py")
    # Apply LoRA
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora_cfg = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=targets,
        attention_only=args.lora_attention_only,
    )
    apply_lora(device, model, lora_cfg, verbose=True)
    mark_only_lora_as_trainable(model)

    trainable = lora_trainable_parameters(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Params] trainable={n_trainable:,} / total={n_total:,} ({100.0*n_trainable/n_total:.4f}%)")

    # Optimizer (from optimizers.py)
    opt_cfg = AdamWConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    optimizer = get_optimizer(trainable, opt_cfg)

    # Scheduler
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    scaler = torch.cuda.amp.GradScaler(enabled=(args.torch_dtype in ("float16", "fp16")) and torch.cuda.is_available())

    best_acc, best_epoch = train(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        device=device,
        use_device_map=use_device_map,
        trainable_params=trainable,
    )
    print(f"[Done] Best eval accuracy = {best_acc:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()


""" python main.py --model_name distilbert/distilbert-base-uncased --dataset_name glue --dataset_config sst2 --text_field sentence --epochs 3 --batch_size 32 --max_length 256 --lr 2e-4 --weight_decay 0.01 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --log_every 100 """

""" python main.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --trust_remote_code --device_map auto --torch_dtype bfloat16 --dataset_name glue --dataset_config sst2 --text_field sentence --epochs 1 --batch_size 8 --max_length 256 --lr 2e-4 --weight_decay 0.01 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --log_every 100 """