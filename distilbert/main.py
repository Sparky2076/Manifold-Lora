# main.py — DistilBERT / 序列分类 + LoRA；须在仓库根执行: python -m distilbert.main
from __future__ import annotations
import argparse
import math
import os
import sys
import time

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torchvision
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from distilbert.models import ModelLoadConfig, load_model_and_tokenizer
from distilbert.utils import build_dataloaders
from optimizers import AdamWConfig, get_optimizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _cuda_transient_error(exc: BaseException) -> bool:
    m = str(exc).lower()
    return "busy" in m or "unavailable" in m


def ensure_cuda_ready(device: torch.device) -> None:
    """Probe CUDA before loading a large model; retry on transient 'busy/unavailable' driver errors."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    retries = max(1, int(os.environ.get("TRAIN_CUDA_RETRY", "8") or "8"))
    sleep_sec = max(1.0, float(os.environ.get("TRAIN_CUDA_RETRY_SEC", "20") or "20"))
    idx = device.index if device.index is not None else 0
    last: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            torch.cuda.set_device(idx)
            torch.cuda.empty_cache()
            # Small allocation + sync — same class of failure as model.to(), but cheap to retry.
            x = torch.ones(512, 512, device=device)
            del x
            torch.cuda.synchronize()
            print(f"[CUDA] device cuda:{idx} ready (after {attempt} attempt(s)).", flush=True)
            return
        except RuntimeError as e:
            last = e
            if not _cuda_transient_error(e):
                raise
            print(
                f"[CUDA] GPU not ready ({attempt}/{retries}): {e}\n"
                f"      Set TRAIN_CUDA_RETRY / TRAIN_CUDA_RETRY_SEC to tune; "
                f"or check nvidia-smi / exclusive GPU scheduling on the cluster.",
                file=sys.stderr,
                flush=True,
            )
            if attempt < retries:
                time.sleep(sleep_sec)
    assert last is not None
    raise last


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_device_map: bool,
    desc: str = "[Test]",
) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=True)
    for batch in pbar:
        if use_device_map:
            model_device = next(model.parameters()).device
            batch = {k: v.to(model_device) for k, v in batch.items()}
        else:
            batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["labels"]

        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

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
    train_csv_path: str = "",
    log_every: int = 50,
    global_step: list | None = None,
) -> float:
    model.train()

    running_loss = 0.0
    seen = 0
    correct = 0
    total = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f"[Train][Epoch {epoch}/{args.epochs}]", dynamic_ncols=True, leave=True)
    for step, batch in enumerate(pbar, start=1):
        if use_device_map:
            model_device = next(model.parameters()).device
            batch = {k: v.to(model_device) for k, v in batch.items()}
        else:
            batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast(
            enabled=(args.torch_dtype in ("float16", "fp16", "bfloat16", "bf16")) and torch.cuda.is_available(),
            dtype=(torch.bfloat16 if args.torch_dtype in ("bfloat16", "bf16") else torch.float16),
        ):
            outputs = model(**batch)
            logits = outputs.logits
            loss = outputs.loss / args.grad_accum_steps

        labels = batch["labels"]
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

            if global_step is not None and train_csv_path:
                global_step[0] += 1
                avg_loss = running_loss / max(seen, 1)
                acc = correct / max(total, 1)
                if global_step[0] % log_every == 0:
                    with open(train_csv_path, "a") as f:
                        f.write(f"{global_step[0]},{avg_loss:.4f},{acc:.4f}\n")

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
    metrics_dir: str = ".",
) -> Tuple[float, int]:
    os.makedirs(metrics_dir, exist_ok=True)
    train_csv_path = os.path.join(metrics_dir, "train.csv")
    test_csv_path = os.path.join(metrics_dir, "test.csv")
    with open(train_csv_path, "w") as f:
        f.write("iteration,train_loss,train_accuracy\n")
    with open(test_csv_path, "w") as f:
        f.write("iteration,test_loss,test_accuracy\n")

    best_acc = -1.0
    best_epoch = -1
    global_step = [0]

    for epoch in range(1, args.epochs + 1):
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
            train_csv_path=train_csv_path,
            log_every=args.log_every,
            global_step=global_step,
        )

        eval_loss, eval_acc = evaluate(
            model=model,
            dataloader=eval_loader,
            device=device,
            use_device_map=use_device_map,
            desc=f"[Test][Epoch {epoch}/{args.epochs}]",
        )

        with open(test_csv_path, "a") as f:
            f.write(f"{global_step[0]},{eval_loss:.4f},{eval_acc:.4f}\n")

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

    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config", type=str, default="sst2")
    parser.add_argument("--text_field", type=str, default="sentence")
    parser.add_argument("--num_labels", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--torch_dtype", type=str, default=None, choices=[None, "float16", "bfloat16", "float32", "fp16", "bf16", "fp32"])
    parser.add_argument("--device_map", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_attention_only", action="store_true")
    parser.add_argument("--lora_targets", type=str, default="q_proj,k_proj,v_proj,out_proj,q_lin,k_lin,v_lin,out_lin,c_attn,c_proj")
    parser.add_argument("--lora_type", type=str, default="default", choices=["default", "mlora"])

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=50, help="Accepted for compatibility; evaluation is per-epoch.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--metrics_dir", type=str, default=".")

    args = parser.parse_args()

    use_device_map = args.device_map is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        ensure_cuda_ready(device)

    model_cfg = ModelLoadConfig(
        model_name=args.model_name,
        num_labels=args.num_labels,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    if not use_device_map:
        try:
            model.to(device)
        except RuntimeError as e:
            if _cuda_transient_error(e):
                print(
                    "[CUDA] model.to() failed after probe succeeded; another process may have grabbed the GPU, "
                    "or the device is in a bad state. Check nvidia-smi on the compute node and cluster GPU binding.",
                    file=sys.stderr,
                    flush=True,
                )
            raise

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

    opt_cfg = AdamWConfig(
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    optimizer = get_optimizer(trainable, opt_cfg)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

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
        metrics_dir=args.metrics_dir,
    )
    print(f"[Done] Best eval accuracy = {best_acc:.4f} @ epoch {best_epoch}")
    train_csv = os.path.join(args.metrics_dir, "train.csv")
    test_csv = os.path.join(args.metrics_dir, "test.csv")
    if os.path.isfile(train_csv) and os.path.isfile(test_csv):
        print(f"SUCCESS: 指标已保存至 {train_csv} 和 {test_csv}")


if __name__ == "__main__":
    main()
