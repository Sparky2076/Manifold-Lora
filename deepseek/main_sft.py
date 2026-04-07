# main_sft.py — DeepSeek / CausalLM 指令微调（SFT）；须在仓库根目录执行: python -m deepseek.main_sft
from __future__ import annotations

import argparse
import math
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from deepseek.models_sft import CausalLMConfig, load_causal_lm_and_tokenizer
from deepseek.utils_sft import SFTDataConfig, SFT_DATASET_PRESETS, build_sft_dataloaders
from optimizers import AdamWConfig, get_optimizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_sft(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_device_map: bool,
    torch_dtype_str: str | None,
    desc: str = "[Eval]",
) -> float:
    """返回平均 LM loss（越低越好）。"""
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    skipped_non_finite = 0

    pbar = tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=True)
    for batch in pbar:
        if use_device_map:
            model_device = next(model.parameters()).device
            batch = {k: v.to(model_device) for k, v in batch.items()}
        else:
            batch = {k: v.to(device) for k, v in batch.items()}

        # CausalLM loss 内部会右移 labels，若 labels[:, 1:] 全为 -100 则该 batch 无有效监督信号
        if not (batch["labels"][:, 1:] != -100).any():
            skipped_non_finite += 1
            continue

        with torch.cuda.amp.autocast(
            enabled=(torch_dtype_str in ("float16", "fp16", "bfloat16", "bf16"))
            and torch.cuda.is_available(),
            dtype=(torch.bfloat16 if torch_dtype_str in ("bfloat16", "bf16") else torch.float16),
        ):
            outputs = model(**batch)
            loss = outputs.loss

        if not torch.isfinite(loss):
            skipped_non_finite += 1
            continue

        loss_sum += loss.item()
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss_sum / max(n_batches, 1):.4f}"})

    pbar.close()
    if skipped_non_finite > 0:
        print(f"[Warn] eval skipped non-finite batches: {skipped_non_finite}")
    if n_batches == 0:
        raise RuntimeError("All eval batches were skipped (no valid supervised tokens after shift).")
    return loss_sum / max(n_batches, 1)


def train_one_epoch_sft(
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
    train_csv_path: str,
    log_every: int,
    global_step: list,
) -> float:
    """结构与 `distilbert/main.py` 的 train_one_epoch 一致。"""
    model.train()
    running_loss = 0.0
    seen = 0
    optimizer.zero_grad(set_to_none=True)
    skipped_non_finite = 0

    pbar = tqdm(
        train_loader,
        desc=f"[SFT Train][Epoch {epoch}/{args.epochs}]",
        dynamic_ncols=True,
        leave=True,
    )
    for step, batch in enumerate(pbar, start=1):
        if use_device_map:
            model_device = next(model.parameters()).device
            batch = {k: v.to(model_device) for k, v in batch.items()}
        else:
            batch = {k: v.to(device) for k, v in batch.items()}

        if not (batch["labels"][:, 1:] != -100).any():
            skipped_non_finite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        with torch.cuda.amp.autocast(
            enabled=(args.torch_dtype in ("float16", "fp16", "bfloat16", "bf16"))
            and torch.cuda.is_available(),
            dtype=(torch.bfloat16 if args.torch_dtype in ("bfloat16", "bf16") else torch.float16),
        ):
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps

        if not torch.isfinite(loss):
            skipped_non_finite += 1
            optimizer.zero_grad(set_to_none=True)
            continue

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

            global_step[0] += 1
            avg_loss = running_loss / max(seen, 1)
            ppl = min(math.exp(min(avg_loss, 20.0)), 1e9)
            if global_step[0] % log_every == 0 and train_csv_path:
                with open(train_csv_path, "a") as f:
                    f.write(f"{global_step[0]},{avg_loss:.4f},{ppl:.4f}\n")

            # 大数据 + max_steps：不得在整 epoch 上遍历全部样本，否则极慢且易 OOM
            if args.max_steps is not None and args.max_steps > 0 and global_step[0] >= args.max_steps:
                pbar.close()
                if skipped_non_finite > 0:
                    print(f"[Warn] train skipped non-finite batches: {skipped_non_finite}")
                return running_loss / max(seen, 1)

        avg_loss = running_loss / max(seen, 1)
        ppl = min(math.exp(min(avg_loss, 20.0)), 1e9)
        pbar.set_postfix(
            {"loss": f"{avg_loss:.4f}", "ppl": f"{ppl:.2f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"}
        )

    pbar.close()
    if skipped_non_finite > 0:
        print(f"[Warn] train skipped non-finite batches: {skipped_non_finite}")
    if seen == 0:
        raise RuntimeError("All train batches were skipped (no valid supervised tokens after shift).")
    return running_loss / max(seen, 1)


def train_sft_loop(
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
    metrics_dir: str,
) -> tuple[float, int]:
    os.makedirs(metrics_dir, exist_ok=True)
    train_csv = os.path.join(metrics_dir, "train_sft.csv")
    test_csv = os.path.join(metrics_dir, "test_sft.csv")
    with open(train_csv, "w") as f:
        f.write("iteration,train_loss,train_perplexity\n")
    with open(test_csv, "w") as f:
        f.write("iteration,eval_loss,eval_perplexity\n")

    best_loss = float("inf")
    best_epoch = -1
    global_step = [0]

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_sft(
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
            train_csv_path=train_csv,
            log_every=args.log_every,
            global_step=global_step,
        )

        eval_loss = evaluate_sft(
            model=model,
            dataloader=eval_loader,
            device=device,
            use_device_map=use_device_map,
            torch_dtype_str=args.torch_dtype,
            desc=f"[SFT Eval][Epoch {epoch}/{args.epochs}]",
        )
        eval_ppl = min(math.exp(min(eval_loss, 20.0)), 1e9)

        with open(test_csv, "a") as f:
            f.write(f"{global_step[0]},{eval_loss:.4f},{eval_ppl:.4f}\n")

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_epoch = epoch

        print(
            f"[Epoch {epoch} Ends] train_loss={train_loss:.4f} | "
            f"eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f} | "
            f"best_eval_loss={best_loss:.4f} (best epoch: {best_epoch})"
        )

        # 用 max_steps 控制训练预算：达到后提前结束（更适合大数据集复验）
        if args.max_steps is not None and args.max_steps > 0 and global_step[0] >= args.max_steps:
            print(f"[Stop] Reached max_steps={args.max_steps} at epoch {epoch}.")
            break

    return best_loss, best_epoch


def main():
    parser = argparse.ArgumentParser(description="DeepSeek / CausalLM 指令微调 (SFT)")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--sft_dataset", type=str, default="HuggingFaceH4/testing_alpaca_small")
    parser.add_argument("--sft_dataset_config", type=str, default=None)
    parser.add_argument("--sft_split", type=str, default="train", help="如 train 或 train[:500]")
    parser.add_argument("--sft_preset", type=str, default=None, help=f"可选: {', '.join(SFT_DATASET_PRESETS.keys())}")
    parser.add_argument("--sft_max_samples", type=int, default=None)
    parser.add_argument(
        "--sft_val_ratio",
        type=float,
        default=0.2,
        help="从训练 split 中划出验证集的比例（如 0.2 ≈ 20%% 样本做 eval）",
    )

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="可选：按 optimizer step 数限制训练预算（优先于 epochs 用于提前结束训练）",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32", "fp16", "bf16", "fp32"],
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="设为 none 则不用 device_map（单卡 .to(cuda)）",
    )

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_attention_only", action="store_true")
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,q_lin,k_lin,v_lin,out_lin,c_attn,c_proj",
    )
    parser.add_argument("--lora_type", type=str, default="default", choices=["default", "mlora"])

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--metrics_dir", type=str, default=".")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="开启 HF gradient checkpointing，显著降低显存（略慢）",
    )

    args = parser.parse_args()

    if args.sft_preset:
        if args.sft_preset not in SFT_DATASET_PRESETS:
            raise ValueError(f"Unknown sft_preset={args.sft_preset}. Choose from {list(SFT_DATASET_PRESETS)}")
        did, dcfg, dsp = SFT_DATASET_PRESETS[args.sft_preset]
        args.sft_dataset = did
        if dcfg is not None:
            args.sft_dataset_config = dcfg
        if dsp is not None:
            args.sft_split = dsp

    set_seed(args.seed)

    dm = (args.device_map or "").strip().lower()
    use_device_map = dm not in ("", "none", "false", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_causal_lm_and_tokenizer(
        CausalLMConfig(
            model_name=args.model_name,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map if use_device_map else None,
        )
    )
    if not use_device_map:
        model.to(device)

    data_cfg = SFTDataConfig(
        dataset_id=args.sft_dataset,
        dataset_config=args.sft_dataset_config,
        split_train=args.sft_split,
        val_ratio=args.sft_val_ratio,
        max_samples=args.sft_max_samples,
        max_length=args.max_length,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader, eval_loader = build_sft_dataloaders(
        tokenizer=tokenizer,
        cfg=data_cfg,
        batch_size=args.batch_size,
    )

    if args.lora_type == "default":
        from lora import LoRAConfig, apply_lora, mark_only_lora_as_trainable, lora_trainable_parameters

        print("[SFT] Using default LoRA from lora.py")
    else:
        from mlora import LoRAConfig, apply_lora, mark_only_lora_as_trainable, lora_trainable_parameters

        print("[SFT] Using mLoRA from mlora.py")

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

    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("[SFT] gradient_checkpointing enabled")
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    trainable = lora_trainable_parameters(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[SFT Params] trainable={n_trainable:,} / total={n_total:,} ({100.0 * n_trainable / n_total:.4f}%)")

    optimizer = get_optimizer(trainable, AdamWConfig(lr=args.lr, weight_decay=args.weight_decay))

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    if args.max_steps is not None and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(args.torch_dtype in ("float16", "fp16")) and torch.cuda.is_available()
    )

    best_loss, best_epoch = train_sft_loop(
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
    print(f"[Done] Best eval loss = {best_loss:.4f} @ epoch {best_epoch}")
    print(f"Metrics: {os.path.join(args.metrics_dir, 'train_sft.csv')} , {os.path.join(args.metrics_dir, 'test_sft.csv')}")


if __name__ == "__main__":
    main()
