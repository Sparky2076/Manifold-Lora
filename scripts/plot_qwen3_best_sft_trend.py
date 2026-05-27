#!/usr/bin/env python3
"""Plot train loss + val perplexity vs step for the best Qwen3 refine run."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", type=Path, required=True)
    p.add_argument("--test-csv", type=Path, required=True)
    p.add_argument("--out-png", type=Path, required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--full-mmlu-acc", type=float, default=None, help="Optional full MMLU mean acc (0-1).")
    args = p.parse_args()

    train_rows = _read_csv(args.train_csv)
    test_rows = _read_csv(args.test_csv)

    train_steps = [int(r["iteration"]) for r in train_rows]
    train_loss = [float(r["train_loss"]) for r in train_rows]
    val_steps = [int(r["iteration"]) for r in test_rows]
    val_ppl = [float(r["eval_perplexity"]) for r in test_rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(train_steps, train_loss, color="#1f77b4", linewidth=2, label="Train loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(val_steps, val_ppl, color="#d62728", marker="o", linewidth=2, label="Val perplexity")
    ax2.set_ylabel("Val perplexity (PPL)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    title = f"Qwen3 SFT trend  lr={args.lr:g}  r={args.lora_r}  wd={args.weight_decay:g}"
    if args.full_mmlu_acc is not None:
        title += f"  |  full MMLU {100 * args.full_mmlu_acc:.2f}%"
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
