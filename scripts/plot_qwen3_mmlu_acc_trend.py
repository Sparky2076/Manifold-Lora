#!/usr/bin/env python3
"""Plot running-mean MMLU accuracy trend from eval JSON or progress CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _read_progress_csv(path: Path) -> tuple[list[int], list[float], str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No data rows in {path}")
    steps = [int(r["step"]) for r in rows]
    accs = [float(r["mean_acc"]) * 100.0 for r in rows]
    if len(rows) == 1:
        return steps, accs, "eval checkpoint (single pass)"
    return steps, accs, "eval progress step"


def _cumulative_from_task_acc(task_acc: dict[str, float]) -> tuple[list[int], list[float], float]:
    """Alphabetical subject order; running mean over all task_acc keys (matches eval mean)."""
    names = sorted(task_acc.keys())
    accs = [float(task_acc[n]) for n in names]
    if not accs:
        raise ValueError("task_acc is empty")
    running: list[float] = []
    total = 0.0
    for i, a in enumerate(accs, start=1):
        total += a
        running.append(100.0 * total / i)
    return list(range(1, len(accs) + 1)), running, 100.0 * sum(accs) / len(accs)


def _final_acc_from_json(payload: dict) -> float:
    for key in ("mmlu_mean_acc", "mean_acc"):
        v = payload.get(key)
        if v is not None and v == v:
            return float(v) * 100.0
    _, running, final = _cumulative_from_task_acc(payload.get("task_acc") or {})
    return running[-1] if running else final


def main() -> int:
    p = argparse.ArgumentParser(description="Plot MMLU running-mean accuracy trend.")
    p.add_argument("--eval-json", type=Path, default=None, help="mmlu_eval_full.json with task_acc")
    p.add_argument("--progress-csv", type=Path, default=None, help="mmlu_eval_full_progress.csv")
    p.add_argument("--out-png", type=Path, required=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--final-acc-pct", type=float, default=None, help="Override final %% in title.")
    args = p.parse_args()

    if args.eval_json is None and args.progress_csv is None:
        p.error("Provide --eval-json and/or --progress-csv")

    x_label = ""
    x_vals: list[int] = []
    y_vals: list[float] = []
    n_points = 0

    if args.progress_csv is not None and args.progress_csv.is_file():
        x_vals, y_vals, x_label = _read_progress_csv(args.progress_csv)
        n_points = len(y_vals)
        if n_points > 1:
            final_pct = y_vals[-1]
        else:
            if args.eval_json is None or not args.eval_json.is_file():
                final_pct = y_vals[-1]
            else:
                payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
                task_acc = payload.get("task_acc") or {}
                if task_acc:
                    x_vals, y_vals, final_pct = _cumulative_from_task_acc(task_acc)
                    x_label = "MMLU subject index (alphabetical)"
                    n_points = len(y_vals)
                else:
                    final_pct = _final_acc_from_json(payload)
    elif args.eval_json is not None and args.eval_json.is_file():
        payload = json.loads(args.eval_json.read_text(encoding="utf-8"))
        task_acc = payload.get("task_acc") or {}
        if not task_acc:
            raise ValueError(f"No task_acc in {args.eval_json}")
        x_vals, y_vals, final_pct = _cumulative_from_task_acc(task_acc)
        x_label = "MMLU subject index (alphabetical)"
        n_points = len(y_vals)
    else:
        raise FileNotFoundError("Neither eval JSON nor progress CSV found")

    if args.final_acc_pct is not None:
        final_pct = args.final_acc_pct
    elif not y_vals:
        raise ValueError("No accuracy points to plot")
    else:
        final_pct = y_vals[-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_vals, y_vals, color="#2ca02c", linewidth=2, marker="o", markersize=4, label="Running mean MMLU acc")
    ax.set_xlabel(x_label)
    ax.set_ylabel("MMLU accuracy (%)")
    ax.set_ylim(0, max(100.0, max(y_vals) * 1.05))
    ax.grid(True, alpha=0.3)
    title = f"Qwen3 MMLU trend  lr={args.lr:g}  r={args.lora_r}  wd={args.weight_decay:g}  |  final {final_pct:.2f}%"
    ax.set_title(title)
    ax.legend(loc="lower right")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out_png}  points={n_points}  x={x_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
