#!/usr/bin/env python3
"""Validation perplexity vs step for **best LoRA / best mLoRA** on ``alpaca_train_full`` Top-K runs.

Best run per family = lowest **minimum eval perplexity seen during training** (``best_eval_perplexity``
in ``summary.csv``), **not** ``last_eval_perplexity``.

Outputs ``deepseek_autogrid/figures/alpaca_train_full_best_lora_mlora_val_ppl.png``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LORA_SUMMARY = PROJECT_ROOT / "deepseek_autogrid" / "results_final" / "summary.csv"
DEFAULT_MLORA_SUMMARY = PROJECT_ROOT / "deepseek_autogrid" / "results_mlora_final" / "summary.csv"
DEFAULT_OUT = PROJECT_ROOT / "deepseek_autogrid" / "figures" / "alpaca_train_full_best_lora_mlora_val_ppl.png"


def _best_row(summary: Path) -> dict:
    with summary.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("status") == "ok" and str(r.get("best_eval_perplexity", "")).strip()]
    if not rows:
        raise SystemExit(f"No ok rows in {summary}")
    return min(rows, key=lambda r: float(r["best_eval_perplexity"]))


def _load_val_curve(run_dir: Path) -> tuple[list[int], list[float]]:
    p = run_dir / "test_sft.csv"
    xs: list[int] = []
    ys: list[float] = []
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append(int(row["iteration"]))
            ys.append(float(row["eval_perplexity"]))
    return xs, ys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora-summary", type=Path, default=DEFAULT_LORA_SUMMARY)
    ap.add_argument("--mlora-summary", type=Path, default=DEFAULT_MLORA_SUMMARY)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Need matplotlib: pip install matplotlib") from e

    br = _best_row(args.lora_summary)
    bm = _best_row(args.mlora_summary)
    dr = (PROJECT_ROOT / br["metrics_dir"]).resolve()
    dm = (PROJECT_ROOT / bm["metrics_dir"]).resolve()
    xr, yr = _load_val_curve(dr)
    xm, ym = _load_val_curve(dm)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.2))
    plt.plot(
        xr,
        yr,
        label=(
            f"LoRA (best min val PPL={float(br['best_eval_perplexity']):.4f} @ step {br['best_iteration']}; "
            f"last={float(br['last_eval_perplexity']):.4f})"
        ),
        linewidth=1.8,
    )
    plt.plot(
        xm,
        ym,
        label=(
            f"mLoRA (best min val PPL={float(bm['best_eval_perplexity']):.4f} @ step {bm['best_iteration']}; "
            f"last={float(bm['last_eval_perplexity']):.4f})"
        ),
        linewidth=1.8,
    )
    plt.xlabel("training step")
    plt.ylabel("eval perplexity (validation)")
    plt.title("DeepSeek R1-Distill-Qwen-1.5B — full Alpaca Top-K — best run per adapter (by min val PPL)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
