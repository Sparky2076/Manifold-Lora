#!/usr/bin/env python3
"""Run full-factor DistilBERT LoRA grid: lr × r × alpha × weight_decay (betas fixed in config.py)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from distilbert_autogrid.config import (
    ADAM_BETA1_FIXED,
    ADAM_BETA2_FIXED,
    EPOCHS_DEFAULT,
    MODEL_NAME_DEFAULT,
    PROJECT_ROOT,
    RESULTS_ROOT,
    grid_size,
    iter_grid,
    run_dir_name,
)


def build_train_cmd(
    metrics_dir: Path,
    lr: float,
    r: int,
    alpha: float,
    epochs: int,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    python_exe: str,
) -> list[str]:
    return [
        python_exe,
        "-m",
        "distilbert.main",
        "--model_name",
        MODEL_NAME_DEFAULT,
        "--dataset_name",
        "glue",
        "--dataset_config",
        "sst2",
        "--text_field",
        "sentence",
        "--epochs",
        str(epochs),
        "--batch_size",
        "4",
        "--grad_accum_steps",
        "8",
        "--max_length",
        "128",
        "--lr",
        str(lr),
        "--weight_decay",
        str(weight_decay),
        "--adam_beta1",
        str(adam_beta1),
        "--adam_beta2",
        str(adam_beta2),
        "--lora_r",
        str(r),
        "--lora_alpha",
        str(alpha),
        "--lora_dropout",
        "0.05",
        "--lora_type",
        "default",
        "--metrics_dir",
        str(metrics_dir),
    ]


def write_run_meta(
    path: Path,
    *,
    lr: float,
    r: int,
    alpha: float,
    epochs: int,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    metrics_dir: str,
    command: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lr": lr,
        "lora_r": r,
        "lora_alpha": alpha,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "adam_beta1": adam_beta1,
        "adam_beta2": adam_beta2,
        "metrics_dir": metrics_dir,
        "command": command,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def append_failed(failed_log: Path, record: dict) -> None:
    failed_log.parent.mkdir(parents=True, exist_ok=True)
    with failed_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_grid() -> int:
    parser = argparse.ArgumentParser(
        description=f"DistilBERT LoRA full grid ({grid_size()} combos: lr×r×alpha×wd; betas fixed)"
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help=f"Root directory for per-run folders (default: {RESULTS_ROOT})",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT, help="Training epochs (single value for all runs)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not train")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the grid on first non-zero exit (default: continue and log failures)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Stop after this many successful training runs",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Stop after this many subprocess attempts (success or fail; for smoke tests)",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for subprocess")
    args = parser.parse_args()

    results_root: Path = args.results_root.resolve()
    failed_log = results_root / "failed_runs.jsonl"

    total = 0
    done = 0
    attempts = 0
    for lr, r, alpha, weight_decay in iter_grid():
        total += 1
        name = run_dir_name(lr, r, alpha, args.epochs, weight_decay)
        metrics_dir = results_root / name
        cmd = build_train_cmd(
            metrics_dir,
            lr,
            r,
            alpha,
            args.epochs,
            weight_decay,
            ADAM_BETA1_FIXED,
            ADAM_BETA2_FIXED,
            args.python,
        )

        if not args.dry_run:
            meta_path = metrics_dir / "run_meta.json"
            write_run_meta(
                meta_path,
                lr=lr,
                r=r,
                alpha=alpha,
                epochs=args.epochs,
                weight_decay=weight_decay,
                adam_beta1=ADAM_BETA1_FIXED,
                adam_beta2=ADAM_BETA2_FIXED,
                metrics_dir=str(metrics_dir),
                command=cmd,
            )

        if args.dry_run:
            print("[dry-run]", " ".join(cmd))
            continue

        attempts += 1
        t0 = time.perf_counter()
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                check=False,
            )
        except OSError as e:
            append_failed(
                failed_log,
                {
                    "lr": lr,
                    "lora_r": r,
                    "lora_alpha": alpha,
                    "epochs": args.epochs,
                    "weight_decay": weight_decay,
                    "adam_beta1": ADAM_BETA1_FIXED,
                    "adam_beta2": ADAM_BETA2_FIXED,
                    "metrics_dir": str(metrics_dir),
                    "error": str(e),
                    "exit_code": None,
                },
            )
            if args.stop_on_error:
                print(f"[fatal] OSError: {e}", file=sys.stderr)
                return 1
            continue

        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            append_failed(
                failed_log,
                {
                    "lr": lr,
                    "lora_r": r,
                    "lora_alpha": alpha,
                    "epochs": args.epochs,
                    "weight_decay": weight_decay,
                    "adam_beta1": ADAM_BETA1_FIXED,
                    "adam_beta2": ADAM_BETA2_FIXED,
                    "metrics_dir": str(metrics_dir),
                    "exit_code": proc.returncode,
                    "seconds": round(elapsed, 3),
                },
            )
            print(
                f"[fail] lr={lr} r={r} a={alpha} wd={weight_decay} "
                f"code={proc.returncode} ({elapsed:.1f}s) -> {metrics_dir}",
                file=sys.stderr,
            )
            if args.stop_on_error:
                return proc.returncode
        else:
            done += 1
            print(
                f"[ok {done}] lr={lr} r={r} a={alpha} wd={weight_decay} "
                f"({elapsed:.1f}s) -> {metrics_dir}"
            )

        if args.max_runs is not None and done >= args.max_runs:
            print(f"[stop] --max-runs {args.max_runs} reached ({done} successful).")
            break
        if args.max_attempts is not None and attempts >= args.max_attempts:
            print(f"[stop] --max-attempts {args.max_attempts} reached.")
            break

    if args.dry_run:
        print(f"[dry-run] {total} combinations (no training).")
        return 0

    print(f"[done] successful={done}, total_loops={total}, results_root={results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_grid())
