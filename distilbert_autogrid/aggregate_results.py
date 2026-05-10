#!/usr/bin/env python3
"""Scan grid results: read run_meta.json + test.csv, write summary.csv."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from distilbert_autogrid.config import EPOCHS_DEFAULT, PROJECT_ROOT, RESULTS_ROOT, grid_size


def read_run_meta(run_dir: Path) -> dict | None:
    p = run_dir / "run_meta.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def best_from_test_csv(test_csv: Path) -> tuple[float | None, float | None, int | None]:
    """Return (best_val_acc, test_loss at best acc, last_iteration_at_best)."""
    if not test_csv.is_file():
        return None, None, None
    rows: list[tuple[int, float, float]] = []
    with test_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                it = int(row["iteration"])
                loss = float(row["test_loss"])
                acc = float(row["test_accuracy"])
            except (KeyError, ValueError):
                continue
            rows.append((it, loss, acc))
    if not rows:
        return None, None, None
    best_acc = max(acc for _, _, acc in rows)
    candidates = [(it, loss, acc) for it, loss, acc in rows if acc == best_acc]
    it, loss, acc = min(candidates, key=lambda x: x[0])
    return acc, loss, it


def test_rows_count(test_csv: Path) -> int:
    """Count data rows in test.csv as (total lines - header), same rule as run_grid_bsub.sh."""
    if not test_csv.is_file():
        return 0
    try:
        lines = test_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
        return max(0, len(lines) - 1)
    except OSError:
        return 0


def aggregate() -> int:
    parser = argparse.ArgumentParser(description="Build summary.csv from distilbert_autogrid results")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help=f"Grid results root (default: {RESULTS_ROOT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <results-root>/summary.csv)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS_DEFAULT,
        help="Fallback epochs when run_meta.json lacks epochs (default: config EPOCHS_DEFAULT)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Do not fail when ok rows are fewer than full grid size",
    )
    args = parser.parse_args()

    root: Path = args.results_root.resolve()
    out_path = args.output or (root / "summary.csv")

    fieldnames = [
        "lr",
        "lora_r",
        "lora_alpha",
        "epochs",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "best_val_acc",
        "best_val_loss",
        "best_iteration",
        "last_val_acc",
        "last_val_loss",
        "last_iteration",
        "metrics_dir",
        "status",
    ]

    rows_out: list[dict] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        test_csv = run_dir / "test.csv"
        meta = read_run_meta(run_dir)

        last_acc = last_loss = last_it = None
        if test_csv.is_file():
            with test_csv.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                last = None
                for row in r:
                    try:
                        last = (
                            int(row["iteration"]),
                            float(row["test_loss"]),
                            float(row["test_accuracy"]),
                        )
                    except (KeyError, ValueError):
                        continue
                if last is not None:
                    last_it, last_loss, last_acc = last

        best_acc, best_loss, best_it = best_from_test_csv(test_csv)

        if meta:
            lr = meta.get("lr")
            r_ = meta.get("lora_r")
            alpha = meta.get("lora_alpha")
            ep = meta.get("epochs")
            wd = meta.get("weight_decay", "")
            b1 = meta.get("adam_beta1", "")
            b2 = meta.get("adam_beta2", "")
        else:
            lr = r_ = alpha = ep = wd = b1 = b2 = ""

        required_epochs = args.epochs
        if meta and meta.get("epochs") not in ("", None):
            try:
                required_epochs = int(meta.get("epochs"))
            except (TypeError, ValueError):
                required_epochs = args.epochs
        rows_n = test_rows_count(test_csv)

        if test_csv.is_file() and best_acc is not None and rows_n >= required_epochs:
            status = "ok"
        elif test_csv.is_file() and rows_n < required_epochs:
            status = "incomplete_test_rows"
        elif meta and not test_csv.is_file():
            status = "incomplete"
        elif not meta and not test_csv.is_file():
            continue
        else:
            status = "parse_error"

        rows_out.append(
            {
                "lr": lr,
                "lora_r": r_,
                "lora_alpha": alpha,
                "epochs": ep,
                "weight_decay": wd,
                "adam_beta1": b1,
                "adam_beta2": b2,
                "best_val_acc": best_acc if best_acc is not None else "",
                "best_val_loss": best_loss if best_loss is not None else "",
                "best_iteration": best_it if best_it is not None else "",
                "last_val_acc": last_acc if last_acc is not None else "",
                "last_val_loss": last_loss if last_loss is not None else "",
                "last_iteration": last_it if last_it is not None else "",
                "metrics_dir": run_dir.resolve().relative_to(PROJECT_ROOT).as_posix(),
                "status": status,
            }
        )

    def _sort_key(x: dict) -> tuple:
        ba = x["best_val_acc"]
        if ba == "":
            return (1, 0.0)
        return (0, -float(ba))

    rows_out.sort(key=_sort_key)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    expected = grid_size()
    ok_rows = sum(1 for r in rows_out if r.get("status") == "ok")
    print(f"Wrote {len(rows_out)} rows to {out_path} (ok={ok_rows}, expected={expected})")
    if not args.allow_incomplete and ok_rows < expected:
        print(
            f"[aggregate_results] ERROR: grid incomplete (ok={ok_rows} < expected={expected}). "
            f"Finish missing runs first or pass --allow-incomplete.",
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(aggregate())
