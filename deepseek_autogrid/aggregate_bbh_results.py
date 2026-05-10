#!/usr/bin/env python3
"""Collect ``bbh_eval.json`` under a DeepSeek results tree into ``bbh_summary.csv``."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate bbh_eval.json files into bbh_summary.csv")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="Default: <results-root>/bbh_summary.csv")
    args = p.parse_args()

    root = args.results_root.resolve()
    out = args.output or (root / "bbh_summary.csv")
    rows: list[dict[str, object]] = []

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        payload = _read_json(run_dir / "bbh_eval.json")
        if not payload:
            continue
        mean_acc = payload.get("bbh_mean_acc")
        if mean_acc is None:
            continue
        try:
            mean_f = float(mean_acc)
        except (TypeError, ValueError):
            continue
        if mean_f != mean_f:  # NaN
            continue
        task_acc = payload.get("bbh_task_acc") or {}
        n_tasks = len(task_acc) if isinstance(task_acc, dict) else 0
        rows.append(
            {
                "run_name": run_dir.name,
                "bbh_mean_acc": mean_f,
                "bbh_tasks": n_tasks,
                "metrics_dir": str(run_dir.resolve()),
                "bbh_eval_json": str((run_dir / "bbh_eval.json").resolve()),
            }
        )

    rows.sort(key=lambda r: float(r["bbh_mean_acc"]), reverse=True)
    fieldnames = ["run_name", "bbh_mean_acc", "bbh_tasks", "metrics_dir", "bbh_eval_json"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
