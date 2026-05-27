#!/usr/bin/env python3
"""Collect ``mmlu_eval.json`` (or any eval JSON with ``mmlu_mean_acc`` / ``mean_acc``) under a results tree."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_mean_mmlu(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    for key in ("mmlu_mean_acc", "mean_acc"):
        v = payload.get(key)
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            return f
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate mmlu_eval.json -> mmlu_summary.csv")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None, help="Default: <results-root>/mmlu_summary.csv")
    p.add_argument(
        "--eval-json-name",
        type=str,
        default="mmlu_eval.json",
        help="Per-run JSON to scan (default: mmlu_eval.json; full MMLU: mmlu_eval_full.json).",
    )
    args = p.parse_args()

    root = args.results_root.resolve()
    out = args.output or (root / "mmlu_summary.csv")
    rows: list[dict[str, object]] = []

    eval_name = args.eval_json_name.strip() or "mmlu_eval.json"
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        jpath = run_dir / eval_name
        mean_f = _read_mean_mmlu(jpath)
        if mean_f is None:
            continue
        rows.append(
            {
                "run_name": run_dir.name,
                "mmlu_mean_acc": mean_f,
                "metrics_dir": str(run_dir.resolve()),
                "mmlu_eval_json": str(jpath.resolve()),
            }
        )

    rows.sort(key=lambda r: float(r["mmlu_mean_acc"]), reverse=True)
    fieldnames = ["run_name", "mmlu_mean_acc", "metrics_dir", "mmlu_eval_json"]
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
