#!/usr/bin/env python3
"""Print Top-K run directory names from deepseek_autogrid summary.csv (by eval perplexity)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Pick Top-K DeepSeek grid runs by best_eval_perplexity.")
    p.add_argument("--summary-csv", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--require-status-ok", action="store_true", help="Only rows with status == ok")
    p.add_argument("--lora-type", type=str, default="default", help="Filter meta lora_type (default: default)")
    args = p.parse_args()

    rows = list(csv.DictReader(args.summary_csv.open(newline="", encoding="utf-8")))
    filtered = []
    for r in rows:
        if args.require_status_ok and (r.get("status") or "").strip() != "ok":
            continue
        if (r.get("lora_type") or "default").strip() != args.lora_type.strip():
            continue
        md = (r.get("metrics_dir") or "").replace("\\", "/").rstrip("/")
        name = md.split("/")[-1] if md else ""
        if not name:
            continue
        try:
            ppl = float(r.get("best_eval_perplexity") or "")
        except ValueError:
            continue
        filtered.append((ppl, name))

    filtered.sort(key=lambda x: x[0])
    for _, name in filtered[: max(args.top_k, 0)]:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
