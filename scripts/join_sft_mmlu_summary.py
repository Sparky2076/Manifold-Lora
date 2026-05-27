#!/usr/bin/env python3
"""Left-join aggregate SFT ``summary.csv`` with ``mmlu_summary.csv`` on run directory name."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Join summary.csv + mmlu_summary.csv -> summary_mmlu.csv")
    ap.add_argument("--sft-summary", type=Path, required=True)
    ap.add_argument("--mmlu-summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    mmlu_by_run: dict[str, dict[str, str]] = {}
    with args.mmlu_summary.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = (row.get("run_name") or "").strip()
            if name:
                mmlu_by_run[name] = row

    with args.sft_summary.open(newline="", encoding="utf-8") as f:
        sft_rows = list(csv.DictReader(f))

    if not sft_rows:
        print("Empty SFT summary", file=sys.stderr)
        return 2

    fieldnames = list(sft_rows[0].keys())
    extras = ["mmlu_mean_acc", "mmlu_eval_json"]
    for x in extras:
        if x not in fieldnames:
            fieldnames.append(x)

    out_rows: list[dict[str, str]] = []
    for r in sft_rows:
        md = (r.get("metrics_dir") or "").replace("\\", "/").rstrip("/")
        run_key = md.split("/")[-1]
        mj = mmlu_by_run.get(run_key, {})
        new_r = dict(r)
        acc = mj.get("mmlu_mean_acc", "")
        new_r["mmlu_mean_acc"] = acc if acc is not None else ""
        new_r["mmlu_eval_json"] = mj.get("mmlu_eval_json", "") if mj else ""
        out_rows.append(new_r)

    def _sort_key(row: dict[str, str]):
        raw = (row.get("mmlu_mean_acc") or "").strip()
        try:
            return -float(raw)
        except ValueError:
            return 1.0

    out_rows.sort(key=_sort_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Wrote {len(out_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
