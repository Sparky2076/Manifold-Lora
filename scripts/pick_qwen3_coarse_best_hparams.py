#!/usr/bin/env python3
"""Print env exports for smoke/refine from coarse summary (min best_eval_perplexity)."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("qwen3_autogrid/results_mmlu/summary.csv"),
    )
    p.add_argument("--top-k", type=int, default=3)
    args = p.parse_args()
    rows = []
    with args.summary_csv.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("status") or "").strip() != "ok":
                continue
            try:
                ppl = float(r["best_eval_perplexity"])
            except (KeyError, ValueError):
                continue
            if ppl != ppl:
                continue
            rows.append((ppl, r))
    if not rows:
        print(f"[pick_qwen3_coarse_best] no ok rows in {args.summary_csv}", file=sys.stderr)
        return 1
    rows.sort(key=lambda x: x[0])
    best = rows[0][1]
    print(f"# best_eval_ppl={rows[0][0]:.6f} run={best.get('metrics_dir','')}")
    for i, (ppl, r) in enumerate(rows[: args.top_k], 1):
        print(
            f"# top{i} ppl={ppl:.4f} lr={r.get('lr')} r={r.get('lora_r')} "
            f"a={r.get('lora_alpha')} wd={r.get('weight_decay')}"
        )
    print(f"export SMOKE_LR={best['lr']}")
    print(f"export SMOKE_LORA_R={best['lora_r']}")
    print(f"export SMOKE_LORA_ALPHA={best['lora_alpha']}")
    print(f"export SMOKE_WEIGHT_DECAY={best['weight_decay']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
