#!/usr/bin/env python3
"""Strict completion scan for Qwen3 coarse grid (45 combos, st500)."""
import csv
import importlib
import math
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
m = importlib.import_module("qwen3_autogrid.config")
ROOT = PROJECT / "qwen3_autogrid" / "results_mmlu"
MAX_STEPS = int(m.MAX_STEPS_DEFAULT)
EVAL_EVERY = int(m.EVAL_EVERY_DEFAULT)
expected_rows = MAX_STEPS // EVAL_EVERY


def is_complete(d: Path):
    ckpt = d / "sft_lora_state.pt"
    csvf = d / "test_sft.csv"
    if not ckpt.exists() or not csvf.exists():
        return False, "missing files"
    with csvf.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < expected_rows:
        return False, f"rows={len(rows)}/{expected_rows}"
    it = max(int(float(r.get("iteration", r.get("step", 0)))) for r in rows)
    if it < MAX_STEPS:
        return False, f"max_iter={it}"
    last = max(rows, key=lambda r: int(float(r.get("iteration", r.get("step", 0)))))
    for key in ("eval_perplexity", "eval_ppl", "ppl", "val_ppl"):
        if key in last and last[key] not in ("", "nan", "NaN"):
            ppl = float(last[key])
            if not math.isfinite(ppl):
                return False, f"ppl={ppl}"
            break
    else:
        return False, "missing eval ppl"
    return True, "ok"


def main():
    expected = [m.run_dir_name(lr, r, alpha, MAX_STEPS, wd) for lr, r, alpha, wd in m.iter_grid()]
    complete, incomplete = [], []
    for name in expected:
        d = ROOT / name
        ok, msg = is_complete(d) if d.is_dir() else (False, "no dir")
        (complete if ok else incomplete).append((name, msg))
    print(f"COMPLETE={len(complete)}/{len(expected)} INCOMPLETE={len(incomplete)}")
    for name, msg in incomplete:
        print(f"INCOMPLETE: {name} ({msg})")
    return 0 if len(complete) == len(expected) else 1


if __name__ == "__main__":
    sys.exit(main())
