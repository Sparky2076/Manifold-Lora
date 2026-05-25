#!/usr/bin/env python3
"""Summarize strictly-complete Qwen3 coarse grid runs on cluster."""
from __future__ import annotations

import csv
import importlib
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
m = importlib.import_module("qwen3_autogrid.config")
results = ROOT / "qwen3_autogrid" / "results_mmlu"
st, ev = m.MAX_STEPS_DEFAULT, m.EVAL_EVERY_DEFAULT
exp = st // ev
rows = []
for lr, r, a, wd in m.iter_grid():
    d = results / m.run_dir_name(lr, r, a, st, wd)
    pt, csvp = d / "sft_lora_state.pt", d / "test_sft.csv"
    if not (pt.is_file() and csvp.is_file()):
        continue
    recs = list(csv.DictReader(csvp.open(encoding="utf-8")))
    if len(recs) < exp:
        continue
    max_it = max(int(float(r["iteration"])) for r in recs)
    if max_it < st:
        continue
    ppls = [float(x["eval_perplexity"]) for x in recs]
    losses = [float(x["eval_loss"]) for x in recs]
    ok = all(math.isfinite(p) and math.isfinite(l) for p, l in zip(ppls, losses))
    best_i = min(range(len(ppls)), key=lambda i: ppls[i])
    rows.append(
        {
            "name": d.name,
            "lr": lr,
            "r": r,
            "a": a,
            "wd": wd,
            "best_ppl": ppls[best_i],
            "best_step": int(recs[best_i]["iteration"]),
            "last_ppl": ppls[-1],
            "last_step": max_it,
            "pt_mb": pt.stat().st_size / 1e6,
            "curve": " ".join(f"{int(recs[i]['iteration'])}:{ppls[i]:.3f}" for i in range(len(recs))),
            "ok": ok,
        }
    )
rows.sort(key=lambda x: x["best_ppl"])
print(f"strict_complete={len(rows)}/45")
print("rank\tbest_ppl\tlast_ppl\tlr\tr\ta\twd\tcurve(st:ppl)")
for i, r in enumerate(rows, 1):
    a = int(r["a"]) if r["a"] == int(r["a"]) else r["a"]
    print(
        f"{i}\t{r['best_ppl']:.4f}\t{r['last_ppl']:.4f}\t{r['lr']:.4g}\t{r['r']}\t{a}\t{r['wd']}\t{r['curve']}"
    )
if rows:
    bp = [r["best_ppl"] for r in rows]
    lp = [r["last_ppl"] for r in rows]
    print("---")
    print(f"best_ppl: min={min(bp):.4f} max={max(bp):.4f} mean={sum(bp)/len(bp):.4f}")
    print(f"last_ppl: min={min(lp):.4f} max={max(lp):.4f} mean={sum(lp)/len(lp):.4f}")
    print(f"all_finite: {all(r['ok'] for r in rows)}")
    print(f"best_overall: {rows[0]['name']} ppl={rows[0]['best_ppl']:.4f}")
