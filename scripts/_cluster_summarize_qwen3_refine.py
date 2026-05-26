#!/usr/bin/env python3
"""Summarize strictly-complete Qwen3 refine grid runs on cluster."""
from __future__ import annotations

import csv
import importlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
m = importlib.import_module("qwen3_autogrid.config_refine")
st, ev = m.MAX_STEPS_DEFAULT, m.EVAL_EVERY_DEFAULT
exp = st // ev

_CANDIDATE_ROOTS = (
    ROOT / "qwen3_autogrid" / "results_mmlu_refine",
    ROOT / "qwen3_autogrid" / "results_refine",
)


def _pick_results_root() -> Path:
    for p in _CANDIDATE_ROOTS:
        if p.is_dir() and any(p.iterdir()):
            return p
    return _CANDIDATE_ROOTS[0]


def main() -> int:
    results = _pick_results_root()
    print(f"results_root={results}")
    rows = []
    for lr, r, a, wd in m.iter_grid():
        d = results / m.run_dir_name(lr, r, a, st, wd)
        pt, csvp = d / "sft_lora_state.pt", d / "test_sft.csv"
        if pt.is_file() and csvp.is_file():
            recs = list(csv.DictReader(csvp.open(encoding="utf-8")))
            max_it = max(int(float(x["iteration"])) for x in recs) if recs else 0
            ppls = [float(x["eval_perplexity"]) for x in recs]
            ok = len(recs) >= exp and max_it >= st and all(math.isfinite(p) for p in ppls)
            if ok:
                best_i = min(range(len(ppls)), key=lambda i: ppls[i])
                rows.append(
                    (ppls[best_i], lr, r, a, wd, ppls[best_i], ppls[-1], int(recs[best_i]["iteration"]), d.name, "complete")
                )
            else:
                rows.append((1e9, lr, r, a, wd, None, None, max_it, d.name, f"partial({len(recs)}/{exp},max{max_it})"))
        elif d.exists() and any(d.iterdir()):
            rows.append((1e9, lr, r, a, wd, None, None, 0, d.name, "partial_files"))
        else:
            rows.append((1e9, lr, r, a, wd, None, None, 0, d.name, "empty"))

    complete = [r for r in rows if r[9] == "complete"]
    print(f"strict_complete={len(complete)}/48")
    print(f"partial={sum(1 for r in rows if r[9].startswith('partial'))} empty={sum(1 for r in rows if r[9]=='empty')}")
    complete.sort(key=lambda x: x[0])
    print("rank\tbest_ppl\tlast_ppl\tlr\tr\ta\twd\tname")
    for i, (_, lr, r, a, wd, bp, lp, _, name, _) in enumerate(complete[:15], 1):
        ai = int(a) if a == int(a) else a
        print(f"{i}\t{bp:.4f}\t{lp:.4f}\t{lr:.4g}\t{r}\t{ai}\t{wd}\t{name[:48]}")
    if complete:
        bp = [x[0] for x in complete]
        print(f"--- min={min(bp):.4f} max={max(bp):.4f} mean={sum(bp)/len(bp):.4f} all_finite=True")
    print("--- incomplete (first 10) ---")
    for r in [x for x in rows if x[9] != "complete"][:10]:
        print(r[9], r[8][:52])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
