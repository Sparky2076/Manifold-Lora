#!/usr/bin/env bash
# One-shot: refresh topk5 bundle + print full MMLU acc for top-5 runs.
set -euo pipefail
cd ~/Manifold-Lora
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch 2>/dev/null || true

ROOT=qwen3_autogrid/results_mmlu_refine
BUNDLE="${ROOT}/topk5_plot_bundle"

python3 scripts/export_qwen3_topk_plot_bundle.py \
  --summary-csv "${ROOT}/summary_mmlu.csv" \
  --results-root "${ROOT}" \
  --out-dir "${BUNDLE}" \
  --top-k 5

python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path("qwen3_autogrid/results_mmlu_refine")
bundle = root / "topk5_plot_bundle"
summary = bundle / "summary_topk.csv"
rows = list(csv.DictReader(summary.open(newline="", encoding="utf-8")))
out_rows = []
for r in rows:
    md = (r.get("metrics_dir") or "").replace("\\", "/").rstrip("/")
    run = md.split("/")[-1]
    jpath = root / run / "mmlu_eval_full.json"
    full_acc = ""
    if jpath.is_file():
        d = json.loads(jpath.read_text(encoding="utf-8"))
        full_acc = d.get("mmlu_mean_acc", d.get("mean_acc", ""))
    r = dict(r)
    r["run_name"] = run
    r["mmlu_full_mean_acc"] = full_acc
    out_rows.append(r)
    print(f"{run}\ttiny={r.get('mmlu_mean_acc','')}\tfull={full_acc}")

out = root / "summary_topk_full_mmlu.csv"
if out_rows:
    fields = list(out_rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out}")
PY
