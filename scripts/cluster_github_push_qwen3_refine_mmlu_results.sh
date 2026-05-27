#!/usr/bin/env bash
# Aggregate refine tinyMMLU + join -> summary_mmlu.csv; commit lightweight CSVs + push.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REFINE_ROOT="$ROOT/qwen3_autogrid/results_mmlu_refine"

python3 -m deepseek_autogrid.aggregate_mmlu_results --results-root "$REFINE_ROOT"
python3 scripts/join_sft_mmlu_summary.py \
  --sft-summary "$REFINE_ROOT/summary.csv" \
  --mmlu-summary "$REFINE_ROOT/mmlu_summary.csv" \
  --output "$REFINE_ROOT/summary_mmlu.csv"

git add \
  qwen3_autogrid/results_mmlu_refine/mmlu_summary.csv \
  qwen3_autogrid/results_mmlu_refine/summary_mmlu.csv \
  qwen3_autogrid/results_mmlu_refine/.gitignore \
  2>/dev/null || true

if git diff --cached --quiet; then
  echo "[github-push] nothing to commit for refine mmlu summaries"
else
  git commit -m "Qwen3 refine grid: tinyMMLU summary and summary_mmlu.csv (48 runs)."
fi
git push origin HEAD
echo "[github-push] pushed $(git rev-parse --short HEAD)"
