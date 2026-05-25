#!/usr/bin/env bash
# Aggregate coarse grid + commit summary/analysis + push to origin (login node or local).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m deepseek_autogrid.aggregate_results --config-module qwen3_autogrid.config --allow-incomplete || true
python3 -m deepseek_autogrid.analyze_results \
  --config-module qwen3_autogrid.config \
  --summary qwen3_autogrid/results_mmlu/summary.csv \
  --output qwen3_autogrid/results_mmlu/deepseek_grid_analysis.md \
  --allow-incomplete || true

git add \
  qwen3_autogrid/config.py \
  qwen3_autogrid/results_mmlu/summary.csv \
  qwen3_autogrid/results_mmlu/deepseek_grid_analysis.md \
  qwen3_autogrid/results_mmlu/.gitignore \
  2>/dev/null || true

if git diff --cached --quiet; then
  echo "[github-push] nothing to commit for coarse summary"
else
  git commit -m "$(cat <<'EOF'
Qwen3 coarse LoRA grid: aggregate results_mmlu summary and analysis.

EOF
)"
fi
git push origin HEAD
echo "[github-push] pushed $(git rev-parse --short HEAD)"
