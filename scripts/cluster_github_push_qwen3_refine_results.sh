#!/usr/bin/env bash
# Aggregate refine grid + commit summary/analysis + push to origin (login node or local).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m deepseek_autogrid.aggregate_results --config-module qwen3_autogrid.config_refine
python3 -m deepseek_autogrid.analyze_results \
  --config-module qwen3_autogrid.config_refine \
  --summary qwen3_autogrid/results_mmlu_refine/summary.csv \
  --output qwen3_autogrid/results_mmlu_refine/deepseek_grid_analysis.md

git add \
  qwen3_autogrid/config_refine.py \
  qwen3_autogrid/results_mmlu_refine/summary.csv \
  qwen3_autogrid/results_mmlu_refine/deepseek_grid_analysis.md \
  qwen3_autogrid/results_mmlu_refine/.gitignore \
  2>/dev/null || true

if git diff --cached --quiet; then
  echo "[github-push] nothing to commit for refine summary"
else
  git commit -m "$(cat <<'EOF'
Qwen3 refine LoRA grid 48/48: aggregate results_mmlu_refine summary and analysis.

EOF
)"
fi
git push origin HEAD
echo "[github-push] pushed $(git rev-parse --short HEAD)"
