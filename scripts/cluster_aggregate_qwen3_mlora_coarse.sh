#!/usr/bin/env bash
# Aggregate mLoRA coarse grid into results_mmlu_mlora/summary.csv
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python3 -m deepseek_autogrid.aggregate_results \
  --config-module qwen3_autogrid.config \
  --results-root qwen3_autogrid/results_mmlu_mlora \
  "$@"
