#!/usr/bin/env bash
# Phase 2: submit BBH (lm-eval) for Top-K runs from phase-1 summary.csv (LoRA grid).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_DIR/deepseek_autogrid/results/summary.csv}"
TOP_K="${TOP_K:-10}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$PROJECT_DIR/deepseek_autogrid/results}"

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    [[ -f "$_try/etc/profile.d/conda.sh" ]] && { export CONDA_ROOT="$_try"; break; }
  done
fi

echo "[bbh-topk] summary=$SUMMARY_CSV top_k=$TOP_K adapter_root=$ADAPTER_ROOT"

while IFS= read -r run; do
  [[ -z "${run}" ]] && continue
  export JOB_NAME="bbh_${run}"
  export METRICS_DIR="${ADAPTER_ROOT%/}/${run}"
  export ADAPTER_PATH="$METRICS_DIR"
  export OUTPUT_JSON="${METRICS_DIR}/bbh_eval.json"
  echo "[bbh-topk] submit $run"
  bash scripts/server_submit_deepseek_bbh.sh
done < <(python "$PROJECT_DIR/scripts/pick_topk_deepseek_runs.py" --summary-csv "$SUMMARY_CSV" --top-k "$TOP_K" --require-status-ok --lora-type default)

echo "[bbh-topk] done"
