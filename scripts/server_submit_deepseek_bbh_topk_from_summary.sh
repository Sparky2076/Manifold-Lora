#!/usr/bin/env bash
# Phase 2: submit BBH (lm-eval) for Top-K runs from phase-1 summary.csv (LoRA grid).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_DIR/deepseek_autogrid/results/summary.csv}"
TOP_K="${TOP_K:-10}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$PROJECT_DIR/deepseek_autogrid/results}"
RESULTS_ROOT="${RESULTS_ROOT:-$ADAPTER_ROOT}"
EXPORT_MERGED_FIRST="${EXPORT_MERGED_FIRST:-1}"

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    [[ -f "$_try/etc/profile.d/conda.sh" ]] && { export CONDA_ROOT="$_try"; break; }
  done
fi

echo "[bbh-topk] summary=$SUMMARY_CSV top_k=$TOP_K results_root=$RESULTS_ROOT"

if [[ "${EXPORT_MERGED_FIRST}" == "1" ]]; then
  echo "[bbh-topk] exporting model_merged_hf for Top-K (set EXPORT_MERGED_FIRST=0 to skip)"
  SUMMARY_CSV="$SUMMARY_CSV" TOP_K="$TOP_K" RESULTS_ROOT="$RESULTS_ROOT" bash "$PROJECT_DIR/scripts/export_merged_deepseek_topk.sh"
fi

while IFS= read -r run; do
  [[ -z "${run}" ]] && continue
  export JOB_NAME="bbh_${run}"
  export METRICS_DIR="${RESULTS_ROOT%/}/${run}"
  export ADAPTER_PATH="$METRICS_DIR"
  export OUTPUT_JSON="${METRICS_DIR}/bbh_eval.json"
  export SKIP_MERGE=1
  export MERGED_HF_DIR="${METRICS_DIR}/model_merged_hf"
  echo "[bbh-topk] submit BBH (SKIP_MERGE=1) $run"
  bash scripts/server_submit_deepseek_bbh.sh
done < <(python "$PROJECT_DIR/scripts/pick_topk_deepseek_runs.py" --summary-csv "$SUMMARY_CSV" --top-k "$TOP_K" --require-status-ok --lora-type default)

echo "[bbh-topk] done"
