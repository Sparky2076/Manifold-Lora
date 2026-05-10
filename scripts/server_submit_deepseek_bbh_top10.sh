#!/usr/bin/env bash
# Submit the selected 10 DeepSeek LoRA runs to BBH evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export TASKS="${TASKS:-bbh}"
export NUM_FEWSHOT="${NUM_FEWSHOT:-3}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
export EVAL_LIMIT="${EVAL_LIMIT:-0}"

RUNS=(
  lr_3p0000e-04_r32_a32_st500_wd_0p0000e00
  lr_3p0000e-04_r32_a32_st500_wd_1p0000e-02
  lr_3p0000e-04_r64_a32_st500_wd_1p0000e-02
  lr_3p0000e-04_r64_a16_st500_wd_1p0000e-02
  lr_3p0000e-04_r32_a64_st500_wd_1p0000e-02
  lr_3p0000e-04_r64_a16_st500_wd_0p0000e00
  lr_3p0000e-04_r64_a32_st500_wd_0p0000e00
  lr_3p0000e-04_r32_a16_st500_wd_1p0000e-02
  lr_3p0000e-05_r32_a64_st1200_wd_0p0000e00
  lr_3p0000e-05_r32_a64_st1200_wd_1p0000e-02
)

echo "[bbh-top10] submitting ${#RUNS[@]} jobs from $PROJECT_DIR"
for RUN in "${RUNS[@]}"; do
  export JOB_NAME="bbh_${RUN}"
  ADAPTER_ROOT="${ADAPTER_ROOT:-$PROJECT_DIR/deepseek_autogrid/results}"
  export METRICS_DIR="${ADAPTER_ROOT%/}/${RUN}"
  export ADAPTER_PATH="$METRICS_DIR"
  export OUTPUT_JSON="${METRICS_DIR}/bbh_eval.json"
  echo "[bbh-top10] submit ${RUN}"
  bash scripts/server_submit_deepseek_bbh.sh
done

echo "[bbh-top10] done"
