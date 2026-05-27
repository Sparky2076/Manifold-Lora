#!/usr/bin/env bash
# nohup tinyMMLU eval sweep for Qwen3 MMLU coarse grid (after SFT jobs exist).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

export LORA_TYPE="${LORA_TYPE:-default}"
export DEEPSEEK_GRID_CONFIG_MODULE="${DEEPSEEK_GRID_CONFIG_MODULE:-qwen3_autogrid.config}"
if [[ "${LORA_TYPE}" == "mlora" ]]; then
  export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora}"
  export GRID_JOB_PREFIX="${GRID_JOB_PREFIX:-qwen3_mmlu_mlora_tinyeval}"
  _log_default="$PROJECT_DIR/qwen3_mmlu_mlora_tinyeval_submit.log"
  _pid_default="$PROJECT_DIR/qwen3_mmlu_mlora_tinyeval_nohup.pid"
else
  export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/qwen3_autogrid/results_mmlu}"
  export GRID_JOB_PREFIX="${GRID_JOB_PREFIX:-qwen3_mmlu_tinyeval}"
  _log_default="$PROJECT_DIR/qwen3_mmlu_tinyeval_submit.log"
  _pid_default="$PROJECT_DIR/qwen3_mmlu_tinyeval_nohup.pid"
fi

LOG="${QWEN3_MMLU_TINYEVAL_LOG:-$_log_default}"
sed -i 's/\r$//' scripts/*.sh qwen3_autogrid/*.sh 2>/dev/null || true

echo "[qwen3-mmlu-coarse-eval] log=$LOG results=$RESULTS_ROOT lora_type=$LORA_TYPE"
nohup bash "$PROJECT_DIR/scripts/run_qwen3_mmlu_grid_eval_bsub.sh" >>"$LOG" 2>&1 &
echo $! >"${QWEN3_MMLU_TINYEVAL_PIDFILE:-$_pid_default}"
echo "[qwen3-mmlu-coarse-eval] nohup pid=$! pidfile=${QWEN3_MMLU_TINYEVAL_PIDFILE:-$_pid_default}"