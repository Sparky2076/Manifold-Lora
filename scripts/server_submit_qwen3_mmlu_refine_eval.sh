#!/usr/bin/env bash
# nohup tinyMMLU eval sweep for Qwen3 refine grid (48 SFT runs under results_mmlu_refine/).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck source=scripts/_cluster_lsf_env.sh
source "$PROJECT_DIR/scripts/_cluster_lsf_env.sh"

export LORA_TYPE="${LORA_TYPE:-default}"
export DEEPSEEK_GRID_CONFIG_MODULE="${DEEPSEEK_GRID_CONFIG_MODULE:-qwen3_autogrid.config_refine}"
if [[ "${LORA_TYPE}" == "mlora" ]]; then
  export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora_refine}"
  export GRID_JOB_PREFIX="${GRID_JOB_PREFIX:-qwen3_mmlu_mlora_refine_tinyeval}"
  export GRID_PID_FILE="${GRID_PID_FILE:-$PROJECT_DIR/qwen3_autogrid/.grid_submitter_mmlu_mlora_refine_tinyeval.pid}"
  _log_default="$PROJECT_DIR/qwen3_mmlu_mlora_refine_tinyeval_submit.log"
  _pid_default="$PROJECT_DIR/qwen3_mmlu_mlora_refine_tinyeval_nohup.pid"
else
  export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_refine}"
  export GRID_JOB_PREFIX="${GRID_JOB_PREFIX:-qwen3_mmlu_refine_tinyeval}"
  export GRID_PID_FILE="${GRID_PID_FILE:-$PROJECT_DIR/qwen3_autogrid/.grid_submitter_mmlu_refine_tinyeval.pid}"
  _log_default="$PROJECT_DIR/qwen3_mmlu_refine_tinyeval_submit.log"
  _pid_default="$PROJECT_DIR/qwen3_mmlu_refine_tinyeval_nohup.pid"
fi

LOG="${QWEN3_MMLU_REFINE_TINYEVAL_LOG:-$_log_default}"
sed -i 's/\r$//' scripts/*.sh qwen3_autogrid/*.sh 2>/dev/null || true

export GRID_MAX_RUN="${GRID_MAX_RUN:-5}"
export GRID_MAX_PEND="${GRID_MAX_PEND:-1}"
export GRID_POLL_SEC="${GRID_POLL_SEC:-15}"
export SUBMIT_SLEEP_SEC="${SUBMIT_SLEEP_SEC:-2}"
export EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-gpu15,gpu17,gpu18}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
unset MAX_STEPS EVAL_EVERY

echo "[qwen3-mmlu-refine-eval] log=$LOG results=$RESULTS_ROOT prefix=$GRID_JOB_PREFIX lora_type=$LORA_TYPE"
export DEEPSEEK_GRID_CONFIG_MODULE RESULTS_ROOT GRID_JOB_PREFIX GRID_PID_FILE
export GRID_MAX_RUN GRID_MAX_PEND GRID_POLL_SEC SUBMIT_SLEEP_SEC EXCLUDE_HOSTS TRUST_REMOTE_CODE
nohup bash "$PROJECT_DIR/scripts/run_qwen3_mmlu_grid_eval_bsub.sh" >>"$LOG" 2>&1 &
echo $! >"${QWEN3_MMLU_REFINE_TINYEVAL_PIDFILE:-$_pid_default}"
echo "[qwen3-mmlu-refine-eval] nohup pid=$! pidfile=${QWEN3_MMLU_REFINE_TINYEVAL_PIDFILE:-$_pid_default}"
