#!/usr/bin/env bash
# Submit full DistilBERT LoRA grid as separate LSF jobs (lr × r × alpha × wd; betas fixed in config).
# Grid definition lives in distilbert_autogrid/config.py only.
#
# Usage (cluster):
#   cd ~/Manifold-Lora && sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
#   bash distilbert_autogrid/run_grid_bsub.sh
#
# Optional env: RESULTS_ROOT, EPOCHS, QUEUE, LORA_TYPE, LORA_DROPOUT, BATCH_SIZE, GRAD_ACCUM_STEPS
# Resume: GRID_RESUME=1 (default) skips combos whose test.csv already has >= EPOCHS eval rows; GRID_RESUME=0 submits all.
# Throttle (avoid cluster Pending limit / permission denied): before each bsub, wait until bjobs ok.
#   GRID_MAX_RUN=N     (0=off) 最多 N 个 RUN → wait while RUN > N（例 N=5：仅当 RUN≤5 时可再交；5 RUN+0 PEND 时仍可交第 6 个进 PEND）
#   GRID_MAX_PEND=M    (0=off) 最多 M 个 PEND → wait while PEND >= M（例 M=1：仅当 PEND=0 时才交，避免连点 bsub 堆出 2 个 PEND）
#   GRID_POLL_SEC=30   seconds between bjobs checks while waiting.
#   SUBMIT_SLEEP_SEC=180  default: sleep 3 minutes after each bsub (reduce same-node GPU pile-up); override per run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/distilbert_autogrid/results}"
GRID_RESUME="${GRID_RESUME:-1}"
GRID_MAX_RUN="${GRID_MAX_RUN:-0}"
GRID_MAX_PEND="${GRID_MAX_PEND:-0}"
GRID_POLL_SEC="${GRID_POLL_SEC:-30}"
SUBMIT_SLEEP_SEC="${SUBMIT_SLEEP_SEC:-180}"

# Wait until bjobs counts allow another submission (LSF: STAT column RUN / PEND).
_grid_wait_slot() {
  [[ "${GRID_MAX_RUN}" == "0" && "${GRID_MAX_PEND}" == "0" ]] && return 0
  command -v bjobs >/dev/null 2>&1 || return 0
  local u="${USER:-${LOGNAME:-}}"
  [[ -z "$u" ]] && return 0
  while true; do
    local run_n pend_n
    run_n=$(bjobs -u "$u" 2>/dev/null | awk 'NR>1 && $3 ~ /^RUN/ {c++} END{print c+0}')
    pend_n=$(bjobs -u "$u" 2>/dev/null | awk 'NR>1 && $3 ~ /^PEND/ {c++} END{print c+0}')
    local need_wait=0
    if [[ "${GRID_MAX_RUN}" =~ ^[0-9]+$ ]] && [[ "${GRID_MAX_RUN}" -gt 0 ]] && [[ "${run_n}" -gt "${GRID_MAX_RUN}" ]]; then
      need_wait=1
    fi
    if [[ "${GRID_MAX_PEND}" =~ ^[0-9]+$ ]] && [[ "${GRID_MAX_PEND}" -gt 0 ]] && [[ "${pend_n}" -ge "${GRID_MAX_PEND}" ]]; then
      need_wait=1
    fi
    if [[ "$need_wait" -eq 0 ]]; then
      return 0
    fi
    echo "[grid] throttle: RUN=${run_n} PEND=${pend_n} (RUN: wait if >${GRID_MAX_RUN}; PEND: wait if >=${GRID_MAX_PEND}) sleep ${GRID_POLL_SEC}s ..." >&2
    sleep "${GRID_POLL_SEC}"
  done
}
export EPOCHS="${EPOCHS:-$(python -c "from distilbert_autogrid.config import EPOCHS_DEFAULT; print(EPOCHS_DEFAULT)")}"
export ADAM_BETA1="${ADAM_BETA1:-$(python -c "from distilbert_autogrid.config import ADAM_BETA1_FIXED; print(ADAM_BETA1_FIXED)")}"
export ADAM_BETA2="${ADAM_BETA2:-$(python -c "from distilbert_autogrid.config import ADAM_BETA2_FIXED; print(ADAM_BETA2_FIXED)")}"

python -c "
import os
from distilbert_autogrid.config import iter_grid, run_dir_name, EPOCHS_DEFAULT
ep = int(os.environ.get('EPOCHS', str(EPOCHS_DEFAULT)))
for lr, r, alpha, wd in iter_grid():
    name = run_dir_name(lr, r, alpha, ep, wd)
    print(f'{name}\t{lr}\t{r}\t{alpha}\t{wd}')
" | while IFS=$'\t' read -r NAME LR R A WD; do
  METRICS_DIR="$RESULTS_ROOT/$NAME"
  export JOB_NAME="distilbert_grid_${NAME}"
  export LR="$LR" LORA_R="$R" LORA_ALPHA="$A" EPOCHS="$EPOCHS"
  export WEIGHT_DECAY="$WD"
  export METRICS_DIR
  export LORA_TYPE="${LORA_TYPE:-default}"
  export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
  export BATCH_SIZE="${BATCH_SIZE:-4}"
  export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

  if [[ "${GRID_RESUME}" != "0" ]] && [[ -f "${METRICS_DIR}/test.csv" ]]; then
    lines=$(wc -l < "${METRICS_DIR}/test.csv" | tr -d ' \t')
    data_rows=$((lines - 1))
    if [[ "${data_rows}" -ge "${EPOCHS}" ]]; then
      echo "[grid] skip (complete: ${data_rows} eval rows >= EPOCHS=${EPOCHS}): ${NAME}" >&2
      continue
    fi
  fi

  _grid_wait_slot
  bash distilbert/scripts/submit_bsub.sh
  sleep "${SUBMIT_SLEEP_SEC}"
done

echo "Done. Submitted grid jobs; results under $RESULTS_ROOT"
