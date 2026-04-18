#!/usr/bin/env bash
# DeepSeek SFT full-factor grid submit (lr × r × alpha × wd)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

LORA_TYPE="${LORA_TYPE:-default}"
if [[ -z "${RESULTS_ROOT:-}" ]]; then
  if [[ "${LORA_TYPE}" == "mlora" ]]; then
    RESULTS_ROOT="$PROJECT_DIR/deepseek_autogrid/results_mlora"
  else
    RESULTS_ROOT="$PROJECT_DIR/deepseek_autogrid/results"
  fi
fi

GRID_RESUME="${GRID_RESUME:-1}"
# 与 DistilBERT 网格一致：默认限制 PEND，避免连点 bsub 触发站点「Pending 上限 / User permission denied」后整脚本退出。
# GRID_MAX_PEND=1 → 仅当 PEND=0 时才再交下一单；GRID_MAX_RUN=0 表示不限制 RUN（可自行设 5 等）。
GRID_MAX_RUN="${GRID_MAX_RUN:-0}"
GRID_MAX_PEND="${GRID_MAX_PEND:-1}"
GRID_POLL_SEC="${GRID_POLL_SEC:-30}"
SUBMIT_SLEEP_SEC="${SUBMIT_SLEEP_SEC:-180}"
GRID_MAX_PASSES="${GRID_MAX_PASSES:-0}"

export MAX_STEPS="${MAX_STEPS:-$(python -c "from deepseek_autogrid.config import MAX_STEPS_DEFAULT; print(MAX_STEPS_DEFAULT)")}"
export EVAL_EVERY="${EVAL_EVERY:-$(python -c "from deepseek_autogrid.config import EVAL_EVERY_DEFAULT; print(EVAL_EVERY_DEFAULT)")}"
export SFT_PRESET="${SFT_PRESET:-$(python -c "from deepseek_autogrid.config import SFT_PRESET_DEFAULT; print(SFT_PRESET_DEFAULT)")}"
export SFT_VAL_RATIO="${SFT_VAL_RATIO:-$(python -c "from deepseek_autogrid.config import SFT_VAL_RATIO_DEFAULT; print(SFT_VAL_RATIO_DEFAULT)")}"
export ADAM_BETA1="${ADAM_BETA1:-$(python -c "from deepseek_autogrid.config import ADAM_BETA1_FIXED; print(ADAM_BETA1_FIXED)")}"
export ADAM_BETA2="${ADAM_BETA2:-$(python -c "from deepseek_autogrid.config import ADAM_BETA2_FIXED; print(ADAM_BETA2_FIXED)")}"

_grid_wait_slot() {
  [[ "$GRID_MAX_RUN" == "0" && "$GRID_MAX_PEND" == "0" ]] && return 0
  command -v bjobs >/dev/null 2>&1 || return 0
  local u="${USER:-${LOGNAME:-}}"
  [[ -z "$u" ]] && return 0
  while true; do
    local run_n pend_n need_wait=0
    run_n=$(bjobs -u "$u" 2>/dev/null | awk 'NR>1 && $3 ~ /^RUN/ {c++} END{print c+0}')
    pend_n=$(bjobs -u "$u" 2>/dev/null | awk 'NR>1 && $3 ~ /^PEND/ {c++} END{print c+0}')
    [[ "$GRID_MAX_RUN" =~ ^[0-9]+$ && "$GRID_MAX_RUN" -gt 0 && "$run_n" -gt "$GRID_MAX_RUN" ]] && need_wait=1
    [[ "$GRID_MAX_PEND" =~ ^[0-9]+$ && "$GRID_MAX_PEND" -gt 0 && "$pend_n" -ge "$GRID_MAX_PEND" ]] && need_wait=1
    [[ "$need_wait" -eq 0 ]] && return 0
    echo "[deepseek-grid] throttle RUN=${run_n} PEND=${pend_n}, sleep ${GRID_POLL_SEC}s ..." >&2
    sleep "$GRID_POLL_SEC"
  done
}

_grid_pending_jobs() {
  command -v bjobs >/dev/null 2>&1 || { echo 0; return 0; }
  local u="${USER:-${LOGNAME:-}}"
  [[ -z "$u" ]] && { echo 0; return 0; }
  bjobs -u "$u" 2>/dev/null | awk 'NR>1 && $3 ~ /^(RUN|PEND)$/ && $7 ~ /^deepseek_grid_/ {c++} END{print c+0}'
}

_grid_wait_all_done() {
  command -v bjobs >/dev/null 2>&1 || return 0
  while true; do
    local n
    n="$(_grid_pending_jobs)"
    [[ "$n" -le 0 ]] && return 0
    echo "[deepseek-grid] waiting RUN/PEND deepseek_grid_*: $n (sleep ${GRID_POLL_SEC}s) ..." >&2
    sleep "$GRID_POLL_SEC"
  done
}

_grid_is_complete() {
  local md="$1"
  [[ -f "$md/test_sft.csv" ]] || return 1
  local lines
  lines=$(wc -l < "$md/test_sft.csv" | tr -d ' \t')
  [[ $((lines - 1)) -ge 1 ]]
}

pass=0
while true; do
  pass=$((pass + 1))
  need_n=0
  submit_n=0
  echo "[deepseek-grid] pass ${pass}: scan + submit missing/incomplete combos ..." >&2

  while IFS=$'\t' read -r NAME LR R A WD; do
    METRICS_DIR="$RESULTS_ROOT/$NAME"
    if [[ "$GRID_RESUME" != "0" ]] && _grid_is_complete "$METRICS_DIR"; then
      continue
    fi
    need_n=$((need_n + 1))

    export JOB_NAME="deepseek_grid_${NAME}"
    export LR="$LR" LORA_R="$R" LORA_ALPHA="$A"
    export WEIGHT_DECAY="$WD"
    export METRICS_DIR
    export LORA_TYPE
    export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
    export BATCH_SIZE="${BATCH_SIZE:-2}"
    export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
    export MAX_LENGTH="${MAX_LENGTH:-512}"
    export TORCH_DTYPE="${TORCH_DTYPE:-float32}"

    # bsub 失败（如 User permission denied）时只等待重试，不退出整个网格循环
    while true; do
      _grid_wait_slot
      if bash deepseek/scripts/submit_bsub_sft.sh; then
        break
      fi
      echo "[deepseek-grid] bsub failed; sleep ${GRID_POLL_SEC}s then re-check slot and retry ..." >&2
      sleep "$GRID_POLL_SEC"
    done
    submit_n=$((submit_n + 1))
    sleep "$SUBMIT_SLEEP_SEC"
  done < <(python -c "
import os
from deepseek_autogrid.config import iter_grid, run_dir_name, MAX_STEPS_DEFAULT
st = int(os.environ.get('MAX_STEPS', str(MAX_STEPS_DEFAULT)))
for lr, r, alpha, wd in iter_grid():
    print(f'{run_dir_name(lr, r, alpha, st, wd)}\\t{lr}\\t{r}\\t{alpha}\\t{wd}')
")

  if [[ "$need_n" -eq 0 ]]; then
    echo "[deepseek-grid] all combos complete (MAX_STEPS=${MAX_STEPS})."
    break
  fi
  echo "[deepseek-grid] pass ${pass}: submitted ${submit_n}, pending completeness scan..." >&2
  _grid_wait_all_done
  echo "[deepseek-grid] pass ${pass}: run queue drained, rescan..." >&2

  if [[ "$GRID_MAX_PASSES" =~ ^[0-9]+$ && "$GRID_MAX_PASSES" -gt 0 && "$pass" -ge "$GRID_MAX_PASSES" ]]; then
    echo "[deepseek-grid] stop: GRID_MAX_PASSES=${GRID_MAX_PASSES} reached; may still be incomplete." >&2
    break
  fi
done

echo "Done. DeepSeek grid loop finished; results under $RESULTS_ROOT"
