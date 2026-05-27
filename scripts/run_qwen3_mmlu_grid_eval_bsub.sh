#!/usr/bin/env bash
# Submit tinyMMLU lm-eval jobs for each completed Qwen3 MMLU coarse SFT run missing mmlu_eval.json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/scripts/_cluster_lsf_env.sh" ]]; then
  # shellcheck source=scripts/_cluster_lsf_env.sh
  source "$PROJECT_DIR/scripts/_cluster_lsf_env.sh"
fi

export DEEPSEEK_GRID_CONFIG_MODULE="${DEEPSEEK_GRID_CONFIG_MODULE:-qwen3_autogrid.config}"

LORA_TYPE="${LORA_TYPE:-default}"
if [[ -z "${RESULTS_ROOT:-}" ]]; then
  if [[ "${LORA_TYPE}" == "mlora" ]]; then
    RESULTS_ROOT="$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora"
  else
    RESULTS_ROOT="$PROJECT_DIR/qwen3_autogrid/results_mmlu"
  fi
fi

GRID_JOB_PREFIX="${GRID_JOB_PREFIX:-qwen3_mmlu_tinyeval}"
GRID_RESUME="${GRID_RESUME:-1}"
GRID_MAX_RUN="${GRID_MAX_RUN:-0}"
GRID_MAX_PEND="${GRID_MAX_PEND:-1}"
GRID_POLL_SEC="${GRID_POLL_SEC:-30}"
SUBMIT_SLEEP_SEC="${SUBMIT_SLEEP_SEC:-60}"
GRID_MAX_PASSES="${GRID_MAX_PASSES:-0}"

export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    [[ -f "$_try/etc/profile.d/conda.sh" ]] && { export CONDA_ROOT="$_try"; break; }
  done
fi
for _conda_root in "${CONDA_ROOT:-}" "$HOME/miniconda3" "/nfsshare/home/${USER:-$LOGNAME}/miniconda3"; do
  [[ -n "$_conda_root" && -x "$_conda_root/envs/${CONDA_ENV_NAME:-torch}/bin/python3" ]] || continue
  export PATH="$_conda_root/envs/${CONDA_ENV_NAME:-torch}/bin:${PATH}"
  export CONDA_ROOT="$_conda_root"
  break
done

echo "[qwen3-mmlu-tinyeval] config_module=${DEEPSEEK_GRID_CONFIG_MODULE} results=${RESULTS_ROOT} prefix=${GRID_JOB_PREFIX}"

GRID_PID_FILE="${GRID_PID_FILE:-$PROJECT_DIR/qwen3_autogrid/.grid_submitter_mmlu_tinyeval.pid}"
echo $$ >"$GRID_PID_FILE"
trap 'rm -f "$GRID_PID_FILE"' EXIT

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
    echo "[qwen3-mmlu-tinyeval] throttle RUN=${run_n} PEND=${pend_n}, sleep ${GRID_POLL_SEC}s ..." >&2
    sleep "$GRID_POLL_SEC"
  done
}

_grid_pending_eval_jobs() {
  command -v bjobs >/dev/null 2>&1 || { echo 0; return 0; }
  local u="${USER:-${LOGNAME:-}}"
  [[ -z "$u" ]] && { echo 0; return 0; }
  local pfx="$GRID_JOB_PREFIX"
  bjobs -u "$u" 2>/dev/null | awk -v pfx="$pfx" 'NR>1 && $3 ~ /^(RUN|PEND)$/ && $7 ~ ("^" pfx "_") {c++} END{print c+0}'
}

_grid_wait_all_done() {
  command -v bjobs >/dev/null 2>&1 || return 0
  while true; do
    local n
    n="$(_grid_pending_eval_jobs)"
    [[ "$n" -le 0 ]] && return 0
    echo "[qwen3-mmlu-tinyeval] waiting RUN/PEND ${GRID_JOB_PREFIX}_*: $n (sleep ${GRID_POLL_SEC}s) ..." >&2
    sleep "$GRID_POLL_SEC"
  done
}

_sft_complete() {
  local md="$1"
  [[ -f "$md/test_sft.csv" ]] || return 1
  local lines
  lines=$(wc -l < "$md/test_sft.csv" | tr -d ' \t')
  [[ $((lines - 1)) -ge 1 ]]
}

_mmlu_json_ok() {
  local j="$1"
  [[ -f "$j" ]] || return 1
  python3 - "$j" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
for k in ("mmlu_mean_acc", "mean_acc"):
    v = data.get(k)
    if v is None:
        continue
    try:
        x = float(v)
    except (TypeError, ValueError):
        continue
    if x == x:
        sys.exit(0)
sys.exit(1)
PY
}

pass=0
while true; do
  pass=$((pass + 1))
  need_n=0
  submit_n=0
  echo "[qwen3-mmlu-tinyeval] pass ${pass}: scan missing mmlu_eval ..." >&2

  while IFS= read -r NAME; do
    METRICS_DIR="$RESULTS_ROOT/$NAME"
    if ! _sft_complete "$METRICS_DIR"; then
      continue
    fi
    OUT_JSON="${METRICS_DIR}/mmlu_eval.json"
    if [[ "$GRID_RESUME" != "0" ]] && _mmlu_json_ok "$OUT_JSON"; then
      continue
    fi
    need_n=$((need_n + 1))

    export JOB_NAME="${GRID_JOB_PREFIX}_${NAME}"
    export METRICS_DIR
    export ADAPTER_PATH="$METRICS_DIR"
    export OUTPUT_JSON="$OUT_JSON"
    export TASKS="${TASKS:-tinyMMLU}"
    export NUM_FEWSHOT="${NUM_FEWSHOT:-0}"
    export EVAL_LIMIT="${EVAL_LIMIT:-0}"
    export SKIP_MERGE="${SKIP_MERGE:-0}"

    while true; do
      _grid_wait_slot
      if bash "$PROJECT_DIR/scripts/server_submit_deepseek_bbh.sh"; then
        break
      fi
      echo "[qwen3-mmlu-tinyeval] bsub failed; sleep ${GRID_POLL_SEC}s ..." >&2
      sleep "$GRID_POLL_SEC"
    done
    submit_n=$((submit_n + 1))
    sleep "$SUBMIT_SLEEP_SEC"
  done < <(python3 -c "
import os, importlib
m = importlib.import_module(os.environ.get('DEEPSEEK_GRID_CONFIG_MODULE', 'qwen3_autogrid.config'))
st = int(os.environ.get('GRID_MAX_STEPS', str(m.MAX_STEPS_DEFAULT)))
for lr, r, alpha, wd in m.iter_grid():
    print(m.run_dir_name(lr, r, alpha, st, wd))
")

  if [[ "${GRID_RESUME}" == "0" ]]; then
    echo "[qwen3-mmlu-tinyeval] GRID_RESUME=0: one-shot submitted ${submit_n}; draining." >&2
    _grid_wait_all_done
    break
  fi

  if [[ "$need_n" -eq 0 ]]; then
    echo "[qwen3-mmlu-tinyeval] all evals done or waiting on SFT." >&2
    break
  fi
  echo "[qwen3-mmlu-tinyeval] pass ${pass}: submitted ${submit_n}; drain ..." >&2
  _grid_wait_all_done

  if [[ "$GRID_MAX_PASSES" =~ ^[0-9]+$ && "$GRID_MAX_PASSES" -gt 0 && "$pass" -ge "$GRID_MAX_PASSES" ]]; then
    echo "[qwen3-mmlu-tinyeval] GRID_MAX_PASSES reached." >&2
    break
  fi
done

echo "Done. tinyMMLU eval loop finished; root=$RESULTS_ROOT"
