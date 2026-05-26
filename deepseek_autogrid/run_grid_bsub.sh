#!/usr/bin/env bash
# DeepSeek SFT full-factor grid submit (lr × r × alpha × wd)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Login-node nohup may not inherit conda; prefer torch env python for config imports.
for _conda_root in "${CONDA_ROOT:-}" "$HOME/miniconda3" "/nfsshare/home/${USER:-$LOGNAME}/miniconda3"; do
  [[ -n "$_conda_root" && -x "$_conda_root/envs/${CONDA_ENV_NAME:-torch}/bin/python" ]] || continue
  export PATH="$_conda_root/envs/${CONDA_ENV_NAME:-torch}/bin:${PATH}"
  export CONDA_ROOT="$_conda_root"
  break
done

# HF_HOME / HF_DATASETS_CACHE defaults for NFS + bsub job env (scripts/cluster_hf_cache_env.sh)
if ! command -v bsub >/dev/null 2>&1 && [[ -f "$PROJECT_DIR/scripts/_cluster_lsf_env.sh" ]]; then
  # shellcheck source=scripts/_cluster_lsf_env.sh
  source "$PROJECT_DIR/scripts/_cluster_lsf_env.sh"
fi

if [[ -f "$PROJECT_DIR/scripts/cluster_hf_cache_env.sh" ]]; then
  # shellcheck source=scripts/cluster_hf_cache_env.sh
  source "$PROJECT_DIR/scripts/cluster_hf_cache_env.sh"
fi

export DEEPSEEK_GRID_CONFIG_MODULE="${DEEPSEEK_GRID_CONFIG_MODULE:-deepseek_autogrid.config}"
export GRID_JOB_PREFIX="${GRID_JOB_PREFIX:-deepseek_grid}"

# Model / LoRA / trust_remote_code defaults from grid config (wrappers may pre-export env).
_GRID_PY='import importlib, os; m = importlib.import_module(os.environ["DEEPSEEK_GRID_CONFIG_MODULE"])'
export MODEL_NAME="${MODEL_NAME:-$(python -c "${_GRID_PY}; print(getattr(m, \"MODEL_NAME_DEFAULT\", \"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B\"))")}"

_lora_tgt_default="$(python -c "${_GRID_PY}; print(getattr(m, \"LORA_TARGETS_DEFAULT\", \"\") or \"\")")"
if [[ -n "${_lora_tgt_default}" && -z "${LORA_TARGETS:-}" ]]; then
  export LORA_TARGETS="${_lora_tgt_default}"
fi

_tr_default="$(python -c "${_GRID_PY}; print(1 if bool(getattr(m, \"TRUST_REMOTE_CODE_DEFAULT\", False)) else 0)")"
if [[ -z "${TRUST_REMOTE_CODE:-}" && "${_tr_default}" == "1" ]]; then
  export TRUST_REMOTE_CODE=1
fi

LORA_TYPE="${LORA_TYPE:-default}"
if [[ -z "${RESULTS_ROOT:-}" ]]; then
  if [[ "${LORA_TYPE}" == "mlora" ]]; then
    RESULTS_ROOT="$PROJECT_DIR/deepseek_autogrid/results_mlora"
  else
    RESULTS_ROOT="$PROJECT_DIR/deepseek_autogrid/results"
  fi
fi

# LoRA / mLoRA 分文件记录 PID，避免两路网格互相覆盖；便于定位。
if [[ "${LORA_TYPE}" == "mlora" ]]; then
  GRID_PID_FILE="${GRID_PID_FILE:-$PROJECT_DIR/deepseek_autogrid/.grid_submitter_mlora.pid}"
else
  GRID_PID_FILE="${GRID_PID_FILE:-$PROJECT_DIR/deepseek_autogrid/.grid_submitter.pid}"
fi
echo $$ >"$GRID_PID_FILE"
trap 'rm -f "$GRID_PID_FILE"' EXIT
echo "[deepseek-grid] config_module=${DEEPSEEK_GRID_CONFIG_MODULE} submitter_pid=$$ host=$(hostname) lora_type=${LORA_TYPE}  | 定位: bash scripts/grid_submitter_status.sh 或 ps -p $$ -f" >&2

# GRID_RESUME=1（默认）：只补未完成的组合。GRID_RESUME=0：强制对每个组合 bsub（无视已有结果），且**只跑一轮递交**后退出，避免无限重复整网提交。
GRID_RESUME="${GRID_RESUME:-1}"
# DeepSeek 网格：默认限制 PEND，避免连点 bsub 触发站点「Pending 上限 / User permission denied」后整脚本退出。
# GRID_MAX_PEND=1 → 仅当 PEND=0 时才再交下一单；GRID_MAX_RUN=0 表示不限制 RUN（可自行设 5 等）。
GRID_MAX_RUN="${GRID_MAX_RUN:-0}"
GRID_MAX_PEND="${GRID_MAX_PEND:-1}"
GRID_POLL_SEC="${GRID_POLL_SEC:-30}"
SUBMIT_SLEEP_SEC="${SUBMIT_SLEEP_SEC:-180}"
GRID_MAX_PASSES="${GRID_MAX_PASSES:-0}"

# Grid training length always from config module (smoke pipelines may leave MAX_STEPS=2 in env).
if [[ -n "${GRID_MAX_STEPS:-}" ]]; then
  export MAX_STEPS="$GRID_MAX_STEPS"
else
  export MAX_STEPS="$(python -c "import importlib,os;m=importlib.import_module(os.environ['DEEPSEEK_GRID_CONFIG_MODULE']);print(m.MAX_STEPS_DEFAULT)")"
fi
if [[ -n "${GRID_EVAL_EVERY:-}" ]]; then
  export EVAL_EVERY="$GRID_EVAL_EVERY"
else
  export EVAL_EVERY="$(python -c "import importlib,os;m=importlib.import_module(os.environ['DEEPSEEK_GRID_CONFIG_MODULE']);print(m.EVAL_EVERY_DEFAULT)")"
fi
echo "[deepseek-grid] MAX_STEPS=${MAX_STEPS} EVAL_EVERY=${EVAL_EVERY} RESULTS_ROOT=${RESULTS_ROOT:-<default>}" >&2
export SFT_PRESET="${SFT_PRESET:-$(python -c "import importlib,os;m=importlib.import_module(os.environ['DEEPSEEK_GRID_CONFIG_MODULE']);print(m.SFT_PRESET_DEFAULT)")}"
export SFT_VAL_RATIO="${SFT_VAL_RATIO:-$(python -c "import importlib,os;m=importlib.import_module(os.environ['DEEPSEEK_GRID_CONFIG_MODULE']);print(m.SFT_VAL_RATIO_DEFAULT)")}"
export SFT_FORMAT="${SFT_FORMAT:-$(python -c "import importlib,os;m=importlib.import_module(os.environ['DEEPSEEK_GRID_CONFIG_MODULE']);print(getattr(m,'SFT_FORMAT_DEFAULT','chat'))")}"
export ADAM_BETA1="${ADAM_BETA1:-$(python -c "import importlib,os;m=importlib.import_module(os.environ['DEEPSEEK_GRID_CONFIG_MODULE']);print(m.ADAM_BETA1_FIXED)")}"
export ADAM_BETA2="${ADAM_BETA2:-$(python -c "import importlib,os;m=importlib.import_module(os.environ['DEEPSEEK_GRID_CONFIG_MODULE']);print(m.ADAM_BETA2_FIXED)")}"

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
  local pfx="${GRID_JOB_PREFIX:-deepseek_grid}"
  bjobs -u "$u" 2>/dev/null | awk -v pfx="$pfx" 'NR>1 && $3 ~ /^(RUN|PEND)$/ && $7 ~ ("^" pfx "_") {c++} END{print c+0}'
}

_grid_job_in_queue() {
  local jname="$1"
  command -v bjobs >/dev/null 2>&1 || return 1
  local u="${USER:-${LOGNAME:-}}"
  [[ -z "$u" || -z "$jname" ]] && return 1
  bjobs -u "$u" -J "$jname" 2>/dev/null | awk 'NR>1 && $3 ~ /^(RUN|PEND)$/' | grep -q .
}

_grid_wait_all_done() {
  command -v bjobs >/dev/null 2>&1 || return 0
  while true; do
    local n
    n="$(_grid_pending_jobs)"
    [[ "$n" -le 0 ]] && return 0
    echo "[deepseek-grid] waiting RUN/PEND ${GRID_JOB_PREFIX}_*: $n (sleep ${GRID_POLL_SEC}s) ..." >&2
    sleep "$GRID_POLL_SEC"
  done
}

_grid_is_complete() {
  local md="$1"
  [[ -f "$md/sft_lora_state.pt" ]] || return 1
  [[ -f "$md/test_sft.csv" ]] || return 1
  local max_steps="${MAX_STEPS:-500}"
  local eval_every="${EVAL_EVERY:-100}"
  local expected=$((max_steps / eval_every))
  [[ "$expected" -ge 1 ]] || expected=1
  local lines eval_rows last_it
  lines=$(wc -l < "$md/test_sft.csv" | tr -d ' \t')
  eval_rows=$((lines - 1))
  [[ "$eval_rows" -ge "$expected" ]] || return 1
  last_it=$(awk -F, 'NR>1 {it=$1+0; if(it>max) max=it} END{print max+0}' "$md/test_sft.csv")
  [[ "$last_it" -ge "$max_steps" ]] || return 1
  return 0
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

    export JOB_NAME="${GRID_JOB_PREFIX}_${NAME}"
    if [[ "$GRID_RESUME" != "0" ]] && _grid_job_in_queue "$JOB_NAME"; then
      continue
    fi
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
import os, importlib
m = importlib.import_module(os.environ.get('DEEPSEEK_GRID_CONFIG_MODULE', 'deepseek_autogrid.config'))
st = int(os.environ.get('MAX_STEPS', str(m.MAX_STEPS_DEFAULT)))
for lr, r, alpha, wd in m.iter_grid():
    print(f'{m.run_dir_name(lr, r, alpha, st, wd)}\\t{lr}\\t{r}\\t{alpha}\\t{wd}')
")

  # GRID_RESUME=0：全量重交只做一轮；否则队列排空后会再次扫描且仍不 skip，会把 90 组整网反复 bsub（无限循环）。
  if [[ "${GRID_RESUME}" == "0" ]]; then
    echo "[deepseek-grid] GRID_RESUME=0: one-shot pass submitted ${submit_n} job(s); draining queue then exit (no second pass)." >&2
    _grid_wait_all_done
    break
  fi

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
