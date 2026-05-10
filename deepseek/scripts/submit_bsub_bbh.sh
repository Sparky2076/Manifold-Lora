#!/usr/bin/env bash
# Submit a single DeepSeek BBH (lm-eval) job to LSF.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

JOB_NAME="${JOB_NAME:-deepseek_bbh}"
QUEUE="${QUEUE:-gpu}"
NGPU=1
METRICS_DIR="${METRICS_DIR:-${ADAPTER_PATH:-}}"
if [[ -z "${METRICS_DIR}" ]]; then
  echo "[submit_bsub_bbh] set METRICS_DIR or ADAPTER_PATH" >&2
  exit 1
fi
OUTPUT_JSON="${OUTPUT_JSON:-$METRICS_DIR/bbh_eval.json}"
EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-gpu17}"

for v in TASKS NUM_FEWSHOT EVAL_BATCH_SIZE EVAL_LIMIT MODEL_NAME TORCH_DTYPE TRUST_REMOTE_CODE OUTPUT_JSON SKIP_MERGE MERGED_HF_DIR FORCE_REMERGE CONDA_ROOT CONDA_BASE CONDA_ENV_NAME; do
  eval "export $v=\"\${$v:-}\""
done

RUN_CMD="bash $SCRIPT_DIR/run_deepseek_bbh_bsub.sh"
for v in TASKS NUM_FEWSHOT EVAL_BATCH_SIZE EVAL_LIMIT MODEL_NAME TORCH_DTYPE TRUST_REMOTE_CODE OUTPUT_JSON METRICS_DIR ADAPTER_PATH SKIP_MERGE MERGED_HF_DIR FORCE_REMERGE CONDA_ROOT CONDA_BASE CONDA_ENV_NAME; do
  eval "val=\${$v}"
  if [[ -n "${val:-}" ]]; then
    RUN_CMD="export $v='$val'; $RUN_CMD"
  fi
done

HOST_FILTER_OPT=()
if [[ -n "${EXCLUDE_HOSTS// /}" ]]; then
  host_expr=""
  IFS=',' read -r -a hs <<< "$EXCLUDE_HOSTS"
  for h in "${hs[@]}"; do
    h="${h// /}"
    [[ -z "$h" ]] && continue
    [[ -n "$host_expr" ]] && host_expr+=" && "
    host_expr+="hname!='${h}'"
  done
  [[ -n "$host_expr" ]] && HOST_FILTER_OPT=(-R "select[${host_expr}]")
fi

bsub -J "$JOB_NAME" \
  -o "%J.out" \
  -e "%J.err" \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=64G]" \
  "${HOST_FILTER_OPT[@]}" \
  -gpu "num=${NGPU}" \
  "$RUN_CMD"

echo "[submit_bsub_bbh] submitted; check: bjobs"
