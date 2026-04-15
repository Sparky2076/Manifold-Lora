#!/usr/bin/env bash
# 提交单个 DeepSeek SFT 作业到 LSF
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

JOB_NAME="${JOB_NAME:-deepseek_sft}"
QUEUE="${QUEUE:-gpu}"
NGPU=1
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
METRICS_DIR="${METRICS_DIR:-$PROJECT_DIR/deepseek/results}"
EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-gpu17}"

for v in SFT_PRESET SFT_VAL_RATIO MAX_STEPS EVAL_EVERY BATCH_SIZE GRAD_ACCUM_STEPS MAX_LENGTH LR WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 LORA_TYPE LORA_R LORA_ALPHA LORA_DROPOUT TORCH_DTYPE CONDA_ROOT CONDA_BASE CONDA_ENV_NAME; do
  eval "export $v=\"\${$v:-}\""
done

RUN_CMD="bash $SCRIPT_DIR/run_deepseek_sft_bsub.sh '$MODEL_NAME' '$METRICS_DIR'"
for v in SFT_PRESET SFT_VAL_RATIO MAX_STEPS EVAL_EVERY BATCH_SIZE GRAD_ACCUM_STEPS MAX_LENGTH LR WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 LORA_TYPE LORA_R LORA_ALPHA LORA_DROPOUT TORCH_DTYPE CONDA_ROOT CONDA_BASE CONDA_ENV_NAME; do
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

echo "已提交 DeepSeek SFT 任务，查看: bjobs"
