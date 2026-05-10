#!/usr/bin/env bash
# 在服务器上运行：提交 LSF 训练任务（单卡，DistilBERT 分类）
# 用法: cd ~/Manifold-Lora && bash distilbert/scripts/submit_bsub.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

JOB_NAME="${JOB_NAME:-manifold_lora}"
QUEUE="${QUEUE:-gpu}"
NGPU=1
MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
METRICS_DIR="${METRICS_DIR:-$PROJECT_DIR/distilbert/results}"
EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-gpu17}"

export EPOCHS="${EPOCHS:-}"
export BATCH_SIZE="${BATCH_SIZE:-}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
export LR="${LR:-}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-}"
export ADAM_BETA1="${ADAM_BETA1:-}"
export ADAM_BETA2="${ADAM_BETA2:-}"
export LORA_TYPE="${LORA_TYPE:-}"
export LORA_R="${LORA_R:-}"
export LORA_ALPHA="${LORA_ALPHA:-}"
export LORA_DROPOUT="${LORA_DROPOUT:-}"
export CONDA_ROOT="${CONDA_ROOT:-}"
export CONDA_BASE="${CONDA_BASE:-}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

RUN_CMD="bash $SCRIPT_DIR/run_train_bsub.sh '$MODEL_NAME' '$METRICS_DIR'"
for v in EPOCHS BATCH_SIZE GRAD_ACCUM_STEPS LR WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 LORA_TYPE LORA_R LORA_ALPHA LORA_DROPOUT CONDA_ROOT CONDA_BASE CONDA_ENV_NAME; do
  eval "val=\${$v}"
  if [[ -n "${val:-}" ]]; then
    RUN_CMD="export $v='$val'; $RUN_CMD"
  fi
done

HOST_FILTER_OPT=()
if [[ -n "${EXCLUDE_HOSTS// /}" ]]; then
  host_expr=""
  IFS=',' read -r -a _hosts <<< "$EXCLUDE_HOSTS"
  for h in "${_hosts[@]}"; do
    h="${h// /}"
    [[ -z "$h" ]] && continue
    if [[ -n "$host_expr" ]]; then
      host_expr+=" && "
    fi
    host_expr+="hname!='${h}'"
  done
  if [[ -n "$host_expr" ]]; then
    HOST_FILTER_OPT=(-R "select[${host_expr}]")
  fi
fi

bsub -J "$JOB_NAME" \
  -o "%J.out" \
  -e "%J.err" \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=32G]" \
  "${HOST_FILTER_OPT[@]}" \
  -gpu "num=${NGPU}" \
  "$RUN_CMD"

echo "已提交任务，查看: bjobs"
