#!/usr/bin/env bash
# 提交 DeepSeek SFT 单卡任务
# 用法: cd ~/Manifold-Lora && sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh deepseek/scripts/*.sh && bash deepseek/scripts/submit_bsub_sft.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

JOB_NAME="${JOB_NAME:-deepseek_sft}"
QUEUE="${QUEUE:-gpu}"
NGPU=1
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
METRICS_DIR="${METRICS_DIR:-$PROJECT_DIR/deepseek/results}"

export EPOCHS="${EPOCHS:-}"
export BATCH_SIZE="${BATCH_SIZE:-}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
export MAX_LENGTH="${MAX_LENGTH:-}"
export LR="${LR:-}"
export LORA_TYPE="${LORA_TYPE:-}"
export LORA_R="${LORA_R:-}"
export LORA_ALPHA="${LORA_ALPHA:-}"
export LORA_DROPOUT="${LORA_DROPOUT:-}"
export SFT_PRESET="${SFT_PRESET:-}"
export SFT_VAL_RATIO="${SFT_VAL_RATIO:-}"
export SFT_DATASET="${SFT_DATASET:-}"
export SFT_SPLIT="${SFT_SPLIT:-}"

RUN_CMD="bash $SCRIPT_DIR/run_deepseek_sft_bsub.sh '$MODEL_NAME' '$METRICS_DIR'"
for v in EPOCHS BATCH_SIZE GRAD_ACCUM_STEPS MAX_LENGTH LR LORA_TYPE LORA_R LORA_ALPHA LORA_DROPOUT SFT_PRESET SFT_VAL_RATIO SFT_DATASET SFT_SPLIT; do
  eval "val=\${$v}"
  if [[ -n "${val:-}" ]]; then
    RUN_CMD="export $v='$val'; $RUN_CMD"
  fi
done

bsub -J "$JOB_NAME" \
  -o "%J.out" \
  -e "%J.err" \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=32G]" \
  -gpu "num=${NGPU}" \
  "$RUN_CMD"

echo "已提交 SFT 任务，查看: bjobs"
