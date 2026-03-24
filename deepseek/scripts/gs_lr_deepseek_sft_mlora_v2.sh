#!/usr/bin/env bash
# DeepSeek SFT (mLoRA) 第二轮学习率网格
# 目标：第一轮在 5e-5 仍持续下降，继续向更高学习率探索
#
# 服务器:
#   cd ~/Manifold-Lora
#   sed -i 's/\r$//' deepseek/scripts/*.sh
#   bash deepseek/scripts/gs_lr_deepseek_sft_mlora_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# mLoRA v2: 更高 lr 区间继续探测（3 点）
LR_LIST=(${LR_LIST:-5e-5 8e-5 1e-4})
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"

SFT_PRESET="${SFT_PRESET:-alpaca_train_1k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.2}"

LORA_TYPE="mlora"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

for LR in "${LR_LIST[@]}"; do
  SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
  OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid_v2/${SFT_PRESET}_mlora_v2_lr_${SAFE_LR}"
  mkdir -p "$OUT_DIR"
  echo "==== SFT mLoRA v2 submit LR=$LR -> $OUT_DIR ===="
  JOB_NAME="sft2_${SFT_PRESET}_mlora_lr_${SAFE_LR}" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    MAX_LENGTH="$MAX_LENGTH" \
    LR="$LR" \
    LORA_TYPE="$LORA_TYPE" \
    LORA_R="$LORA_R" \
    LORA_ALPHA="$LORA_ALPHA" \
    LORA_DROPOUT="$LORA_DROPOUT" \
    SFT_PRESET="$SFT_PRESET" \
    SFT_VAL_RATIO="$SFT_VAL_RATIO" \
    METRICS_DIR="$OUT_DIR" \
    bash "$SCRIPT_DIR/submit_bsub_sft.sh"
  sleep 2
done

echo "Done. v2 指标目录: deepseek/results/sft_grid_v2/<preset>_mlora_v2_lr_*/"
