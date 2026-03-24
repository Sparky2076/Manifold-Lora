#!/usr/bin/env bash
# DeepSeek SFT + mLoRA 学习率网格；结果写入 deepseek/results/sft_grid/
# 服务器:
#   cd ~/Manifold-Lora
#   sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh deepseek/scripts/*.sh
#   bash deepseek/scripts/gs_lr_deepseek_sft_mlora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# mLoRA 常用更高学习率区间（可通过外部环境变量覆盖）
LR_LIST=(${LR_LIST:-1e-5 2e-5 5e-5})
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"

# 默认更大子集与更大验证集
SFT_PRESET="${SFT_PRESET:-alpaca_train_1k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.2}"

LORA_TYPE="mlora"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

for LR in "${LR_LIST[@]}"; do
  SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
  OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid/${SFT_PRESET}_mlora_lr_${SAFE_LR}"
  mkdir -p "$OUT_DIR"
  echo "==== SFT mLoRA submit LR=$LR -> $OUT_DIR ===="
  JOB_NAME="sft_${SFT_PRESET}_mlora_lr_${SAFE_LR}" \
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

echo "Done. 各 job 指标: deepseek/results/sft_grid/<子目录>/train_sft.csv 与 test_sft.csv"
