#!/usr/bin/env bash
# DeepSeek SFT (mLoRA) 大集复验 — 方案A（8 job 的 mLoRA 部分：4 job）
# 用 max_steps 控制预算。
#
# 服务器:
#   cd ~/Manifold-Lora
#   sed -i 's/\r$//' deepseek/scripts/*.sh
#   bash deepseek/scripts/gs_lr_deepseek_sft_mlora_big_v1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SFT_PRESET="${SFT_PRESET:-mix_chat_real_300k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.02}"

# mLoRA 候选区间来自 tuning_logs：8e-5 / 1e-4
LR_LIST=(${LR_LIST:-8e-5 1e-4})

# 预算档：短训/中训（optimizer steps）
STEPS_LIST=(${STEPS_LIST:-2000 4000})

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"

LORA_TYPE="mlora"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

TORCH_DTYPE="${TORCH_DTYPE:-float32}"

for LR in "${LR_LIST[@]}"; do
  for MAX_STEPS in "${STEPS_LIST[@]}"; do
    SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
    OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid_big/${SFT_PRESET}_mlora_big_lr_${SAFE_LR}_steps_${MAX_STEPS}"
    mkdir -p "$OUT_DIR"
    echo "==== SFT mLoRA big submit LR=$LR steps=$MAX_STEPS -> $OUT_DIR ===="
    JOB_NAME="sftbig_${SFT_PRESET}_mlora_${SAFE_LR}_s${MAX_STEPS}" \
      EPOCHS="${EPOCHS:-99}" \
      MAX_STEPS="$MAX_STEPS" \
      TORCH_DTYPE="$TORCH_DTYPE" \
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
done

echo "Done. 指标目录: deepseek/results/sft_grid_big/${SFT_PRESET}_mlora_big_*"

