#!/usr/bin/env bash
# DeepSeek SFT 粗筛：LoRA 单卡 3 任务（并发友好）
#
# 用法:
#   cd ~/Manifold-Lora
#   sed -i 's/\r$//' deepseek/scripts/*.sh
#   bash deepseek/scripts/gs_sft_lora_grid3_coarse.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SFT_PRESET="${SFT_PRESET:-mix_chat_real_300k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.02}"

# 粗筛预算：建议 2k~4k；默认取中间值
MAX_STEPS="${MAX_STEPS:-3000}"
EPOCHS="${EPOCHS:-8}"

# 单卡高并发
NGPU="${NGPU:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

LORA_TYPE="default"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# LoRA 粗筛学习率 3 点
LR_LIST=(${LR_LIST:-1e-5 2e-5 3e-5})

for LR in "${LR_LIST[@]}"; do
  SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
  OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid_coarse/${SFT_PRESET}_lora_lr_${SAFE_LR}_s${MAX_STEPS}"
  mkdir -p "$OUT_DIR"
  echo "==== coarse LoRA submit LR=$LR steps=$MAX_STEPS -> $OUT_DIR ===="
  JOB_NAME="sftc_${SFT_PRESET}_lora_${SAFE_LR}_s${MAX_STEPS}" \
    NGPU="$NGPU" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    EPOCHS="$EPOCHS" \
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

echo "Done. total_jobs=3 (LoRA coarse)"

