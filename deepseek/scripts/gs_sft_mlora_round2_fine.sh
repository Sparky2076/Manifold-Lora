#!/usr/bin/env bash
# DeepSeek SFT mLoRA 第二轮（fine grid）
# 目的：围绕 round1 最优 lr=1e-4 做细化搜索。
#
# 用法:
#   cd ~/Manifold-Lora
#   sed -i 's/\r$//' deepseek/scripts/*.sh
#   chmod +x deepseek/scripts/*.sh
#   bash deepseek/scripts/gs_sft_mlora_round2_fine.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SFT_PRESET="${SFT_PRESET:-mix_chat_real_300k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.02}"

# 第二轮适度拉长预算，验证稳定性
MAX_STEPS="${MAX_STEPS:-6000}"
EPOCHS="${EPOCHS:-12}"

# 单卡并发友好（如需多卡可覆盖 NGPU/NPROC_PER_NODE）
NGPU="${NGPU:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

LORA_TYPE="mlora"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# 围绕 1e-4 细化 3 点
LR_LIST=(${LR_LIST:-9e-5 1e-4 1.1e-4})

for LR in "${LR_LIST[@]}"; do
  SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
  OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid_round2/${SFT_PRESET}_mlora_lr_${SAFE_LR}_s${MAX_STEPS}"
  mkdir -p "$OUT_DIR"
  echo "==== round2 mLoRA submit LR=$LR steps=$MAX_STEPS -> $OUT_DIR ===="
  JOB_NAME="sftr2_${SFT_PRESET}_mlora_${SAFE_LR}_s${MAX_STEPS}" \
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

echo "Done. total_jobs=3 (mLoRA round2 fine)"

