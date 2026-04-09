#!/usr/bin/env bash
# DeepSeek SFT mLoRA 第三轮（步数预算）
# 固定 round2 最优 lr=1.1e-4，扫 max_steps，验证「慢热 / 需更长训练」假设。
#
# 默认 LR 可被覆盖：LR=1e-4 STEPS_LIST="10000" bash ...
#
# 用法:
#   bash deepseek/scripts/gs_sft_mlora_round3_steps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SFT_PRESET="${SFT_PRESET:-mix_chat_real_300k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.02}"
EPOCHS="${EPOCHS:-16}"

NGPU="${NGPU:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

LORA_TYPE="mlora"
LR="${LR:-1.1e-4}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
STEPS_LIST=(${STEPS_LIST:-8000 10000 12000})

for MAX_STEPS in "${STEPS_LIST[@]}"; do
  OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid_round3/${SFT_PRESET}_mlora_lr_${SAFE_LR}_s${MAX_STEPS}"
  mkdir -p "$OUT_DIR"
  echo "==== round3 mLoRA submit LR=$LR steps=$MAX_STEPS -> $OUT_DIR ===="
  JOB_NAME="sftr3_${SFT_PRESET}_mlora_${SAFE_LR}_s${MAX_STEPS}" \
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

echo "Done. total_jobs=${#STEPS_LIST[@]} (mLoRA round3: fixed lr, step sweep)"
