#!/usr/bin/env bash
# DeepSeek SFT LoRA 第三轮（步数预算）
#
# 说明（关于 LR）：
#   - 默认 LR=3.5e-5 仅表示「在 round2 细扫结论下，本轮先固定学习率、只扫 max_steps」，
#     不是「以后永远不调 LR」。
#   - 长训后若 eval 回升、不稳或仍想对比，可随时设环境变量覆盖，例如：
#       LR=3e-5 bash deepseek/scripts/gs_sft_lora_round3_steps.sh
#   - 若需要再开一轮 LR 网格，继续用 round2 脚本或复制本脚本改成 LR_LIST 循环即可。
#
# 扫 max_steps，看 eval 是否继续改善或进入平台期。
#
# 用法:
#   cd ~/Manifold-Lora
#   sed -i 's/\r$//' deepseek/scripts/*.sh
#   chmod +x deepseek/scripts/*.sh
#   bash deepseek/scripts/gs_sft_lora_round3_steps.sh
#
# 覆盖示例:
#   STEPS_LIST="8000 10000" bash deepseek/scripts/gs_sft_lora_round3_steps.sh
#   LR=3e-5 STEPS_LIST="10000" bash deepseek/scripts/gs_sft_lora_round3_steps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SFT_PRESET="${SFT_PRESET:-mix_chat_real_300k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.02}"

# 外层 epoch 仅作上限（训练由 max_steps 截断）
EPOCHS="${EPOCHS:-16}"

NGPU="${NGPU:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

LORA_TYPE="default"
LR="${LR:-3.5e-5}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
STEPS_LIST=(${STEPS_LIST:-8000 10000 12000})

for MAX_STEPS in "${STEPS_LIST[@]}"; do
  OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid_round3/${SFT_PRESET}_lora_lr_${SAFE_LR}_s${MAX_STEPS}"
  mkdir -p "$OUT_DIR"
  echo "==== round3 LoRA submit LR=$LR steps=$MAX_STEPS -> $OUT_DIR ===="
  JOB_NAME="sftr3_${SFT_PRESET}_lora_${SAFE_LR}_s${MAX_STEPS}" \
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

echo "Done. total_jobs=${#STEPS_LIST[@]} (LoRA round3: fixed lr, step sweep)"
