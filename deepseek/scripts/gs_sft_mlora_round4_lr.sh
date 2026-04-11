#!/usr/bin/env bash
# DeepSeek SFT mLoRA 第四轮（固定 max_steps=10000，扫学习率）
#
# 说明：此前仓库里曾用文件名 round5 指代本实验；按调参轮次编号，本轮为 **mLoRA round 4**
#（round1 粗筛 / round2 细扫 / round3 步数 sweep 之后的第一轮「固定步数 lr 细扫」）。
#
# 指标目录仍使用 `sft_grid_round5/`，与已提交作业路径一致。
#
# 用法:
#   bash deepseek/scripts/gs_sft_mlora_round4_lr.sh
#
# 覆盖示例:
#   LR_LIST="1e-4 1.1e-4" MAX_STEPS=10000 bash deepseek/scripts/gs_sft_mlora_round4_lr.sh
# 默认 LR：1.1e-4 / 1.3e-4 / 1.5e-4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SFT_PRESET="${SFT_PRESET:-mix_chat_real_300k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.02}"
MAX_STEPS="${MAX_STEPS:-10000}"
EPOCHS="${EPOCHS:-20}"

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

LR_LIST=(${LR_LIST:-1.1e-4 1.3e-4 1.5e-4})

for LR in "${LR_LIST[@]}"; do
  SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
  OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid_round5/${SFT_PRESET}_mlora_lr_${SAFE_LR}_s${MAX_STEPS}"
  mkdir -p "$OUT_DIR"
  echo "==== round4 mLoRA submit LR=$LR steps=$MAX_STEPS -> $OUT_DIR ===="
  JOB_NAME="sftr4_${SFT_PRESET}_mlora_${SAFE_LR}_s${MAX_STEPS}" \
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

echo "Done. total_jobs=${#LR_LIST[@]} (mLoRA round4: fixed steps=${MAX_STEPS}, lr sweep)"
