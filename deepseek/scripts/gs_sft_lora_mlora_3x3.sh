#!/usr/bin/env bash
# DeepSeek SFT 调参脚本：一次提交 6 个任务（LoRA 3 + mLoRA 3）
# 目标：在当前数据集下快速比较 LoRA / mLoRA 的最优学习率区间。
#
# 用法（服务器）:
#   cd ~/Manifold-Lora
#   sed -i 's/\r$//' deepseek/scripts/*.sh
#   bash deepseek/scripts/gs_sft_lora_mlora_3x3.sh
#
# 可覆盖参数示例:
#   SFT_PRESET=mix_chat_real_300k SFT_VAL_RATIO=0.02 MAX_STEPS=8000 EPOCHS=12 \
#   LORA_LR_LIST="1e-5 2e-5 3e-5" MLORA_LR_LIST="5e-5 8e-5 1e-4" \
#   bash deepseek/scripts/gs_sft_lora_mlora_3x3.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 数据与评估
SFT_PRESET="${SFT_PRESET:-mix_chat_real_300k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.02}"

# 训练预算（调参期以 max_steps 为主；epochs 仅作为外层上限）
MAX_STEPS="${MAX_STEPS:-8000}"
EPOCHS="${EPOCHS:-12}"

# 训练并行/吞吐
NGPU="${NGPU:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$NGPU}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

# LoRA 结构参数（两种方法先保持一致，方便公平比较）
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# 每种方法固定提交 3 个学习率任务
LORA_LR_LIST=(${LORA_LR_LIST:-1e-5 2e-5 3e-5})
MLORA_LR_LIST=(${MLORA_LR_LIST:-5e-5 8e-5 1e-4})

submit_one() {
  local method="$1"   # lora | mlora
  local lr="$2"
  local safe_lr
  safe_lr=$(echo "$lr" | sed 's/\./p/g; s/-/_/g')

  local out_dir="$PROJECT_DIR/deepseek/results/sft_grid_big/${SFT_PRESET}_${method}_grid3_lr_${safe_lr}_s${MAX_STEPS}"
  mkdir -p "$out_dir"

  echo "==== submit method=$method lr=$lr steps=$MAX_STEPS -> $out_dir ===="
  JOB_NAME="sft_${SFT_PRESET}_${method}_${safe_lr}_s${MAX_STEPS}" \
    NGPU="$NGPU" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    EPOCHS="$EPOCHS" \
    MAX_STEPS="$MAX_STEPS" \
    TORCH_DTYPE="$TORCH_DTYPE" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    MAX_LENGTH="$MAX_LENGTH" \
    LR="$lr" \
    LORA_TYPE="$method" \
    LORA_R="$LORA_R" \
    LORA_ALPHA="$LORA_ALPHA" \
    LORA_DROPOUT="$LORA_DROPOUT" \
    SFT_PRESET="$SFT_PRESET" \
    SFT_VAL_RATIO="$SFT_VAL_RATIO" \
    METRICS_DIR="$out_dir" \
    bash "$SCRIPT_DIR/submit_bsub_sft.sh"
}

echo ">>> Submit LoRA grid (3 jobs)"
for LR in "${LORA_LR_LIST[@]}"; do
  submit_one "default" "$LR"
  sleep 2
done

echo ">>> Submit mLoRA grid (3 jobs)"
for LR in "${MLORA_LR_LIST[@]}"; do
  submit_one "mlora" "$LR"
  sleep 2
done

echo "Done. total_jobs=6"
echo "Metrics base dir: deepseek/results/sft_grid_big/"

