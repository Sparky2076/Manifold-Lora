#!/usr/bin/env bash
# DeepSeek SFT 学习率网格（小指令集 + main_sft.py）
# 服务器: cd ~/Manifold-Lora && sed -i 's/\r$//' scripts/*.sh && bash scripts/gs_lr_deepseek_sft.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LR_LIST=(1e-5 2e-5 5e-5)
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
SFT_PRESET="${SFT_PRESET:-testing_alpaca_small}"
LORA_TYPE="${LORA_TYPE:-default}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

for LR in "${LR_LIST[@]}"; do
  # 目录名避免特殊字符：1e-5 -> 1e_5
  SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
  OUT_DIR="$PROJECT_DIR/results/sft_grid/${SFT_PRESET}_lr_${SAFE_LR}"
  mkdir -p "$OUT_DIR"
  echo "==== SFT submit LR=$LR -> $OUT_DIR ===="
  JOB_NAME="sft_${SFT_PRESET}_lr_${SAFE_LR}" \
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
    METRICS_DIR="$OUT_DIR" \
    bash "$SCRIPT_DIR/submit_bsub_sft.sh"
  sleep 2
done

echo "Done. 各 job 的 metrics 在 results/sft_grid/ 下对应目录的 train_sft.csv / test_sft.csv"
