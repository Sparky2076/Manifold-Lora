#!/usr/bin/env bash
# mLoRA 学习率网格搜索：每个 lr 跑 20 epoch，DistilBERT + SST2
# 在服务器上运行: cd ~/Manifold-Lora && sed -i 's/\r$//' scripts/*.sh && bash scripts/gs_lr_mlora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# 可改这里扩展/缩小搜索范围（mLoRA 第 2 轮 lr 精调）
# 第 1 轮结果显示在 5e-5 一侧表现最好，这一轮围绕其附近加密：
LR_LIST=(3e-5 5e-5 7e-5 1e-4)
EPOCHS=20
LORA_TYPE=mlora
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

for LR in "${LR_LIST[@]}"; do
  echo "==== Submitting mLoRA, LR=$LR, EPOCHS=$EPOCHS ===="
  JOB_NAME="mlora_lr_${LR}" \
  EPOCHS=$EPOCHS \
  LORA_TYPE=$LORA_TYPE \
  LORA_R=$LORA_R \
  LORA_ALPHA=$LORA_ALPHA \
  LORA_DROPOUT=$LORA_DROPOUT \
  LR=$LR \
  bash scripts/submit_bsub.sh
  sleep 2
done

echo "Done. Check: bjobs"
