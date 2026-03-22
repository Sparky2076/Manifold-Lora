#!/usr/bin/env bash
# mLoRA 学习率网格搜索：每个 lr 跑 20 epoch，DistilBERT + SST2
# 在服务器上运行: cd ~/Manifold-Lora && sed -i 's/\r$//' distilbert/scripts/*.sh && bash distilbert/scripts/gs_lr_mlora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

LR_LIST=(1.2e-4 1.5e-4 2e-4)
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
  bash distilbert/scripts/submit_bsub.sh
  sleep 2
done

echo "Done. Check: bjobs"
