#!/usr/bin/env bash
# LoRA 学习率网格搜索：每个 lr 跑 20 epoch，DistilBERT + SST2
# 在服务器上运行: cd ~/Manifold-Lora && sed -i 's/\r$//' distilbert/scripts/*.sh && bash distilbert/scripts/gs_lr_lora.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

LR_LIST=(2.5e-5 3e-5 3.5e-5)
EPOCHS=20
LORA_TYPE=default
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

for LR in "${LR_LIST[@]}"; do
  echo "==== Submitting LoRA, LR=$LR, EPOCHS=$EPOCHS ===="
  JOB_NAME="lora_lr_${LR}" \
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
