#!/usr/bin/env bash
# DeepSeek SFT (LoRA) 大集复验 — 方案A（8 job 的 LoRA 部分：4 job）
# 用 max_steps 控制预算，避免“1/2 epoch”在大集上不可比。
#
# 服务器:
#   cd ~/Manifold-Lora
#   sed -i 's/\r$//' deepseek/scripts/*.sh
#   bash deepseek/scripts/gs_lr_deepseek_sft_big_v1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

SFT_PRESET="${SFT_PRESET:-mix_chat_real_300k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.02}"

# 默认网格：先扫学习率，再比较容量/正则；都可通过环境变量覆盖
LR_LIST=(${LR_LIST:-1e-5 2e-5 3e-5})
LORA_R_LIST=(${LORA_R_LIST:-8 16})
LORA_DROPOUT_LIST=(${LORA_DROPOUT_LIST:-0.05 0.1})

# 预算档：短训/中训（optimizer steps）
STEPS_LIST=(${STEPS_LIST:-2000 6000})

# 多卡 DDP：每卡 batch（全局吞吐 ≈ BATCH_SIZE * NGPU * GRAD_ACCUM_STEPS）
NGPU="${NGPU:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$NGPU}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"

LORA_TYPE="default"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

for LR in "${LR_LIST[@]}"; do
  for LORA_R in "${LORA_R_LIST[@]}"; do
    LORA_ALPHA=$((LORA_R * 2))
    for LORA_DROPOUT in "${LORA_DROPOUT_LIST[@]}"; do
      for MAX_STEPS in "${STEPS_LIST[@]}"; do
        SAFE_LR=$(echo "$LR" | sed 's/\./p/g; s/-/_/g')
        SAFE_DROPOUT=$(echo "$LORA_DROPOUT" | sed 's/\./p/g')
        OUT_DIR="$PROJECT_DIR/deepseek/results/sft_grid_big/${SFT_PRESET}_lora_big_lr_${SAFE_LR}_r${LORA_R}_a${LORA_ALPHA}_d${SAFE_DROPOUT}_steps_${MAX_STEPS}"
        mkdir -p "$OUT_DIR"
        echo "==== SFT LoRA big submit LR=$LR r=$LORA_R a=$LORA_ALPHA d=$LORA_DROPOUT steps=$MAX_STEPS -> $OUT_DIR ===="
        # epochs 在 max_steps 方案下仅作外层循环上限，不是主要预算控制器。
        JOB_NAME="sftbig_${SFT_PRESET}_lora_${SAFE_LR}_r${LORA_R}_d${SAFE_DROPOUT}_s${MAX_STEPS}" \
          NGPU="$NGPU" \
          NPROC_PER_NODE="$NPROC_PER_NODE" \
          EPOCHS="${EPOCHS:-20}" \
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
    done
  done
done

echo "Done. 指标目录: deepseek/results/sft_grid_big/${SFT_PRESET}_lora_big_*"

