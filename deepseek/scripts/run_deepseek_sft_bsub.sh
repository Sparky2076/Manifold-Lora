#!/usr/bin/env bash
# 计算节点执行：DeepSeek SFT（须在仓库根目录调用）
# 由 deepseek/scripts/submit_bsub_sft.sh 提交
# 多卡：设置 NPROC_PER_NODE>1，本脚本会用 torchrun 启动 DDP（每进程一卡）。

set -euo pipefail

MODEL_NAME="${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
METRICS_DIR="${2:-.}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LR="${LR:-2e-5}"
LORA_TYPE="${LORA_TYPE:-default}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

SFT_PRESET="${SFT_PRESET:-alpaca_train_1k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.2}"
SFT_DATASET="${SFT_DATASET:-}"
SFT_SPLIT="${SFT_SPLIT:-train}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch

TORCH_DTYPE="${TORCH_DTYPE:-float32}"
MAX_STEPS="${MAX_STEPS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

GC_FLAG=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  GC_FLAG+=(--gradient_checkpointing)
fi

MS_FLAG=()
if [[ -n "${MAX_STEPS}" ]]; then
  MS_FLAG+=(--max_steps "${MAX_STEPS}")
fi

# DDP 时 main_sft 会忽略 device_map；单卡仍可用 auto 省显存
DEVICE_MAP="${DEVICE_MAP:-auto}"
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  DEVICE_MAP="none"
fi

BASE_ARGS=(
  --model_name "$MODEL_NAME"
  --trust_remote_code
  --device_map "$DEVICE_MAP"
  --torch_dtype "$TORCH_DTYPE"
  --epochs "$EPOCHS"
  "${MS_FLAG[@]}"
  --batch_size "$BATCH_SIZE"
  --max_length "$MAX_LENGTH"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --lr "$LR"
  --weight_decay 0.01
  --max_grad_norm 1.0
  --lora_type "$LORA_TYPE"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --lora_dropout "$LORA_DROPOUT"
  --log_every 5
  --metrics_dir "$METRICS_DIR"
  --sft_val_ratio "$SFT_VAL_RATIO"
  "${GC_FLAG[@]}"
)

if [[ -n "${SFT_DATASET}" ]]; then
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m deepseek.main_sft \
      "${BASE_ARGS[@]}" --sft_dataset "$SFT_DATASET" --sft_split "$SFT_SPLIT"
  else
    exec python -m deepseek.main_sft \
      "${BASE_ARGS[@]}" --sft_dataset "$SFT_DATASET" --sft_split "$SFT_SPLIT"
  fi
else
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m deepseek.main_sft \
      "${BASE_ARGS[@]}" --sft_preset "$SFT_PRESET"
  else
    exec python -m deepseek.main_sft \
      "${BASE_ARGS[@]}" --sft_preset "$SFT_PRESET"
  fi
fi
