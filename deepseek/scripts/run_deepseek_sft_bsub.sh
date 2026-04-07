#!/usr/bin/env bash
# 计算节点执行：DeepSeek SFT（须在仓库根目录 python -m deepseek.main_sft）
# 由 deepseek/scripts/submit_bsub_sft.sh 提交
# 注意：不用 bash 数组语法，避免被 sh 调用或 sed 误改后出现 line 35 `(` 语法错误

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

# 更大子集：tatsu-lab/alpaca 前 1000 条；验证集比例见 SFT_VAL_RATIO
SFT_PRESET="${SFT_PRESET:-alpaca_train_1k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.2}"
SFT_DATASET="${SFT_DATASET:-}"
SFT_SPLIT="${SFT_SPLIT:-train}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch

# 不支持 bf16 的 GPU 用 float32；想加速可改为 float16（需 GPU 支持且稳定）
TORCH_DTYPE="${TORCH_DTYPE:-float32}"
MAX_STEPS="${MAX_STEPS:-}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

GC_FLAG=()
if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  GC_FLAG+=(--gradient_checkpointing)
fi

if [[ -n "${SFT_DATASET}" ]]; then
  exec python -m deepseek.main_sft \
    --model_name "$MODEL_NAME" \
    --trust_remote_code \
    --device_map auto \
    --torch_dtype "$TORCH_DTYPE" \
    --epochs "$EPOCHS" \
    ${MAX_STEPS:+--max_steps "$MAX_STEPS"} \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --lr "$LR" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --lora_type "$LORA_TYPE" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --log_every 5 \
    --metrics_dir "$METRICS_DIR" \
    --sft_val_ratio "$SFT_VAL_RATIO" \
    --sft_dataset "$SFT_DATASET" \
    --sft_split "$SFT_SPLIT" \
    "${GC_FLAG[@]}"
else
  exec python -m deepseek.main_sft \
    --model_name "$MODEL_NAME" \
    --trust_remote_code \
    --device_map auto \
    --torch_dtype "$TORCH_DTYPE" \
    --epochs "$EPOCHS" \
    ${MAX_STEPS:+--max_steps "$MAX_STEPS"} \
    --batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --grad_accum_steps "$GRAD_ACCUM_STEPS" \
    --lr "$LR" \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --lora_type "$LORA_TYPE" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --log_every 5 \
    --metrics_dir "$METRICS_DIR" \
    --sft_val_ratio "$SFT_VAL_RATIO" \
    --sft_preset "$SFT_PRESET" \
    "${GC_FLAG[@]}"
fi
