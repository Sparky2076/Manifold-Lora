#!/usr/bin/env bash
# 供 bsub 在计算节点执行：激活环境并运行训练
# 不要直接运行，由 submit_bsub.sh 提交

set -euo pipefail

MODEL_NAME="${1:-distilbert-base-uncased}"
METRICS_DIR="${2:-.}"

# 允许通过环境变量覆盖部分超参数，便于做 LoRA / mLoRA 实验对比
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-1e-5}"
LORA_TYPE="${LORA_TYPE:-default}"           # default 或 mlora
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch

# DistilBERT/BERT 等不支持 device_map，大模型（如 DeepSeek）才加这些参数
if [[ "$MODEL_NAME" == *distilbert* ]] || [[ "$MODEL_NAME" == *bert-base* ]] || [[ "$MODEL_NAME" == *roberta* ]]; then
  EXTRA_ARGS=""
else
  EXTRA_ARGS="--trust_remote_code --device_map auto --torch_dtype float16"
fi

python main.py \
  --model_name "$MODEL_NAME" \
  $EXTRA_ARGS \
  --dataset_name glue --dataset_config sst2 --text_field sentence \
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --max_length 128 \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --lr "$LR" --weight_decay 0.01 --max_grad_norm 1.0 \
  --lora_type "$LORA_TYPE" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --log_every 50 \
  --metrics_dir "$METRICS_DIR"



