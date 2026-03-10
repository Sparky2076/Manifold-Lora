#!/usr/bin/env bash
# 供 bsub 在计算节点执行：激活环境并运行训练
# 不要直接运行，由 submit_bsub.sh 提交

set -euo pipefail

MODEL_NAME="${1:-distilbert-base-uncased}"
METRICS_DIR="${2:-.}"

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
  --epochs 50 --batch_size 4 --max_length 128 \
  --grad_accum_steps 8 \
  --lr 1e-5 --weight_decay 0.01 --max_grad_norm 1.0 \
  --lora_type default --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --log_every 50 \
  --metrics_dir "$METRICS_DIR"
