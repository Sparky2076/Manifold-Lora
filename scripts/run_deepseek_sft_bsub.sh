#!/usr/bin/env bash
# 计算节点执行：DeepSeek / CausalLM 指令微调 (main_sft.py)
# 由 submit_bsub_sft.sh 提交，勿在登录节点直接长跑

set -euo pipefail

MODEL_NAME="${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
METRICS_DIR="${2:-.}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LR="${LR:-2e-5}"
LORA_TYPE="${LORA_TYPE:-default}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# 小指令集：testing_alpaca_small | alpaca_gpt4_500 | alpaca_train_500 | alpaca_train_1k
SFT_PRESET="${SFT_PRESET:-testing_alpaca_small}"
SFT_DATASET="${SFT_DATASET:-}"
SFT_SPLIT="${SFT_SPLIT:-train}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch

EXTRA=(--trust_remote_code --device_map auto --torch_dtype float16)

CMD=(python main_sft.py
  --model_name "$MODEL_NAME"
  "${EXTRA[@]}"
  --epochs "$EPOCHS"
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
)

if [[ -n "${SFT_DATASET}" ]]; then
  CMD+=(--sft_dataset "$SFT_DATASET" --sft_split "$SFT_SPLIT")
else
  CMD+=(--sft_preset "$SFT_PRESET")
fi

"${CMD[@]}"
