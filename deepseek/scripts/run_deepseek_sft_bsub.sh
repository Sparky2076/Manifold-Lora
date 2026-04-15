#!/usr/bin/env bash
# 计算节点执行：激活环境并运行 DeepSeek SFT
set -euo pipefail

MODEL_NAME="${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
METRICS_DIR="${2:-.}"

SFT_PRESET="${SFT_PRESET:-alpaca_train_1k}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.2}"
MAX_STEPS="${MAX_STEPS:-1500}"
EVAL_EVERY="${EVAL_EVERY:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-512}"
LR="${LR:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
LORA_TYPE="${LORA_TYPE:-default}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TORCH_DTYPE="${TORCH_DTYPE:-float32}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

_resolve_conda_sh() {
  local r c
  for r in "${CONDA_ROOT:-}" "${CONDA_BASE:-}"; do
    [[ -n "$r" && -f "$r/etc/profile.d/conda.sh" ]] && { echo "$r/etc/profile.d/conda.sh"; return 0; }
  done
  if command -v conda >/dev/null 2>&1; then
    c="$(conda info --base 2>/dev/null || true)"
    [[ -n "$c" && -f "$c/etc/profile.d/conda.sh" ]] && { echo "$c/etc/profile.d/conda.sh"; return 0; }
  fi
  for r in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3" "/opt/conda"; do
    [[ -f "$r/etc/profile.d/conda.sh" ]] && { echo "$r/etc/profile.d/conda.sh"; return 0; }
  done
  return 1
}

CONDA_SH="$(_resolve_conda_sh)" || true
if [[ -z "${CONDA_SH:-}" || ! -f "$CONDA_SH" ]]; then
  echo "[run_deepseek_sft_bsub] conda.sh not found, set CONDA_ROOT first." >&2
  exit 1
fi
source "$CONDA_SH"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch}"
conda activate "$CONDA_ENV_NAME"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
echo "[run_deepseek_sft_bsub] host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | head -8 >&2 || true

META_PATH="$METRICS_DIR/run_meta.json"
if [[ ! -f "$META_PATH" ]] && [[ -n "${METRICS_DIR:-}" ]] && [[ "$METRICS_DIR" != "." ]]; then
  mkdir -p "$METRICS_DIR"
  python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
p = Path(os.environ.get("METRICS_DIR", ".")) / "run_meta.json"
if not p.exists():
    payload = {
        "lr": float(os.environ.get("LR", "3e-5")),
        "lora_r": int(float(os.environ.get("LORA_R", "16"))),
        "lora_alpha": float(os.environ.get("LORA_ALPHA", "32")),
        "weight_decay": float(os.environ.get("WEIGHT_DECAY", "0.01")),
        "max_steps": int(float(os.environ.get("MAX_STEPS", "1500"))),
        "eval_every": int(float(os.environ.get("EVAL_EVERY", "100"))),
        "sft_preset": os.environ.get("SFT_PRESET", "alpaca_train_1k"),
        "sft_val_ratio": float(os.environ.get("SFT_VAL_RATIO", "0.2")),
        "lora_type": os.environ.get("LORA_TYPE", "default"),
        "metrics_dir": str(Path(os.environ.get("METRICS_DIR", ".")).resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
fi

python -m deepseek.main_sft \
  --model_name "$MODEL_NAME" \
  --sft_preset "$SFT_PRESET" --sft_val_ratio "$SFT_VAL_RATIO" \
  --max_steps "$MAX_STEPS" --eval_every "$EVAL_EVERY" \
  --batch_size "$BATCH_SIZE" --grad_accum_steps "$GRAD_ACCUM_STEPS" --max_length "$MAX_LENGTH" \
  --lr "$LR" --weight_decay "$WEIGHT_DECAY" --adam_beta1 "$ADAM_BETA1" --adam_beta2 "$ADAM_BETA2" \
  --lora_type "$LORA_TYPE" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --torch_dtype "$TORCH_DTYPE" \
  --metrics_dir "$METRICS_DIR"
