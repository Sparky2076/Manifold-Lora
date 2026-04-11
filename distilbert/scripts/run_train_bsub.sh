#!/usr/bin/env bash
# 供 bsub 在计算节点执行：激活环境并运行 DistilBERT 分类训练
# 不要直接运行，由 distilbert/scripts/submit_bsub.sh 提交

set -euo pipefail

MODEL_NAME="${1:-distilbert-base-uncased}"
METRICS_DIR="${2:-.}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
ADAM_BETA1="${ADAM_BETA1:-0.9}"
ADAM_BETA2="${ADAM_BETA2:-0.999}"
LORA_TYPE="${LORA_TYPE:-default}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate torch

# LSF 网格：若尚未有 run_meta.json（本地 run_grid.py 会预写），则从环境变量写入，供 aggregate_results 使用
META_PATH="$METRICS_DIR/run_meta.json"
if [[ ! -f "$META_PATH" ]] && [[ -n "${METRICS_DIR:-}" ]] && [[ "$METRICS_DIR" != "." ]]; then
  mkdir -p "$METRICS_DIR"
  python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path

md = os.environ.get("METRICS_DIR", ".")
p = Path(md) / "run_meta.json"
if p.is_file():
    raise SystemExit(0)

def num(x, default, cast):
    v = os.environ.get(x, "").strip()
    if not v:
        return default
    return cast(v)

payload = {
    "lr": num("LR", 1e-5, float),
    "lora_r": num("LORA_R", 8, int),
    "lora_alpha": num("LORA_ALPHA", 16.0, float),
    "epochs": num("EPOCHS", 50, int),
    "weight_decay": num("WEIGHT_DECAY", 0.01, float),
    "adam_beta1": num("ADAM_BETA1", 0.9, float),
    "adam_beta2": num("ADAM_BETA2", 0.999, float),
    "metrics_dir": str(Path(md).resolve()),
    "command": [],
    "created_utc": datetime.now(timezone.utc).isoformat(),
}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
fi

if [[ "$MODEL_NAME" == *distilbert* ]] || [[ "$MODEL_NAME" == *bert-base* ]] || [[ "$MODEL_NAME" == *roberta* ]]; then
  EXTRA_ARGS=""
else
  EXTRA_ARGS="--trust_remote_code --device_map auto --torch_dtype float16"
fi

python -m distilbert.main \
  --model_name "$MODEL_NAME" \
  $EXTRA_ARGS \
  --dataset_name glue --dataset_config sst2 --text_field sentence \
  --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --max_length 128 \
  --grad_accum_steps "$GRAD_ACCUM_STEPS" \
  --lr "$LR" --weight_decay "$WEIGHT_DECAY" --adam_beta1 "$ADAM_BETA1" --adam_beta2 "$ADAM_BETA2" --max_grad_norm 1.0 \
  --lora_type "$LORA_TYPE" --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout "$LORA_DROPOUT" \
  --log_every 50 \
  --metrics_dir "$METRICS_DIR"
