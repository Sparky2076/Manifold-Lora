#!/usr/bin/env bash
# 从 distilbert_autogrid 的 summary.csv「第一条数据行」读取最优超参（与 aggregate_results 排序一致：best_val_acc 降序），
# 提交 DistilBERT 单作业，训练 EPOCHS=20 作为最终结果。
#
# 用法（必须在仓库根目录执行）:
#   bash scripts/server_submit_distilbert_best_20ep.sh lora
#   bash scripts/server_submit_distilbert_best_20ep.sh mlora
#
# 可选环境变量:
#   CONDA_ROOT / CONDA_ENV_NAME — 与 distilbert/scripts/submit_bsub.sh 相同
#   SUMMARY_CSV — 覆盖默认 summary 路径（高级用法）
#   EXCLUDE_HOSTS — 默认 gpu17
#   JOB_NAME / METRICS_DIR / EPOCHS — 一般无需改；EPOCHS 默认 20

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "lora" && "$MODE" != "mlora" ]]; then
  echo "用法: bash scripts/server_submit_distilbert_best_20ep.sh lora|mlora" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

if [[ "$MODE" == "lora" ]]; then
  DEFAULT_SUMMARY="distilbert_autogrid/results/summary.csv"
  export LORA_TYPE="${LORA_TYPE:-default}"
  DEFAULT_METRICS="distilbert/results_final_best_lora_20ep"
  DEFAULT_JOB="distilbert_best_lora_ep20"
else
  DEFAULT_SUMMARY="distilbert_autogrid/results_mlora/summary.csv"
  export LORA_TYPE="${LORA_TYPE:-mlora}"
  DEFAULT_METRICS="distilbert/results_final_best_mlora_20ep"
  DEFAULT_JOB="distilbert_best_mlora_ep20"
fi

SUMMARY_CSV="${SUMMARY_CSV:-$DEFAULT_SUMMARY}"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "[error] 找不到 summary: $SUMMARY_CSV（请先 aggregate 或检查路径）" >&2
  exit 1
fi

export EPOCHS="${EPOCHS:-20}"
export JOB_NAME="${JOB_NAME:-$DEFAULT_JOB}"
export METRICS_DIR="${METRICS_DIR:-$PROJECT_DIR/$DEFAULT_METRICS}"
export EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-gpu17}"

# 从 CSV 首行导出 LR / LORA_* / WEIGHT_DECAY / ADAM_BETA*
export _BEST20_SUMMARY_PATH="$SUMMARY_CSV"
eval "$(
  python <<'PY'
import csv, os, shlex, sys

path = os.environ["_BEST20_SUMMARY_PATH"]
with open(path, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))

if row.get("status") and row["status"].strip() != "ok":
    print(f'echo "[warn] 首行 status={row["status"]!r}，仍使用该行超参。" >&2', file=sys.stderr)

for ev, key in [
    ("LR", "lr"),
    ("LORA_R", "lora_r"),
    ("LORA_ALPHA", "lora_alpha"),
    ("WEIGHT_DECAY", "weight_decay"),
    ("ADAM_BETA1", "adam_beta1"),
    ("ADAM_BETA2", "adam_beta2"),
]:
    v = (row.get(key) or "").strip()
    if not v:
        sys.exit(f"missing CSV column or empty value: {key}")
    print(f"export {ev}={shlex.quote(v)}")
PY
)"
unset _BEST20_SUMMARY_PATH

echo "[server_submit_distilbert_best_20ep] MODE=$MODE LORA_TYPE=$LORA_TYPE EPOCHS=$EPOCHS" >&2
echo "[server_submit_distilbert_best_20ep] SUMMARY_CSV=$SUMMARY_CSV" >&2
echo "[server_submit_distilbert_best_20ep] METRICS_DIR=$METRICS_DIR" >&2
echo "[server_submit_distilbert_best_20ep] LR=$LR LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA WEIGHT_DECAY=$WEIGHT_DECAY" >&2

bash distilbert/scripts/submit_bsub.sh
