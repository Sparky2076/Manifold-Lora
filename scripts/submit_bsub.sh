#!/usr/bin/env bash
# 在服务器上运行：提交 LSF 训练任务（单卡）
# 用法: cd ~/Manifold-Lora && bash scripts/submit_bsub.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

JOB_NAME="${JOB_NAME:-manifold_lora}"
QUEUE="${QUEUE:-gpu}"
NGPU=1
MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
METRICS_DIR="${METRICS_DIR:-$PROJECT_DIR}"

bsub -J "$JOB_NAME" \
  -o "%J.out" \
  -e "%J.err" \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=32G]" \
  -gpu "num=${NGPU}" \
  "bash $SCRIPT_DIR/run_train_bsub.sh '$MODEL_NAME' '$METRICS_DIR'"

echo "已提交任务，查看: bjobs"
