#!/usr/bin/env bash
# 在服务器上实时查看 train.csv / test.csv 末尾几行（DistilBERT 分类）
# 用法（SSH 登录后）:
#   cd ~/Manifold-Lora
#   bash distilbert/scripts/watch_metrics.sh
# 子目录实验:
#   METRICS_DIR=distilbert/results/class_grid/foo bash distilbert/scripts/watch_metrics.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
METRICS_DIR="${METRICS_DIR:-$PROJECT_DIR/distilbert/results}"

echo "METRICS_DIR=$METRICS_DIR"
echo "每 5 秒刷新一次 train.csv / test.csv 末尾 5 行，按 Ctrl+C 退出。"

while true; do
  clear
  echo "====== train.csv (tail -n 5) ======"
  if [ -f "$METRICS_DIR/train.csv" ]; then
    tail -n 5 "$METRICS_DIR/train.csv"
  else
    echo "train.csv 暂不存在或尚未写入"
  fi

  echo
  echo "====== test.csv (tail -n 5) ======"
  if [ -f "$METRICS_DIR/test.csv" ]; then
    tail -n 5 "$METRICS_DIR/test.csv"
  else
    echo "test.csv 暂不存在或尚未写入"
  fi

  sleep 5
done
