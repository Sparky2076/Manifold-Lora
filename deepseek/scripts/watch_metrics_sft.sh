#!/usr/bin/env bash
# 实时查看 train_sft.csv / test_sft.csv（与根目录 scripts/watch_metrics.sh 形式一致）
# 用法:
#   cd ~/Manifold-Lora && bash deepseek/scripts/watch_metrics_sft.sh
# 网格子目录:
#   METRICS_DIR=deepseek/results/sft_grid/testing_alpaca_small_lr_2e_5 bash deepseek/scripts/watch_metrics_sft.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIR="${METRICS_DIR:-$ROOT/deepseek/results}"
cd "$DIR"

echo "目录: $DIR"
echo "每 5 秒刷新 train_sft.csv / test_sft.csv 末尾 5 行，按 Ctrl+C 退出。"

while true; do
  clear
  echo "====== train_sft.csv (tail -n 5) ======"
  if [ -f train_sft.csv ]; then
    tail -n 5 train_sft.csv
  else
    echo "train_sft.csv 暂不存在或尚未写入"
  fi

  echo
  echo "====== test_sft.csv (tail -n 5) ======"
  if [ -f test_sft.csv ]; then
    tail -n 5 test_sft.csv
  else
    echo "test_sft.csv 暂不存在或尚未写入"
  fi

  sleep 5
done
