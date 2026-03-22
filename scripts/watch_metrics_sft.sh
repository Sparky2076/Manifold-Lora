#!/usr/bin/env bash
# 在服务器上实时查看 train_sft.csv / test_sft.csv（与 watch_metrics.sh 形式一致）
# 用法:
#   cd ~/Manifold-Lora && bash scripts/watch_metrics_sft.sh
# 若指标在子目录（如网格搜索）:
#   METRICS_DIR=results/sft_grid/testing_alpaca_small_lr_1e_5 bash scripts/watch_metrics_sft.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${METRICS_DIR:-$ROOT}"
cd "$DIR"

echo "目录: $DIR"
echo "每 5 秒刷新一次 train_sft.csv / test_sft.csv 末尾 5 行，按 Ctrl+C 退出。"

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
