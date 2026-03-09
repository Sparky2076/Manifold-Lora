#!/usr/bin/env bash
# 在服务器上实时查看 train.csv / test.csv 末尾几行
# 用法（SSH 登录后）:
#   cd ~/Manifold-Lora
#   bash scripts/watch_metrics.sh

cd "$(dirname "$0")/.."

echo "每 5 秒刷新一次 train.csv / test.csv 末尾 5 行，按 Ctrl+C 退出。"

while true; do
  clear
  echo "====== train.csv (tail -n 5) ======"
  if [ -f train.csv ]; then
    tail -n 5 train.csv
  else
    echo "train.csv 暂不存在或尚未写入"
  fi

  echo
  echo "====== test.csv (tail -n 5) ======"
  if [ -f test.csv ]; then
    tail -n 5 test.csv
  else
    echo "test.csv 暂不存在或尚未写入"
  fi

  sleep 5
done

