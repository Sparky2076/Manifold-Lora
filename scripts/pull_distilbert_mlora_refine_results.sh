#!/usr/bin/env bash
# 拉回 mLoRA 第二轮结果。汇总请在服务器先执行一行：
#   cd ~/Manifold-Lora && python -m distilbert_autogrid.aggregate_results --results-root distilbert_autogrid/results_mlora_refine --allow-incomplete
set -euo pipefail
SERVER="${SERVER:-wangxiao@202.121.138.196}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/distilbert_autogrid/results_mlora_refine"
mkdir -p "$DST"
if command -v rsync >/dev/null 2>&1; then
  rsync -avz "$SERVER:~/$REMOTE_DIR/distilbert_autogrid/results_mlora_refine/" "$DST/"
else
  scp -r "$SERVER:~/$REMOTE_DIR/distilbert_autogrid/results_mlora_refine" "$(dirname "$DST")/"
fi
