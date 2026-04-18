#!/usr/bin/env bash
# 从服务器拉回 DistilBERT 网格汇总结果到本地（bash）
# 用法:
#   bash scripts/pull_results.sh
#   SERVER=wangxiao@202.121.138.221 REMOTE_DIR=Manifold-Lora bash scripts/pull_results.sh
# mLoRA 汇总在 results_mlora/ 时：
#   RESULTS_REL=distilbert_autogrid/results_mlora bash scripts/pull_results.sh

set -euo pipefail

SERVER="${SERVER:-wangxiao@202.121.138.221}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
# 相对仓库根：默认 LoRA 的 results/；mLoRA 用 distilbert_autogrid/results_mlora
RESULTS_REL="${RESULTS_REL:-distilbert_autogrid/results}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_RESULTS="$PROJECT_DIR/$RESULTS_REL"

mkdir -p "$LOCAL_RESULTS"
echo "从 $SERVER:~/$REMOTE_DIR/$RESULTS_REL 拉取到 $LOCAL_RESULTS"

scp "$SERVER:~/$REMOTE_DIR/$RESULTS_REL/summary.csv" "$LOCAL_RESULTS/"
scp "$SERVER:~/$REMOTE_DIR/$RESULTS_REL/missing_runs.csv" "$LOCAL_RESULTS/"

# 分析报告优先新路径；不存在时回退旧 docs 路径。
if ! scp "$SERVER:~/$REMOTE_DIR/$RESULTS_REL/distilbert_grid_analysis.md" "$LOCAL_RESULTS/" 2>/dev/null; then
  echo "results 下未找到分析报告，尝试旧路径 docs/ ..."
  scp "$SERVER:~/$REMOTE_DIR/docs/distilbert_grid_analysis.md" "$LOCAL_RESULTS/" 2>/dev/null || \
    echo "[warn] 未找到 distilbert_grid_analysis.md（results/ 和 docs/ 均不存在）"
fi

scp "$SERVER:~/$REMOTE_DIR/$RESULTS_REL/distilbert_grid_snapshot.md" "$LOCAL_RESULTS/" 2>/dev/null || true

echo "拉取完成。"
