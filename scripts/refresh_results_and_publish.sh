#!/usr/bin/env bash
# 一键：拉取服务器汇总文件 -> 校验是否跑满375 -> 本地汇总/分析 -> git 提交并推送
#
# 用法:
#   bash scripts/refresh_results_and_publish.sh
#   COMMIT_MSG="update full 375 results" bash scripts/refresh_results_and_publish.sh
# 可选:
#   SERVER=... REMOTE_DIR=...     传给 pull_results.sh
#   ALLOW_INCOMPLETE=1            允许未跑满时继续（默认严格要求 375）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "==> [1/5] 拉取服务器结果到本地"
bash "$SCRIPT_DIR/pull_results.sh"

STRICT_ARGS=()
if [[ "${ALLOW_INCOMPLETE:-0}" == "1" ]]; then
  STRICT_ARGS+=(--allow-incomplete)
fi

echo "==> [2/5] 本地汇总（默认要求跑满 375）"
python -m distilbert_autogrid.aggregate_results "${STRICT_ARGS[@]}"

echo "==> [3/5] 本地分析（默认要求跑满 375）"
python -m distilbert_autogrid.analyze_results "${STRICT_ARGS[@]}"

echo "==> [4/5] git 状态"
git status -sb

echo "==> [5/5] 提交并推送"
git add \
  distilbert_autogrid/results/summary.csv \
  distilbert_autogrid/results/missing_runs.csv \
  distilbert_autogrid/results/distilbert_grid_analysis.md \
  distilbert_autogrid/results/distilbert_grid_snapshot.md

if git diff --cached --quiet; then
  echo "没有结果文件变化，跳过 commit/push。"
  exit 0
fi

MSG="${COMMIT_MSG:-Update distilbert grid results (summary/missing/analysis)}"
git commit -m "$MSG"
git push
echo "完成。"
