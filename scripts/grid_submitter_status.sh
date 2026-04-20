#!/usr/bin/env bash
# 查看本仓库网格「提交循环」进程（run_grid_bsub）是否在跑、PID 是多少。
# 用法: bash scripts/grid_submitter_status.sh
# 依赖: deepseek/distilbert 的 run_grid_bsub 启动时会写入 .grid_submitter.pid

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "hostname: $(hostname)"
echo ""

for f in deepseek_autogrid/.grid_submitter.pid distilbert_autogrid/.grid_submitter.pid; do
  [[ -f "$f" ]] || continue
  pid="$(tr -d ' \n' < "$f")"
  echo "=== $f -> pid=$pid ==="
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    ps -p "$pid" -o pid,ppid,etime,cmd 2>/dev/null || true
  else
    echo "(无此进程或 PID 文件过期；可 rm -f $f)"
  fi
  echo ""
done

echo "本机实时搜索（与登录节点一致时有效）:"
pgrep -af run_grid_bsub 2>/dev/null || echo "(pgrep 无匹配)"
