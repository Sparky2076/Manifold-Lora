#!/usr/bin/env bash
# 在本地 Git Bash / PowerShell 运行，上传代码到服务器
# 用法: bash scripts/upload.sh

set -euo pipefail

SERVER="${SERVER:-wangxiao@202.121.138.196}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "上传到 $SERVER:~/$REMOTE_DIR/"

scp "$PROJECT_DIR/optimizers.py" \
    "$PROJECT_DIR/lora.py" \
    "$PROJECT_DIR/mlora.py" \
    "$SERVER:~/$REMOTE_DIR/"

[ -f "$PROJECT_DIR/requirements.txt" ] && scp "$PROJECT_DIR/requirements.txt" "$SERVER:~/$REMOTE_DIR/"

# DistilBERT 分类：整包上传
scp -r "$PROJECT_DIR/distilbert" "$SERVER:~/$REMOTE_DIR/"

# DistilBERT 全因子网格（config + run_grid + run_grid_bsub）
scp -r "$PROJECT_DIR/distilbert_autogrid" "$SERVER:~/$REMOTE_DIR/"

# DeepSeek SFT：整包上传
scp -r "$PROJECT_DIR/deepseek" "$SERVER:~/$REMOTE_DIR/"

# 根目录共享脚本
scp "$PROJECT_DIR/scripts/upload.sh" \
    "$PROJECT_DIR/scripts/upload.ps1" \
    "$PROJECT_DIR/scripts/commit_and_push.sh" \
    "$SERVER:~/$REMOTE_DIR/scripts/" 2>/dev/null || true

echo "上传完成（含 distilbert/、distilbert_autogrid/、deepseek/）"
