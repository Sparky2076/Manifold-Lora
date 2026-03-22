#!/usr/bin/env bash
# 在本地 Git Bash / PowerShell 运行，上传代码到服务器
# 用法: bash scripts/upload.sh

set -euo pipefail

SERVER="${SERVER:-wangxiao@202.121.138.196}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "上传到 $SERVER:~/$REMOTE_DIR/"

scp "$PROJECT_DIR/main.py" \
    "$PROJECT_DIR/models.py" \
    "$PROJECT_DIR/utils.py" \
    "$PROJECT_DIR/optimizers.py" \
    "$PROJECT_DIR/lora.py" \
    "$PROJECT_DIR/mlora.py" \
    "$SERVER:~/$REMOTE_DIR/"

[ -f "$PROJECT_DIR/requirements.txt" ] && scp "$PROJECT_DIR/requirements.txt" "$SERVER:~/$REMOTE_DIR/"

# DeepSeek SFT：整包上传（含 scripts、results 占位、README）
scp -r "$PROJECT_DIR/deepseek" "$SERVER:~/$REMOTE_DIR/"

scp "$PROJECT_DIR/scripts/submit_bsub.sh" \
    "$PROJECT_DIR/scripts/run_train_bsub.sh" \
    "$PROJECT_DIR/scripts/watch_metrics.sh" \
    "$PROJECT_DIR/scripts/gs_lr_lora.sh" \
    "$PROJECT_DIR/scripts/gs_lr_mlora.sh" \
    "$SERVER:~/$REMOTE_DIR/scripts/" 2>/dev/null || true

echo "上传完成（含 deepseek/ 目录）"
