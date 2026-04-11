#!/usr/bin/env bash
# 仅上传 DistilBERT 分类 + 全因子网格所需文件（体积小、速度快）
# 含：根目录共享模块、distilbert/、distilbert_autogrid/、scripts/
# 用法: bash scripts/upload.sh

set -euo pipefail

SERVER="${SERVER:-wangxiao@202.121.138.196}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "上传到 $SERVER:~/$REMOTE_DIR/ （DistilBERT + distilbert_autogrid + 根模块 + scripts）"

scp "$PROJECT_DIR/optimizers.py" \
    "$PROJECT_DIR/lora.py" \
    "$PROJECT_DIR/mlora.py" \
    "$SERVER:~/$REMOTE_DIR/"

[ -f "$PROJECT_DIR/requirements.txt" ] && scp "$PROJECT_DIR/requirements.txt" "$SERVER:~/$REMOTE_DIR/"

scp -r "$PROJECT_DIR/distilbert" "$SERVER:~/$REMOTE_DIR/"
scp -r "$PROJECT_DIR/distilbert_autogrid" "$SERVER:~/$REMOTE_DIR/"

scp "$PROJECT_DIR/scripts/upload.sh" \
    "$PROJECT_DIR/scripts/upload.ps1" \
    "$PROJECT_DIR/scripts/commit_and_push.sh" \
    "$PROJECT_DIR/scripts/server_submit_distilbert_grid.sh" \
    "$SERVER:~/$REMOTE_DIR/scripts/" 2>/dev/null || true

echo "上传完成。"
