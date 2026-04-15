#!/usr/bin/env bash
# 仅上传 DistilBERT 分类 + 全因子网格所需文件（体积小、速度快）
# 含：根目录共享模块、distilbert/、distilbert_autogrid/、scripts/
# 用法: bash scripts/upload.sh

set -euo pipefail

SERVER="${SERVER:-wangxiao@202.121.138.196}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "上传到 $SERVER:~/$REMOTE_DIR/ （增量；默认排除 results/）"

scp "$PROJECT_DIR/optimizers.py" \
    "$PROJECT_DIR/lora.py" \
    "$PROJECT_DIR/mlora.py" \
    "$SERVER:~/$REMOTE_DIR/"

[ -f "$PROJECT_DIR/requirements.txt" ] && scp "$PROJECT_DIR/requirements.txt" "$SERVER:~/$REMOTE_DIR/"

_sync_tree_incremental() {
    local src="$1"
    local dst="$2"
    if command -v rsync >/dev/null 2>&1; then
        # 增量同步：只传变化文件；排除实验结果与缓存，避免每次传 246+ 组 CSV。
        rsync -az \
            --exclude 'results/' \
            --exclude '__pycache__/' \
            --exclude '*.pyc' \
            "$src/" "$SERVER:~/$REMOTE_DIR/$dst/"
        return 0
    fi
    echo "[upload.sh] 未找到 rsync，回退为逐文件 scp（仍排除 results/）。" >&2
    while IFS= read -r f; do
        rel="${f#$src/}"
        remote_dir="$SERVER:~/$REMOTE_DIR/$dst/$(dirname "$rel")"
        ssh "$SERVER" "mkdir -p \"$HOME/$REMOTE_DIR/$dst/$(dirname "$rel")\"" >/dev/null 2>&1 || true
        scp "$f" "$remote_dir/"
    done < <(find "$src" -type f ! -path "*/results/*" ! -path "*/__pycache__/*" ! -name "*.pyc")
}

_sync_tree_incremental "$PROJECT_DIR/distilbert" "distilbert"
_sync_tree_incremental "$PROJECT_DIR/distilbert_autogrid" "distilbert_autogrid"

scp "$PROJECT_DIR/scripts/upload.sh" \
    "$PROJECT_DIR/scripts/upload.ps1" \
    "$PROJECT_DIR/scripts/pull_results.sh" \
    "$PROJECT_DIR/scripts/pull_results.ps1" \
    "$PROJECT_DIR/scripts/commit_and_push.sh" \
    "$PROJECT_DIR/scripts/server_submit_distilbert_grid.sh" \
    "$PROJECT_DIR/scripts/server_submit_distilbert_grid_force.sh" \
    "$PROJECT_DIR/scripts/kill_distilbert_grid_bjobs.sh" \
    "$SERVER:~/$REMOTE_DIR/scripts/" 2>/dev/null || true

echo "上传完成。"
