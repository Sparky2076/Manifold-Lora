#!/usr/bin/env bash
# 仅上传 DistilBERT 分类 + 全因子网格所需文件（体积小、速度快）
# 含：根目录共享模块、distilbert/、distilbert_autogrid/、scripts/
# 用法: bash scripts/upload.sh

set -euo pipefail

SERVER="${SERVER:-wangxiao@202.121.138.196}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-/tmp/manifold-lora-upload-%r@%h:%p}"
SSH_OPTS=(-o ControlMaster=auto -o ControlPersist=10m -o "ControlPath=${SSH_CONTROL_PATH}")

_ssh() {
    ssh "${SSH_OPTS[@]}" "$@"
}

_scp() {
    scp "${SSH_OPTS[@]}" "$@"
}

_close_master() {
    _ssh -O exit "$SERVER" >/dev/null 2>&1 || true
}
trap _close_master EXIT

echo "上传到 $SERVER:~/$REMOTE_DIR/ （增量；默认排除 results/）"

_scp "$PROJECT_DIR/optimizers.py" \
    "$PROJECT_DIR/lora.py" \
    "$PROJECT_DIR/mlora.py" \
    "$SERVER:~/$REMOTE_DIR/"

[ -f "$PROJECT_DIR/requirements.txt" ] && _scp "$PROJECT_DIR/requirements.txt" "$SERVER:~/$REMOTE_DIR/"

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
    echo "[upload.sh] 未找到 rsync，回退为单次 tar+ssh 传输（仍排除 results/）。" >&2
    (cd "$src" && tar -cf - \
        --exclude='results' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        .) | _ssh "$SERVER" "mkdir -p ~/$REMOTE_DIR/$dst && tar -xf - -C ~/$REMOTE_DIR/$dst"
}

_sync_tree_incremental "$PROJECT_DIR/distilbert" "distilbert"
_sync_tree_incremental "$PROJECT_DIR/distilbert_autogrid" "distilbert_autogrid"

_scp "$PROJECT_DIR/scripts/upload.sh" \
    "$PROJECT_DIR/scripts/upload.ps1" \
    "$PROJECT_DIR/scripts/pull_results.sh" \
    "$PROJECT_DIR/scripts/pull_results.ps1" \
    "$PROJECT_DIR/scripts/refresh_results_and_publish.sh" \
    "$PROJECT_DIR/scripts/refresh_results_and_publish.ps1" \
    "$PROJECT_DIR/scripts/commit_and_push.sh" \
    "$PROJECT_DIR/scripts/server_submit_distilbert_grid.sh" \
    "$PROJECT_DIR/scripts/server_submit_distilbert_grid_force.sh" \
    "$PROJECT_DIR/scripts/server_submit_distilbert_grid_mlora.sh" \
    "$PROJECT_DIR/scripts/kill_distilbert_grid_bjobs.sh" \
    "$SERVER:~/$REMOTE_DIR/scripts/" 2>/dev/null || true

echo "上传完成。"
