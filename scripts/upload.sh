#!/usr/bin/env bash
# 仅上传 DistilBERT / DeepSeek 网格所需文件（体积小、速度快）
# 含：根目录共享模块、distilbert/、distilbert_autogrid/、deepseek/、deepseek_autogrid/、scripts/
# 用法: bash scripts/upload.sh

set -euo pipefail

SERVER="${SERVER:-wangxiao@202.121.138.221}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -z "${USE_SSH_MULTIPLEX+x}" ]; then
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            USE_SSH_MULTIPLEX=0
            ;;
        *)
            USE_SSH_MULTIPLEX=1
            ;;
    esac
fi
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-/tmp/manifold-lora-upload-$(printf '%s' "$SERVER" | tr '@/:' '_').sock}"
SSH_MUX_OPTS=(-o ControlMaster=auto -o ControlPersist=10m -o "ControlPath=${SSH_CONTROL_PATH}")

_disable_ssh_multiplex() {
    USE_SSH_MULTIPLEX=0
    rm -f "$SSH_CONTROL_PATH" >/dev/null 2>&1 || true
}

_ssh() {
    if [ "${USE_SSH_MULTIPLEX}" = "1" ]; then
        if ssh "${SSH_MUX_OPTS[@]}" "$@"; then
            return 0
        fi
        echo "[upload.sh] SSH 复用失败，自动降级为普通 ssh/scp（本次会多一次密码输入）。" >&2
        _disable_ssh_multiplex
    fi
    ssh "$@"
}

_scp() {
    if [ "${USE_SSH_MULTIPLEX}" = "1" ]; then
        if scp "${SSH_MUX_OPTS[@]}" "$@"; then
            return 0
        fi
        echo "[upload.sh] SCP 复用失败，自动降级为普通 ssh/scp（本次会多一次密码输入）。" >&2
        _disable_ssh_multiplex
    fi
    scp "$@"
}

_close_master() {
    if [ "${USE_SSH_MULTIPLEX}" = "1" ]; then
        ssh "${SSH_MUX_OPTS[@]}" -O exit "$SERVER" >/dev/null 2>&1 || true
    fi
}
trap _close_master EXIT

echo "上传到 $SERVER:~/$REMOTE_DIR/ （增量；默认排除 results/）"

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
    return 1
}

_sync_all_without_rsync() {
    echo "[upload.sh] 执行单次打包上传（根目录共享文件 + distilbert/deepseek + scripts）。" >&2
    (cd "$PROJECT_DIR" && tar -cf - \
        --exclude='distilbert/results' \
        --exclude='distilbert_autogrid/results' \
        --exclude='distilbert_autogrid/results_mlora' \
        --exclude='deepseek/results' \
        --exclude='deepseek_autogrid/results' \
        --exclude='deepseek_autogrid/results_mlora' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        optimizers.py \
        lora.py \
        mlora.py \
        requirements.txt \
        distilbert \
        distilbert_autogrid \
        deepseek \
        deepseek_autogrid \
        scripts 2>/dev/null) | \
        _ssh "$SERVER" "mkdir -p ~/$REMOTE_DIR && tar -xf - -C ~/$REMOTE_DIR"
}

if command -v rsync >/dev/null 2>&1; then
    _scp "$PROJECT_DIR/optimizers.py" \
        "$PROJECT_DIR/lora.py" \
        "$PROJECT_DIR/mlora.py" \
        "$SERVER:~/$REMOTE_DIR/"

    [ -f "$PROJECT_DIR/requirements.txt" ] && _scp "$PROJECT_DIR/requirements.txt" "$SERVER:~/$REMOTE_DIR/"

    _sync_tree_incremental "$PROJECT_DIR/distilbert" "distilbert"
    _sync_tree_incremental "$PROJECT_DIR/distilbert_autogrid" "distilbert_autogrid"
    _sync_tree_incremental "$PROJECT_DIR/deepseek" "deepseek"
    _sync_tree_incremental "$PROJECT_DIR/deepseek_autogrid" "deepseek_autogrid"

    _scp "$PROJECT_DIR/scripts/upload.sh" \
        "$PROJECT_DIR/scripts/upload.ps1" \
        "$PROJECT_DIR/scripts/pull_results.sh" \
        "$PROJECT_DIR/scripts/pull_results.ps1" \
        "$PROJECT_DIR/scripts/pull_distilbert_best_20ep.sh" \
        "$PROJECT_DIR/scripts/pull_distilbert_best_20ep.ps1" \
        "$PROJECT_DIR/scripts/pull_deepseek_results.sh" \
        "$PROJECT_DIR/scripts/pull_deepseek_results.ps1" \
        "$PROJECT_DIR/scripts/refresh_results_and_publish.sh" \
        "$PROJECT_DIR/scripts/refresh_results_and_publish.ps1" \
        "$PROJECT_DIR/scripts/refresh_deepseek_results_and_publish.sh" \
        "$PROJECT_DIR/scripts/refresh_deepseek_results_and_publish.ps1" \
        "$PROJECT_DIR/scripts/commit_and_push.sh" \
        "$PROJECT_DIR/scripts/server_submit_distilbert_grid.sh" \
        "$PROJECT_DIR/scripts/server_submit_distilbert_grid_force.sh" \
        "$PROJECT_DIR/scripts/server_submit_distilbert_grid_mlora.sh" \
        "$PROJECT_DIR/scripts/server_submit_distilbert_best_20ep.sh" \
        "$PROJECT_DIR/scripts/server_submit_distilbert_best_lora_20ep.sh" \
        "$PROJECT_DIR/scripts/server_submit_distilbert_best_mlora_20ep.sh" \
        "$PROJECT_DIR/scripts/server_submit_deepseek_grid.sh" \
        "$PROJECT_DIR/scripts/server_submit_deepseek_grid_mlora.sh" \
        "$PROJECT_DIR/scripts/kill_distilbert_grid_bjobs.sh" \
        "$SERVER:~/$REMOTE_DIR/scripts/" 2>/dev/null || true
else
    _sync_all_without_rsync
fi

echo "上传完成。"
