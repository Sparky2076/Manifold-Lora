#!/usr/bin/env bash
# 从服务器拉回 DistilBERT「最优超参 20 epoch」终局目录（train.csv / test.csv / run_meta.json 等）
# 对应提交脚本：server_submit_distilbert_best_lora_20ep.sh / server_submit_distilbert_best_mlora_20ep.sh
#
# 用法:
#   bash scripts/pull_distilbert_best_20ep.sh
#   SERVER=wangxiao@主机 REMOTE_DIR=Manifold-Lora bash scripts/pull_distilbert_best_20ep.sh
# 只拉 LoRA 或只拉 mLoRA:
#   PULL_WHICH=lora bash scripts/pull_distilbert_best_20ep.sh
#   PULL_WHICH=mlora bash scripts/pull_distilbert_best_20ep.sh

set -euo pipefail

SERVER="${SERVER:-wangxiao@202.121.138.221}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
PULL_WHICH="${PULL_WHICH:-both}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DIST="$PROJECT_DIR/distilbert"

_pull_dir() {
  local name="$1"
  local remote_path="$SERVER:~/$REMOTE_DIR/distilbert/$name"
  local local_parent="$DIST"
  mkdir -p "$local_parent"
  echo "拉取 $remote_path -> $local_parent/$name"
  scp -r "$remote_path" "$local_parent/"
}

mkdir -p "$DIST"

case "$PULL_WHICH" in
  lora)
    _pull_dir "results_final_best_lora_20ep"
    ;;
  mlora)
    _pull_dir "results_final_best_mlora_20ep"
    ;;
  both)
    _pull_dir "results_final_best_lora_20ep"
    _pull_dir "results_final_best_mlora_20ep"
    ;;
  *)
    echo "PULL_WHICH 应为 lora | mlora | both" >&2
    exit 1
    ;;
esac

echo "拉取完成。"
