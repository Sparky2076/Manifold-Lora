#!/usr/bin/env bash
# 兼容入口：转发到 mLoRA **第四轮** 脚本（见 gs_sft_mlora_round4_lr.sh）。
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/gs_sft_mlora_round4_lr.sh" "$@"
