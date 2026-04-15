#!/usr/bin/env bash
# DeepSeek mLoRA 全同网格
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"
export LORA_TYPE="${LORA_TYPE:-mlora}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/deepseek_autogrid/results_mlora}"
exec bash "$PROJECT_DIR/scripts/server_submit_deepseek_grid.sh"
