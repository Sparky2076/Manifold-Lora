#!/usr/bin/env bash
# DeepSeek mLoRA 全同网格
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"
export LORA_TYPE="${LORA_TYPE:-mlora}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/deepseek_autogrid/results_mlora}"
# 仅补未跑完的组合（与默认行为一致）。若 shell 里曾 export GRID_RESUME=0，会整网重交并在旧版脚本上反复循环；
# 此处强制默认走「补跑」，避免误继承。若要刻意全量重跑，请用：
#   GRID_RESUME=0 bash scripts/server_submit_deepseek_grid.sh
# 并自行 export LORA_TYPE=mlora RESULTS_ROOT=.../results_mlora
export GRID_RESUME=1
exec bash "$PROJECT_DIR/scripts/server_submit_deepseek_grid.sh"
