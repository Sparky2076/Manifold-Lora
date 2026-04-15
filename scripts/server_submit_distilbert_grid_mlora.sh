#!/usr/bin/env bash
# 服务器端：DistilBERT mLoRA 全因子网格提交（同 LoRA 网格参数空间）
# 用法:
#   cd ~/Manifold-Lora
#   export CONDA_ROOT="$HOME/miniconda3"
#   bash scripts/server_submit_distilbert_grid_mlora.sh
#
# 默认:
#   LORA_TYPE=mlora
#   RESULTS_ROOT=distilbert_autogrid/results_mlora

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

export LORA_TYPE="${LORA_TYPE:-mlora}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/distilbert_autogrid/results_mlora}"

echo "==> mLoRA grid submit: LORA_TYPE=${LORA_TYPE} RESULTS_ROOT=${RESULTS_ROOT}"
exec bash "$PROJECT_DIR/scripts/server_submit_distilbert_grid.sh"
