#!/usr/bin/env bash
# Login node: HF cache -> mLoRA knowledge_mc_mix smoke -> Phase D nohup coarse grid (45).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

export LORA_TYPE=mlora
export SUBMIT_GRID_AFTER_SMOKE=1
exec bash "$PROJECT_DIR/scripts/cluster_qwen3_login_lora_pipeline.sh"
