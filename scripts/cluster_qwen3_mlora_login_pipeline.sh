#!/usr/bin/env bash
# mLoRA coarse grid only (skip cache/smoke). Set SUBMIT_GRID_AFTER_SMOKE=1 before sourcing.
export LORA_TYPE=mlora
export SUBMIT_GRID_AFTER_SMOKE=1
export SKIP_CACHE=1
export SKIP_SMOKE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
exec bash "$PROJECT_DIR/scripts/cluster_qwen3_login_lora_pipeline.sh"
