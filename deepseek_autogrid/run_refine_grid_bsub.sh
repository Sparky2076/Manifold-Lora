#!/usr/bin/env bash
# DeepSeek LoRA **细网格**（基于第一轮 results/ 指标；写入 results_refine/）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export DEEPSEEK_GRID_CONFIG_MODULE="${DEEPSEEK_GRID_CONFIG_MODULE:-deepseek_autogrid.config_refine}"
export LORA_TYPE="${LORA_TYPE:-default}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/deepseek_autogrid/results_refine}"
export GRID_PID_FILE="${GRID_PID_FILE:-$PROJECT_DIR/deepseek_autogrid/.grid_submitter_refine.pid}"

exec bash "$SCRIPT_DIR/run_grid_bsub.sh"
