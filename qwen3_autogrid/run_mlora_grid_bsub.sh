#!/usr/bin/env bash
# Qwen3 coarse mLoRA (45, alpha=2r): knowledge_mc_mix → results_mmlu_mlora
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export DEEPSEEK_GRID_CONFIG_MODULE="${DEEPSEEK_GRID_CONFIG_MODULE:-qwen3_autogrid.config}"
export GRID_JOB_PREFIX="${GRID_JOB_PREFIX:-qwen3_mmlu_grid_mlora}"
export LORA_TYPE="${LORA_TYPE:-mlora}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora}"
export GRID_PID_FILE="${GRID_PID_FILE:-$PROJECT_DIR/qwen3_autogrid/.grid_submitter_mlora.pid}"

export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
export SFT_ENABLE_THINKING="${SFT_ENABLE_THINKING:-0}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

exec bash "$PROJECT_DIR/deepseek_autogrid/run_grid_bsub.sh"
