#!/usr/bin/env bash
# DeepSeek factorial grid — LoRA only (explicit defaults for two-phase BBH workflow).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export LORA_TYPE="${LORA_TYPE:-default}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/deepseek_autogrid/results}"

exec bash "$SCRIPT_DIR/run_grid_bsub.sh"
