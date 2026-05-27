#!/usr/bin/env bash
# mLoRA coarse tinyMMLU eval (45 runs)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export LORA_TYPE=mlora
exec bash "$PROJECT_DIR/scripts/server_submit_qwen3_mmlu_coarse_eval.sh"
