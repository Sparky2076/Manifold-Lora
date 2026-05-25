#!/usr/bin/env bash
# Refine: smoke (coarse min-PPL hparams) → nohup 48-job refine grid.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export SUBMIT_REFINE_GRID_AFTER_SMOKE=1
export SKIP_CACHE=1
exec bash "$PROJECT_DIR/scripts/cluster_qwen3_refine_login_pipeline.sh"
