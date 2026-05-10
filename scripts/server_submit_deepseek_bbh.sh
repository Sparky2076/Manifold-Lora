#!/usr/bin/env bash
# Submit one DeepSeek BBH eval job (reads METRICS_DIR or ADAPTER_PATH).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    [[ -f "$_try/etc/profile.d/conda.sh" ]] && { export CONDA_ROOT="$_try"; break; }
  done
fi

export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
exec bash "$PROJECT_DIR/deepseek/scripts/submit_bsub_bbh.sh"
