#!/usr/bin/env bash
# DeepSeek 全因子网格（LoRA 默认）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    [[ -f "$_try/etc/profile.d/conda.sh" ]] && { export CONDA_ROOT="$_try"; break; }
  done
fi

echo "==> repo: $PROJECT_DIR"
echo "==> [1/2] sed CRLF -> LF"
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh deepseek/scripts/*.sh deepseek_autogrid/*.sh 2>/dev/null || true

echo "==> [2/2] run deepseek_autogrid/run_grid_bsub.sh (LORA_TYPE=${LORA_TYPE:-default})"
exec bash "$PROJECT_DIR/deepseek_autogrid/run_grid_bsub.sh"
