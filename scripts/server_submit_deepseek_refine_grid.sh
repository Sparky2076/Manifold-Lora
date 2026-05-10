#!/usr/bin/env bash
# DeepSeek LoRA 细网格（config_refine → results_refine/），与第一轮粗网格独立。
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
echo "==> [1/2] sed CRLF -> LF (DeepSeek scripts)"
sed -i 's/\r$//' scripts/*.sh deepseek/scripts/*.sh deepseek_autogrid/*.sh 2>/dev/null || true

echo "==> [2/2] DeepSeek refine grid → deepseek_autogrid/results_refine/"
exec bash "$PROJECT_DIR/deepseek_autogrid/run_refine_grid_bsub.sh"
