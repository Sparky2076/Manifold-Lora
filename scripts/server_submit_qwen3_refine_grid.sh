#!/usr/bin/env bash
# Qwen3 细网格 LoRA（48 jobs → results_mmlu_refine, alpha=2r）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    [[ -f "$_try/etc/profile.d/conda.sh" ]] && { export CONDA_ROOT="$_try"; break; }
  done
fi
if [[ -f "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-torch}"
fi
_torch_py="${CONDA_ROOT:-$HOME/miniconda3}/envs/${CONDA_ENV_NAME:-torch}/bin/python"
if [[ -x "$_torch_py" ]]; then
  export PATH="$(dirname "$_torch_py"):${PATH}"
fi

echo "==> repo: $PROJECT_DIR"
sed -i 's/\r$//' scripts/*.sh deepseek/scripts/*.sh deepseek_autogrid/*.sh qwen3_autogrid/*.sh 2>/dev/null || true

# shellcheck source=scripts/_cluster_lsf_env.sh
source "$PROJECT_DIR/scripts/_cluster_lsf_env.sh"

export GRID_RESUME="${GRID_RESUME:-1}"
export SUBMIT_SLEEP_SEC="${SUBMIT_SLEEP_SEC:-2}"
export GRID_MAX_PEND="${GRID_MAX_PEND:-1}"
export GRID_MAX_RUN="${GRID_MAX_RUN:-5}"
export GRID_POLL_SEC="${GRID_POLL_SEC:-15}"
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_EVAL_OFFLINE="${HF_EVAL_OFFLINE:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
export SFT_ENABLE_THINKING="${SFT_ENABLE_THINKING:-0}"
export TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
export LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
export EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-gpu15,gpu17,gpu18}"
if [[ -f "$PROJECT_DIR/scripts/cluster_hf_cache_env.sh" ]]; then
  # shellcheck source=scripts/cluster_hf_cache_env.sh
  source "$PROJECT_DIR/scripts/cluster_hf_cache_env.sh"
fi
_qwen_snap="$(find "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--Qwen--Qwen3-0.6B/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
if [[ -n "$_qwen_snap" && -f "$_qwen_snap/config.json" ]]; then
  export MODEL_NAME="$_qwen_snap"
else
  export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B}"
fi
echo "==> MODEL_NAME=$MODEL_NAME EXCLUDE_HOSTS=$EXCLUDE_HOSTS"
exec bash "$PROJECT_DIR/qwen3_autogrid/run_refine_grid_bsub.sh"
