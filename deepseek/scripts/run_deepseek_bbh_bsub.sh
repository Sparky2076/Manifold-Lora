#!/usr/bin/env bash
# Compute node: merge LoRA (unless SKIP_MERGE=1) + run lm-eval BBH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

METRICS_DIR="${METRICS_DIR:-${ADAPTER_PATH:-}}"
if [[ -z "${METRICS_DIR}" ]]; then
  echo "[run_deepseek_bbh_bsub] set METRICS_DIR or ADAPTER_PATH to the run directory" >&2
  exit 1
fi

OUTPUT_JSON="${OUTPUT_JSON:-$METRICS_DIR/bbh_eval.json}"
TASKS="${TASKS:-bbh}"
NUM_FEWSHOT="${NUM_FEWSHOT:-3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
MODEL_NAME="${MODEL_NAME:-}"
TORCH_DTYPE="${TORCH_DTYPE:-}"
TRUST_FLAG=()
if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  TRUST_FLAG=(--trust_remote_code)
fi
REMERGE_FLAG=()
if [[ "${FORCE_REMERGE:-0}" == "1" ]]; then
  REMERGE_FLAG=(--force_remerge)
fi
SKIP_MERGE_ARGS=()
if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  SKIP_MERGE_ARGS=(--merged_hf_dir "${MERGED_HF_DIR:-$METRICS_DIR/model_merged_hf}")
fi

_resolve_conda_sh() {
  local r c
  for r in "${CONDA_ROOT:-}" "${CONDA_BASE:-}"; do
    [[ -n "$r" && -f "$r/etc/profile.d/conda.sh" ]] && { echo "$r/etc/profile.d/conda.sh"; return 0; }
  done
  if command -v conda >/dev/null 2>&1; then
    c="$(conda info --base 2>/dev/null || true)"
    [[ -n "$c" && -f "$c/etc/profile.d/conda.sh" ]] && { echo "$c/etc/profile.d/conda.sh"; return 0; }
  fi
  for r in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3" "/opt/conda"; do
    [[ -f "$r/etc/profile.d/conda.sh" ]] && { echo "$r/etc/profile.d/conda.sh"; return 0; }
  done
  return 1
}

CONDA_SH="$(_resolve_conda_sh)" || true
if [[ -z "${CONDA_SH:-}" || ! -f "$CONDA_SH" ]]; then
  echo "[run_deepseek_bbh_bsub] conda.sh not found, set CONDA_ROOT first." >&2
  exit 1
fi
source "$CONDA_SH"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch}"
conda activate "$CONDA_ENV_NAME"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
echo "[run_deepseek_bbh_bsub] host=$(hostname) METRICS_DIR=$METRICS_DIR SKIP_MERGE=${SKIP_MERGE:-0}" >&2

ARGS=(--metrics_dir "$METRICS_DIR" --output_json "$OUTPUT_JSON" --tasks "$TASKS" --num_fewshot "$NUM_FEWSHOT" --batch_size "$EVAL_BATCH_SIZE")
[[ "${#SKIP_MERGE_ARGS[@]}" -gt 0 ]] && ARGS+=("${SKIP_MERGE_ARGS[@]}")
[[ "${#REMERGE_FLAG[@]}" -gt 0 ]] && ARGS+=("${REMERGE_FLAG[@]}")
[[ -n "$MODEL_NAME" ]] && ARGS+=(--model_name "$MODEL_NAME")
[[ -n "$TORCH_DTYPE" ]] && ARGS+=(--torch_dtype "$TORCH_DTYPE")
[[ "${EVAL_LIMIT:-0}" != "0" ]] && ARGS+=(--limit "$EVAL_LIMIT")
[[ "${#TRUST_FLAG[@]}" -gt 0 ]] && ARGS+=("${TRUST_FLAG[@]}")

python -m deepseek.eval_bbh "${ARGS[@]}"
