#!/usr/bin/env bash
# Export model_merged_hf for Top-K DeepSeek runs (after aggregate, before BBH).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_DIR/deepseek_autogrid/results/summary.csv}"
TOP_K="${TOP_K:-10}"
RESULTS_ROOT="${RESULTS_ROOT:-${ADAPTER_ROOT:-$PROJECT_DIR/deepseek_autogrid/results}}"

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

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    [[ -f "$_try/etc/profile.d/conda.sh" ]] && { export CONDA_ROOT="$_try"; break; }
  done
fi

CONDA_SH="$(_resolve_conda_sh)" || true
if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "${CONDA_ENV_NAME:-torch}" 2>/dev/null || true
fi

echo "[export-merged-topk] summary=$SUMMARY_CSV top_k=$TOP_K results_root=$RESULTS_ROOT"

while IFS= read -r run; do
  [[ -z "${run}" ]] && continue
  MD="${RESULTS_ROOT%/}/${run}"
  echo "[export-merged-topk] merge -> model_merged_hf: $MD"
  EXP_ARGS=(--metrics_dir "$MD")
  [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]] && EXP_ARGS+=(--trust_remote_code)
  python -m deepseek.export_merged_hf "${EXP_ARGS[@]}"
done < <(python "$PROJECT_DIR/scripts/pick_topk_deepseek_runs.py" --summary-csv "$SUMMARY_CSV" --top-k "$TOP_K" --require-status-ok --lora-type default)

echo "[export-merged-topk] done"
