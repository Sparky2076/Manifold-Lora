#!/usr/bin/env bash
# Qwen3: lm-eval full MMLU Top-K from summary_mmlu.csv (proxy metric: tinyMMLU column mmlu_mean_acc).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

# shellcheck source=/dev/null
source "$PROJECT_DIR/scripts/_cluster_lsf_env.sh"

LORA_TYPE="${LORA_TYPE:-default}"
if [[ "${LORA_TYPE}" == "mlora" ]]; then
  SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora_refine/summary_mmlu.csv}"
  RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora_refine}"
  PICK_LORA_TYPE="${PICK_LORA_TYPE:-mlora}"
else
  SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_refine/summary_mmlu.csv}"
  RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_refine}"
  PICK_LORA_TYPE="${PICK_LORA_TYPE:-default}"
fi
TOP_K="${TOP_K:-5}"
EXPORT_MERGED_FIRST="${EXPORT_MERGED_FIRST:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
TASKS="${TASKS:-mmlu}"
NUM_FEWSHOT="${NUM_FEWSHOT:-5}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
OUTPUT_JSON_REL="${OUTPUT_JSON_REL:-mmlu_eval_full.json}"
PROGRESS_EVERY="${PROGRESS_EVERY:-0}"
PICK_METRIC="${PICK_METRIC:-mmlu_mean_acc}"
PICK_SORT="${PICK_SORT:-desc}"
EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-gpu15,gpu17,gpu18}"
SKIP_IF_EVAL_DONE="${SKIP_IF_EVAL_DONE:-0}"

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    [[ -f "$_try/etc/profile.d/conda.sh" ]] && { export CONDA_ROOT="$_try"; break; }
  done
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_EVAL_OFFLINE="${HF_EVAL_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

echo "[qwen3-mmlu-topk] summary=$SUMMARY_CSV top_k=$TOP_K tasks=$TASKS fewshot=$NUM_FEWSHOT output=$OUTPUT_JSON_REL pick=${PICK_METRIC}/${PICK_SORT} progress_every=$PROGRESS_EVERY"

if [[ "${EXPORT_MERGED_FIRST}" == "1" ]]; then
  SUMMARY_CSV="$SUMMARY_CSV" TOP_K="$TOP_K" RESULTS_ROOT="$RESULTS_ROOT" TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
    PICK_METRIC="$PICK_METRIC" PICK_SORT="$PICK_SORT" PICK_LORA_TYPE="$PICK_LORA_TYPE" \
    bash "$PROJECT_DIR/scripts/export_merged_deepseek_topk.sh"
fi

while IFS= read -r run; do
  [[ -z "${run}" ]] && continue
  export JOB_NAME="qwen3_mmlu_${run}"
  export METRICS_DIR="${RESULTS_ROOT%/}/${run}"
  export ADAPTER_PATH="$METRICS_DIR"
  export OUTPUT_JSON="${METRICS_DIR}/${OUTPUT_JSON_REL}"
  export PROGRESS_CSV="${METRICS_DIR}/mmlu_eval_full_progress.csv"
  export SKIP_MERGE=1
  export MERGED_HF_DIR="${METRICS_DIR}/model_merged_hf"
  export TRUST_REMOTE_CODE
  export TASKS
  export NUM_FEWSHOT
  export EVAL_LIMIT
  export PROGRESS_EVERY
  export EXCLUDE_HOSTS

  if [[ "${SKIP_IF_EVAL_DONE}" == "1" && -f "$OUTPUT_JSON" ]]; then
    if python3 -c "import json,math; d=json.load(open('$OUTPUT_JSON')); v=float(d.get('mmlu_mean_acc', float('nan'))); exit(0 if v==v else 1)" 2>/dev/null; then
      echo "[qwen3-mmlu-topk] SKIP done: $run"
      continue
    fi
  fi

  echo "[qwen3-mmlu-topk] submit $run -> $OUTPUT_JSON"
  bash scripts/server_submit_deepseek_bbh.sh
done < <(python "$PROJECT_DIR/scripts/pick_topk_deepseek_runs.py" --summary-csv "$SUMMARY_CSV" --top-k "$TOP_K" \
  --require-status-ok --lora-type "$PICK_LORA_TYPE" --metric "$PICK_METRIC" --sort "$PICK_SORT")

echo "[qwen3-mmlu-topk] done"
