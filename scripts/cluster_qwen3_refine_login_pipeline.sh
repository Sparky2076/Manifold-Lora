#!/usr/bin/env bash
# Login node: refine smoke (hparams from coarse min-PPL) → optional nohup refine grid 48.
set -euo pipefail
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"
# shellcheck source=scripts/cluster_hf_cache_env.sh
source "$PROJECT_DIR/scripts/cluster_hf_cache_env.sh"

SUBMIT_REFINE_GRID_AFTER_SMOKE="${SUBMIT_REFINE_GRID_AFTER_SMOKE:-0}"

if [[ -f "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-torch}"
fi

COARSE_SUMMARY="${COARSE_SUMMARY:-$PROJECT_DIR/qwen3_autogrid/results_mmlu/summary.csv}"
if [[ ! -f "$COARSE_SUMMARY" ]]; then
  echo "[qwen3-refine-pipeline] aggregate coarse first: python -m deepseek_autogrid.aggregate_results --config-module qwen3_autogrid.config" >&2
  exit 2
fi

eval "$(python3 "$PROJECT_DIR/scripts/pick_qwen3_coarse_best_hparams.py" --summary-csv "$COARSE_SUMMARY")"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_EVAL_OFFLINE="${HF_EVAL_OFFLINE:-1}"

SMOKE_DIR="${SMOKE_DIR:-$PROJECT_DIR/qwen3_autogrid/smoke_refine_knowledge_mc}"
echo "===== Phase C-refine: smoke lr=$SMOKE_LR r=$SMOKE_LORA_R a=$SMOKE_LORA_ALPHA wd=$SMOKE_WEIGHT_DECAY ====="
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"

# shellcheck source=scripts/_cluster_lsf_env.sh
source "$PROJECT_DIR/scripts/_cluster_lsf_env.sh"
_qwen_snap="$(find "${HF_HOME}/hub/models--Qwen--Qwen3-0.6B/snapshots" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
if [[ -n "$_qwen_snap" && -f "$_qwen_snap/config.json" ]]; then
  export MODEL_NAME="$_qwen_snap"
else
  export MODEL_NAME="Qwen/Qwen3-0.6B"
fi

export METRICS_DIR="$SMOKE_DIR"
export SFT_PRESET="knowledge_mc_mix"
export SFT_FORMAT="chat"
export SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
export MAX_STEPS=2
export EVAL_EVERY=2
export BATCH_SIZE=1
export GRAD_ACCUM_STEPS=1
export TRUST_REMOTE_CODE=1
export TORCH_DTYPE=bfloat16
export LORA_TYPE=default
export LR="$SMOKE_LR"
export LORA_R="$SMOKE_LORA_R"
export LORA_ALPHA="$SMOKE_LORA_ALPHA"
export WEIGHT_DECAY="$SMOKE_WEIGHT_DECAY"
export SFT_ENABLE_THINKING=0
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
export EXCLUDE_HOSTS="${SMOKE_EXCLUDE_HOSTS:-gpu15,gpu17,gpu18}"
export JOB_NAME="${SMOKE_JOB_NAME:-qwen3_refine_smoke}"

echo "[qwen3-refine-pipeline] bsub smoke METRICS_DIR=$METRICS_DIR"
_submit_out="$(bash "$PROJECT_DIR/deepseek/scripts/submit_bsub_sft.sh")"
echo "$_submit_out"
_smoke_jid="$(printf '%s\n' "$_submit_out" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | tail -1)"
[[ -n "$_smoke_jid" ]] || { echo "[qwen3-refine-pipeline] FAIL: no bsub job id" >&2; exit 3; }
echo "[qwen3-refine-pipeline] waiting job_id=$_smoke_jid ..."
while bjobs "$_smoke_jid" 2>/dev/null | grep -qE 'PEND|RUN|SSUSP|USUSP'; do
  sleep 30
done
[[ -f "$SMOKE_DIR/sft_lora_state.pt" ]] || {
  echo "[qwen3-refine-pipeline] FAIL: missing $SMOKE_DIR/sft_lora_state.pt (see ${_smoke_jid}.err)" >&2
  exit 3
}
python - "$SMOKE_DIR" <<'PY'
import csv, json, pathlib, sys
d = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((d / "test_sft.csv").open(encoding="utf-8")))
assert rows, "no test_sft rows"
meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
assert meta.get("sft_preset") == "knowledge_mc_mix"
print("[qwen3-refine-pipeline] Phase C OK preset=knowledge_mc_mix ppl=", rows[-1].get("eval_perplexity"))
PY

if [[ "${SUBMIT_REFINE_GRID_AFTER_SMOKE:-0}" != "1" ]]; then
  echo "[qwen3-refine-pipeline] Done. Submit refine grid: SUBMIT_REFINE_GRID_AFTER_SMOKE=1 bash $0"
  exit 0
fi

sed -i 's/\r$//' scripts/*.sh qwen3_autogrid/*.sh deepseek/scripts/*.sh deepseek_autogrid/*.sh 2>/dev/null || true
LOG="${QWEN3_REFINE_GRID_LOG:-$PROJECT_DIR/qwen3_mmlu_refine_grid_submit.log}"
echo "===== Phase D-refine: nohup refine grid (48) → $LOG ====="
nohup bash "$PROJECT_DIR/scripts/server_submit_qwen3_refine_grid.sh" >>"$LOG" 2>&1 &
echo "[qwen3-refine-pipeline] nohup pid=${!} log=$LOG"
