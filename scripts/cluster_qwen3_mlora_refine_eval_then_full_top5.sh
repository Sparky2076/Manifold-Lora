#!/usr/bin/env bash
# mLoRA: wait refine 48/48 → aggregate → refine tinyMMLU → join → full Top-5 MMLU
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-torch}"
source "$ROOT/scripts/_cluster_lsf_env.sh" 2>/dev/null || true

export LORA_TYPE=mlora
export EXCLUDE_HOSTS="${EXCLUDE_HOSTS:-gpu15,gpu17,gpu18}"
REFINE="$ROOT/qwen3_autogrid/results_mmlu_mlora_refine"
LOG="${HOME}/qwen3_mlora_phase5_6_orchestrator.log"
exec >>"$LOG" 2>&1

_refine_complete() {
  LORA_TYPE=mlora python3 -c "
import importlib, csv
from pathlib import Path
m=importlib.import_module('qwen3_autogrid.config_refine')
root=Path('qwen3_autogrid/results_mmlu_mlora_refine')
st,ev=m.MAX_STEPS_DEFAULT,m.EVAL_EVERY_DEFAULT
exp=st//ev
n=0
for lr,r,a,wd in m.iter_grid():
 d=root/m.run_dir_name(lr,r,a,st,wd)
 pt,csv=d/'sft_lora_state.pt',d/'test_sft.csv'
 if not (pt.is_file() and csv.is_file()): continue
 recs=list(csv.DictReader(csv.open()))
 if len(recs)<exp: continue
 if max(int(float(r['iteration'])) for r in recs)<st: continue
 n+=1
print(n)
"
}

echo "===== phase5-6 orchestrator start $(date -Is) ====="
for i in $(seq 1 720); do
  n=$(_refine_complete)
  echo "$(date +%H:%M:%S) refine_sft=${n}/48"
  [[ "$n" -ge 48 ]] && break
  sleep 120
done
n=$(_refine_complete)
if [[ "$n" -lt 48 ]]; then
  echo "TIMEOUT refine SFT ${n}/48"
  exit 2
fi

echo "===== aggregate refine ====="
python3 -m deepseek_autogrid.aggregate_results \
  --config-module qwen3_autogrid.config_refine \
  --results-root qwen3_autogrid/results_mmlu_mlora_refine

echo "===== start refine tinyMMLU eval ====="
bash "$ROOT/scripts/server_submit_qwen3_mmlu_mlora_refine_eval.sh"

for i in $(seq 1 720); do
  ev=$(find "$REFINE" -name mmlu_eval.json 2>/dev/null | wc -l)
  echo "$(date +%H:%M:%S) refine_tinyeval=${ev}/48"
  [[ "$ev" -ge 48 ]] && break
  sleep 120
done
ev=$(find "$REFINE" -name mmlu_eval.json 2>/dev/null | wc -l)
if [[ "$ev" -lt 48 ]]; then
  echo "TIMEOUT refine tinyeval ${ev}/48"
  exit 3
fi

echo "===== join summary_mmlu.csv ====="
python3 -m deepseek_autogrid.aggregate_mmlu_results --results-root "$REFINE"
python3 "$ROOT/scripts/join_sft_mmlu_summary.py" \
  --sft-summary "$REFINE/summary.csv" \
  --mmlu-summary "$REFINE/mmlu_summary.csv" \
  --output "$REFINE/summary_mmlu.csv"

echo "===== full Top-5 MMLU ====="
bash "$ROOT/scripts/cluster_qwen3_mmlu_mlora_full_top5_submit.sh"
echo "===== phase5-6 orchestrator done $(date -Is) ====="
