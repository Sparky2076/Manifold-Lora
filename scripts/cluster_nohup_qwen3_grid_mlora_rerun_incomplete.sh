#!/usr/bin/env bash
# Restart mLoRA coarse-grid submitter (nohup). Does NOT kill RUN/PEND bjobs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for f in \
  scripts/server_submit_qwen3_grid_mlora.sh \
  scripts/cluster_hf_cache_env.sh \
  deepseek/scripts/submit_bsub_sft.sh \
  deepseek/scripts/run_deepseek_sft_bsub.sh \
  deepseek/main_sft.py \
  deepseek/models_sft.py \
  deepseek_autogrid/run_grid_bsub.sh \
  qwen3_autogrid/run_mlora_grid_bsub.sh
do
  [[ -f "$f" ]] && sed -i 's/\r$//' "$f" || true
done

PIDFILE="${ROOT}/qwen3_autogrid/.grid_submitter_mlora.pid"
if [[ -f "$PIDFILE" ]]; then
  old_pid="$(tr -d ' \n\r' <"$PIDFILE" || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[qwen3-mlora-rerun] stop old grid submitter pid=$old_pid"
    kill "$old_pid" 2>/dev/null || true
    sleep 2
  fi
fi

export GRID_RESUME=1
export LORA_TYPE=mlora
LOG="${QWEN3_MMLU_MLORA_GRID_RERUN_LOG:-${HOME}/qwen3_mmlu_mlora_grid_rerun.log}"
{
  echo "===== qwen3 mLoRA grid rerun submitter $(date -Is) ====="
} >>"$LOG"
nohup bash "$ROOT/scripts/server_submit_qwen3_grid_mlora.sh" >>"$LOG" 2>&1 &
echo "[qwen3-mlora-rerun] nohup pid=$! log=$LOG"
