#!/usr/bin/env bash
# 集群登录节点：Qwen3（仅 MMLU 数据管线）Phase B cache（HF + datasets + tokenizer）→
# Phase C knowledge_mc_mix smoke → Phase D（可选）nohup 粗网格 45（alpha=2r）。
# Phase D 仅在 Phase C smoke 成功后执行（失败会 exit，不会投递网格）。
# Usage:
#   bash scripts/cluster_qwen3_login_lora_pipeline.sh                         # cache + smoke
#   SUBMIT_GRID_AFTER_SMOKE=1 bash scripts/cluster_qwen3_login_lora_pipeline.sh   # + nohup 粗网格 → results_mmlu
#   bash scripts/cluster_qwen3_smoke_then_coarse_grid.sh                         # 同上（强制 SUBMIT_GRID_AFTER_SMOKE=1）
#   SUBMIT_MMLU_GRID_AFTER_SMOKE=1 同上（旧名，与 SUBMIT_GRID_AFTER_SMOKE 等价）
#
# Env:
#   PIP_INSTALL_MISSING=1   – 缺 torch 等时 pip install
#   SKIP_CACHE=1           – 跳过 HF 预缓存
#   SKIP_SMOKE=1           – 跳过 smoke
#   MERGE_AFTER_SMOKE=1    – smoke 后可选 merge（登录节点可能 OOM）
#   SMOKE_DIR              – knowledge_mc_mix smoke 目录（默认 $PROJECT_DIR/qwen3_autogrid/smoke_knowledge_mc，NFS）
#   SMOKE_VIA_BSUB         – 1 强制 bsub smoke；auto（默认）在无 CUDA 登录节点走 bsub
#   GRID_*                  – 与 run_grid_bsub 相同
#
# HF (shared NFS cache; see scripts/cluster_hf_cache_env.sh):
#   HF_HOME / HF_DATASETS_CACHE — default /nfsshare/home/$USER/.cache/huggingface when that path exists
#   HF_HUB_DOWNLOAD_TIMEOUT — default 300s; optional HF_HUB_ETAG_TIMEOUT
#   FORCE_OFFICIAL_HF_ONLY / HF_ENDPOINT — mirror vs official hub (cache script)

set -euo pipefail
# Make nohup/cluster logs show Python progress line-by-line on NFS.
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

LORA_TYPE="${LORA_TYPE:-default}"
if [[ "${LORA_TYPE}" == "mlora" ]]; then
  _grid_submit_script="server_submit_qwen3_grid_mlora.sh"
  _grid_log_default="$PROJECT_DIR/qwen3_mmlu_mlora_grid_submit.log"
  _grid_pid_default="$PROJECT_DIR/qwen3_mmlu_mlora_grid_nohup.pid"
  _smoke_dir_default="$PROJECT_DIR/qwen3_autogrid/smoke_knowledge_mc_mlora"
  _smoke_job_default="qwen3_kmc_mlora_smoke"
else
  _grid_submit_script="server_submit_qwen3_grid.sh"
  _grid_log_default="$PROJECT_DIR/qwen3_mmlu_grid_submit.log"
  _grid_pid_default="$PROJECT_DIR/qwen3_mmlu_grid_nohup.pid"
  _smoke_dir_default="$PROJECT_DIR/qwen3_autogrid/smoke_knowledge_mc"
  _smoke_job_default="qwen3_kmc_smoke"
fi
# shellcheck source=scripts/cluster_hf_cache_env.sh
source "$PROJECT_DIR/scripts/cluster_hf_cache_env.sh"

SUBMIT_GRID_AFTER_SMOKE="${SUBMIT_GRID_AFTER_SMOKE:-0}"
SUBMIT_MMLU_GRID_AFTER_SMOKE="${SUBMIT_MMLU_GRID_AFTER_SMOKE:-0}"
if [[ "${SUBMIT_MMLU_GRID_AFTER_SMOKE}" == "1" ]]; then
  SUBMIT_GRID_AFTER_SMOKE=1
fi

if [[ "${PIP_INSTALL_MISSING:-0}" == "1" ]]; then
  pip install torch transformers datasets tqdm accelerate
fi

if ! python -c "import torch" 2>/dev/null; then
  echo "[qwen3-pipeline] ModuleNotFoundError: torch. Run: conda activate torch 或 PIP_INSTALL_MISSING=1 bash $0" >&2
  exit 2
fi

python <<'PY' || exit 6
from __future__ import annotations

import subprocess
import sys

import transformers
from packaging import version

if version.parse(transformers.__version__) < version.parse("4.51.0"):
    print(
        "[qwen3-pipeline] pip install transformers==4.51.3 "
        f"(have {transformers.__version__}) ..."
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==4.51.3",
            "huggingface_hub>=0.26.0,<1.0",
        ]
    )

import torch

if version.parse(torch.__version__) < version.parse("2.1.0"):
    print(
        "[qwen3-pipeline] pip install torch==2.1.2 "
        f"(have {torch.__version__}; Qwen3 modeling needs >=2.1) ..."
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "torch==2.1.2"]
    )
PY

if [[ "${SKIP_CACHE:-0}" != "1" ]]; then
  echo "===== Phase B: cache HF ====="
  bash "$PROJECT_DIR/scripts/cache_hf_qwen3_login.sh"
else
  echo "[qwen3-pipeline] SKIP_CACHE=1"
fi

# Login-node smoke/grid must not probe huggingface.co after cache (see cache offline verify).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_EVAL_OFFLINE="${HF_EVAL_OFFLINE:-1}"

SMOKE_DIR="${SMOKE_DIR:-$_smoke_dir_default}"
if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  export LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

  echo "===== Phase C: knowledge_mc_mix smoke (max_steps=2) ====="
  rm -rf "$SMOKE_DIR"
  mkdir -p "$SMOKE_DIR"

  _smoke_via_bsub=0
  if [[ "${SMOKE_VIA_BSUB:-auto}" == "1" ]]; then
    _smoke_via_bsub=1
  elif [[ "${SMOKE_VIA_BSUB:-auto}" == "auto" ]]; then
    if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      _smoke_via_bsub=1
    fi
  fi

  if [[ "$_smoke_via_bsub" == "1" ]]; then
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
    export MAX_STEPS=2
    export EVAL_EVERY=2
    export BATCH_SIZE=1
    export GRAD_ACCUM_STEPS=1
    export TRUST_REMOTE_CODE=1
    export TORCH_DTYPE=bfloat16
    export LORA_TYPE="${LORA_TYPE:-default}"
    export LORA_R=16
    export LORA_ALPHA=32
    export SFT_ENABLE_THINKING=0
    export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
    export EXCLUDE_HOSTS="${SMOKE_EXCLUDE_HOSTS:-gpu15,gpu17}"
    export JOB_NAME="${SMOKE_JOB_NAME:-$_smoke_job_default}"
    echo "[qwen3-pipeline] Phase C bsub smoke METRICS_DIR=$METRICS_DIR"
    _submit_out="$(JOB_NAME="$JOB_NAME" bash "$PROJECT_DIR/deepseek/scripts/submit_bsub_sft.sh")"
    echo "$_submit_out"
    _smoke_jid="$(printf '%s\n' "$_submit_out" | sed -n 's/.*Job <\([0-9][0-9]*\)>.*/\1/p' | tail -1)"
    if [[ -z "$_smoke_jid" ]]; then
      echo "[qwen3-pipeline] FAIL: could not parse bsub job id from submit output" >&2
      exit 3
    fi
    echo "[qwen3-pipeline] waiting for smoke bsub job_id=$_smoke_jid ..."
    while bjobs "$_smoke_jid" 2>/dev/null | grep -qE 'PEND|RUN|SSUSP|USUSP'; do
      sleep 30
    done
    if [[ ! -f "$SMOKE_DIR/sft_lora_state.pt" ]]; then
      echo "[qwen3-pipeline] FAIL: bsub smoke ended without $SMOKE_DIR/sft_lora_state.pt (check ${_smoke_jid}.err)" >&2
      exit 3
    fi
  else
    python -m deepseek.main_sft \
      --model_name Qwen/Qwen3-0.6B \
      --trust_remote_code \
      --torch_dtype bfloat16 \
      --lora_targets "$LORA_TARGETS" \
      --sft_preset knowledge_mc_mix \
      --sft_format chat \
      --max_steps 2 \
      --eval_every 2 \
      --batch_size 1 \
      --grad_accum_steps 1 \
      --lora_type "${LORA_TYPE:-default}" \
      --lora_r 16 \
      --lora_alpha 32 \
      --metrics_dir "$SMOKE_DIR"
  fi

  [[ -f "$SMOKE_DIR/sft_lora_state.pt" ]] || {
    echo "[qwen3-pipeline] FAIL: missing $SMOKE_DIR/sft_lora_state.pt" >&2
    exit 3
  }
  python - "$SMOKE_DIR" <<'PY'
import csv
import json
import pathlib
import sys

d = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((d / "test_sft.csv").open(encoding="utf-8")))
assert rows, "no rows in test_sft"
meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
assert meta.get("sft_preset") == "knowledge_mc_mix", meta.get("sft_preset")
assert meta.get("lora_targets"), "run_meta missing lora_targets"
print("[qwen3-pipeline] Phase C OK preset=knowledge_mc_mix lora_targets=", meta.get("lora_targets"))
PY

  if [[ "${MERGE_AFTER_SMOKE:-0}" == "1" ]]; then
    echo "[qwen3-pipeline] optional merge smoke"
    python -m deepseek.export_merged_hf --metrics_dir "$SMOKE_DIR" --trust_remote_code \
      || echo "[qwen3-pipeline] merge smoke failed (SKIP if OOM)"
  fi
else
  echo "[qwen3-pipeline] SKIP_SMOKE=1"
fi

if [[ "${SUBMIT_GRID_AFTER_SMOKE:-0}" != "1" ]]; then
  echo "[qwen3-pipeline] Done. 投递粗网格: SUBMIT_GRID_AFTER_SMOKE=1 bash $0"
  exit 0
fi

sed -i 's/\r$//' scripts/*.sh qwen3_autogrid/*.sh deepseek/scripts/*.sh deepseek_autogrid/*.sh 2>/dev/null || true

LOG="${QWEN3_GRID_SUBMIT_LOG:-$_grid_log_default}"
echo "===== Phase D: nohup coarse grid (45, alpha=2r, lora_type=${LORA_TYPE}) → $LOG ====="
nohup bash "$PROJECT_DIR/scripts/$_grid_submit_script" >>"$LOG" 2>&1 &
GRID_PID=$!
echo "$GRID_PID" >"${QWEN3_GRID_PIDFILE:-$_grid_pid_default}"
echo "[qwen3-pipeline] nohup pid=$GRID_PID log=$LOG pidfile=${QWEN3_GRID_PIDFILE:-$_grid_pid_default} lora_type=${LORA_TYPE}"
