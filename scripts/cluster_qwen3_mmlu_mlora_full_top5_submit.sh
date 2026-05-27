#!/usr/bin/env bash
# nohup: mLoRA Top-5 full MMLU from refine summary_mmlu.csv
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export LORA_TYPE=mlora
export SUMMARY_CSV="${SUMMARY_CSV:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora_refine/summary_mmlu.csv}"
export RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora_refine}"
export PLOT_DIR="${PLOT_DIR:-$PROJECT_DIR/qwen3_autogrid/results_mmlu_mlora_refine/topk5_plot_bundle}"
export QWEN3_MMLU_FULL_TOP5_LOG="${QWEN3_MMLU_FULL_TOP5_LOG:-$PROJECT_DIR/qwen3_mmlu_mlora_full_top5_submit.log}"
export QWEN3_MMLU_FULL_TOP5_PIDFILE="${QWEN3_MMLU_FULL_TOP5_PIDFILE:-$PROJECT_DIR/qwen3_mmlu_mlora_full_top5_nohup.pid}"
exec bash "$PROJECT_DIR/scripts/cluster_qwen3_mmlu_full_top5_submit.sh"
