#!/usr/bin/env bash
set -euo pipefail
SERVER="${SERVER:-wangxiao@202.121.138.196}"
REMOTE_DIR="${REMOTE_DIR:-Manifold-Lora}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_RESULTS="${LOCAL_RESULTS:-$PROJECT_DIR/deepseek_autogrid/results}"
mkdir -p "$LOCAL_RESULTS"
scp "$SERVER:~/$REMOTE_DIR/deepseek_autogrid/results/summary.csv" "$LOCAL_RESULTS/"
scp "$SERVER:~/$REMOTE_DIR/deepseek_autogrid/results/missing_runs.csv" "$LOCAL_RESULTS/" 2>/dev/null || true
scp "$SERVER:~/$REMOTE_DIR/deepseek_autogrid/results/deepseek_grid_analysis.md" "$LOCAL_RESULTS/" 2>/dev/null || true
echo "Done. pulled into $LOCAL_RESULTS"
