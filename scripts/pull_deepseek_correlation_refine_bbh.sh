#!/usr/bin/env bash
# Pull bbh_eval.json for the stable Top-5 correlation-refine runs (README list B).
# Usage: bash scripts/pull_deepseek_correlation_refine_bbh.sh
# Override: SERVER=user@host REMOTE_BASE=/path/to/results_correlation_refine
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER="${SERVER:-wangxiao@202.121.138.196}"
REMOTE_BASE="${REMOTE_BASE:-/nfsshare/home/wangxiao/Manifold-Lora/deepseek_autogrid/results_correlation_refine}"
REMOTE_BASE="${REMOTE_BASE%/}"
LOCAL_BASE="$PROJECT_DIR/deepseek_autogrid/results_correlation_refine"

RUNS=(
  lr_1p0673e-03_r32_a16_st500_wd_1p0000e-02
  lr_2p0000e-04_r64_a32_st500_wd_1p0000e-02
  lr_4p6203e-04_r64_a16_st500_wd_1p0000e-02
  lr_5p6961e-04_r32_a32_st500_wd_1p0000e-02
  lr_7p0224e-04_r64_a32_st500_wd_1p0000e-02
)

echo "Pulling bbh_eval.json from ${SERVER}:${REMOTE_BASE}/ -> ${LOCAL_BASE}"
for r in "${RUNS[@]}"; do
  mkdir -p "${LOCAL_BASE}/${r}"
  echo "  ${SERVER}:${REMOTE_BASE}/${r}/bbh_eval.json"
  scp "${SERVER}:${REMOTE_BASE}/${r}/bbh_eval.json" "${LOCAL_BASE}/${r}/"
done

echo "Done. Regenerate leaderboard:"
echo "  python scripts/summarize_deepseek_bbh_results.py --results-root deepseek_autogrid/results_correlation_refine --summary-csv deepseek_autogrid/results_correlation_refine/summary.csv"
