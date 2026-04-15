#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
bash "$SCRIPT_DIR/pull_deepseek_results.sh"
STRICT_ARGS=()
[[ "${ALLOW_INCOMPLETE:-0}" == "1" ]] && STRICT_ARGS+=(--allow-incomplete)
python -m deepseek_autogrid.aggregate_results "${STRICT_ARGS[@]}"
python -m deepseek_autogrid.analyze_results "${STRICT_ARGS[@]}"
git add deepseek_autogrid/results/summary.csv deepseek_autogrid/results/missing_runs.csv deepseek_autogrid/results/deepseek_grid_analysis.md deepseek_autogrid/results/deepseek_grid_snapshot.md
if git diff --cached --quiet; then
  echo "No deepseek result changes."; exit 0
fi
git commit --trailer "Made-with: Cursor" -m "${COMMIT_MSG:-Update deepseek grid results}"
git push
echo "Done."
