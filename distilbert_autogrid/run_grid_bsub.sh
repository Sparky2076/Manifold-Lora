#!/usr/bin/env bash
# Submit full DistilBERT LoRA grid as separate LSF jobs (lr × r × alpha × wd; betas fixed in config).
# Grid definition lives in distilbert_autogrid/config.py only.
#
# Usage (cluster):
#   cd ~/Manifold-Lora && sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh
#   bash distilbert_autogrid/run_grid_bsub.sh
#
# Optional env: RESULTS_ROOT, EPOCHS, QUEUE, LORA_TYPE, LORA_DROPOUT, BATCH_SIZE, GRAD_ACCUM_STEPS
# Resume: GRID_RESUME=1 (default) skips combos whose test.csv already has >= EPOCHS eval rows; GRID_RESUME=0 submits all.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

RESULTS_ROOT="${RESULTS_ROOT:-$PROJECT_DIR/distilbert_autogrid/results}"
GRID_RESUME="${GRID_RESUME:-1}"
export EPOCHS="${EPOCHS:-$(python -c "from distilbert_autogrid.config import EPOCHS_DEFAULT; print(EPOCHS_DEFAULT)")}"
export ADAM_BETA1="${ADAM_BETA1:-$(python -c "from distilbert_autogrid.config import ADAM_BETA1_FIXED; print(ADAM_BETA1_FIXED)")}"
export ADAM_BETA2="${ADAM_BETA2:-$(python -c "from distilbert_autogrid.config import ADAM_BETA2_FIXED; print(ADAM_BETA2_FIXED)")}"

python -c "
import os
from distilbert_autogrid.config import iter_grid, run_dir_name, EPOCHS_DEFAULT
ep = int(os.environ.get('EPOCHS', str(EPOCHS_DEFAULT)))
for lr, r, alpha, wd in iter_grid():
    name = run_dir_name(lr, r, alpha, ep, wd)
    print(f'{name}\t{lr}\t{r}\t{alpha}\t{wd}')
" | while IFS=$'\t' read -r NAME LR R A WD; do
  METRICS_DIR="$RESULTS_ROOT/$NAME"
  export JOB_NAME="distilbert_grid_${NAME}"
  export LR="$LR" LORA_R="$R" LORA_ALPHA="$A" EPOCHS="$EPOCHS"
  export WEIGHT_DECAY="$WD"
  export METRICS_DIR
  export LORA_TYPE="${LORA_TYPE:-default}"
  export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
  export BATCH_SIZE="${BATCH_SIZE:-4}"
  export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

  if [[ "${GRID_RESUME}" != "0" ]] && [[ -f "${METRICS_DIR}/test.csv" ]]; then
    lines=$(wc -l < "${METRICS_DIR}/test.csv" | tr -d ' \t')
    data_rows=$((lines - 1))
    if [[ "${data_rows}" -ge "${EPOCHS}" ]]; then
      echo "[grid] skip (complete: ${data_rows} eval rows >= EPOCHS=${EPOCHS}): ${NAME}" >&2
      continue
    fi
  fi

  bash distilbert/scripts/submit_bsub.sh
  sleep 2
done

echo "Done. Submitted grid jobs; results under $RESULTS_ROOT"
