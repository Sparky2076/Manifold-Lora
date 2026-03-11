#!/usr/bin/env bash
# 在服务器上运行：提交 LSF 训练任务（单卡）
# 用法: cd ~/Manifold-Lora && bash scripts/submit_bsub.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

JOB_NAME="${JOB_NAME:-manifold_lora}"
QUEUE="${QUEUE:-gpu}"
NGPU=1
MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
METRICS_DIR="${METRICS_DIR:-$PROJECT_DIR}"

# 把网格搜索相关环境变量传入计算节点（LSF 不一定继承当前 shell 的 env）
export EPOCHS="${EPOCHS:-}"
export BATCH_SIZE="${BATCH_SIZE:-}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-}"
export LR="${LR:-}"
export LORA_TYPE="${LORA_TYPE:-}"
export LORA_R="${LORA_R:-}"
export LORA_ALPHA="${LORA_ALPHA:-}"
export LORA_DROPOUT="${LORA_DROPOUT:-}"

RUN_CMD="bash $SCRIPT_DIR/run_train_bsub.sh '$MODEL_NAME' '$METRICS_DIR'"
# 若当前 shell 设置了上述变量，在计算节点上先 export 再跑
for v in EPOCHS BATCH_SIZE GRAD_ACCUM_STEPS LR LORA_TYPE LORA_R LORA_ALPHA LORA_DROPOUT; do
  eval "val=\${$v}"
  if [[ -n "${val:-}" ]]; then
    RUN_CMD="export $v='$val'; $RUN_CMD"
  fi
done

bsub -J "$JOB_NAME" \
  -o "%J.out" \
  -e "%J.err" \
  -q "$QUEUE" \
  -n 1 \
  -R "rusage[mem=32G]" \
  -gpu "num=${NGPU}" \
  "$RUN_CMD"

echo "已提交任务，查看: bjobs"
