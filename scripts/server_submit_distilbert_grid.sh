#!/usr/bin/env bash
# =============================================================================
# Manifold-Lora — 服务器端：完整 DistilBERT 全因子网格提交（LSF / bsub）
# =============================================================================
# 在登录节点、仓库根目录下执行（或任意目录 + 设置 PROJECT_DIR）：
#   bash scripts/server_submit_distilbert_grid.sh
#
# 建议用 tmux（断 SSH 不杀提交循环）：
#   tmux new -s grid
#   cd ~/Manifold-Lora
#   export CONDA_ROOT="$HOME/miniconda3"    # 按实际 conda 根目录修改
#   bash scripts/server_submit_distilbert_grid.sh
#   # 离开：Ctrl+B 再按 D     回来：tmux attach -t grid
#
# 可选环境变量（传给 distilbert_autogrid/run_grid_bsub.sh / submit_bsub）：
#   CONDA_ROOT / CONDA_BASE / CONDA_ENV_NAME   计算节点激活 conda（批作业必设 CONDA_ROOT）
#   EPOCHS  QUEUE  RESULTS_ROOT  GRID_RESUME   LORA_TYPE  LORA_DROPOUT
#   BATCH_SIZE  GRAD_ACCUM_STEPS
#   PROJECT_DIR   仓库根（默认：本脚本所在目录的上一级）
#
# 网格定义：distilbert_autogrid/config.py
# 汇总：python -m distilbert_autogrid.aggregate_results
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_DIR"

echo "==> 仓库根: $PROJECT_DIR"

echo "==> [1/3] sed：脚本换行 CRLF -> LF"
sed -i 's/\r$//' scripts/*.sh distilbert/scripts/*.sh distilbert_autogrid/*.sh 2>/dev/null || true

if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  for _try in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
    if [[ -f "$_try/etc/profile.d/conda.sh" ]]; then
      export CONDA_ROOT="$_try"
      echo "==> 未设置 CONDA_ROOT，使用: $CONDA_ROOT"
      break
    fi
  done
fi

if [[ -n "${CONDA_ROOT:-}" && ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  echo "错误: CONDA_ROOT=${CONDA_ROOT} 下没有 etc/profile.d/conda.sh" >&2
  exit 1
fi
if [[ -n "${CONDA_BASE:-}" && ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "错误: CONDA_BASE=${CONDA_BASE} 下没有 etc/profile.d/conda.sh" >&2
  exit 1
fi
if [[ -z "${CONDA_ROOT:-}" && -z "${CONDA_BASE:-}" ]]; then
  echo "警告: 未检测到 CONDA_ROOT。为免计算节点激活失败，建议: export CONDA_ROOT=\$HOME/miniconda3" >&2
fi

echo "==> [2/3] 环境（节选）: CONDA_ROOT=${CONDA_ROOT:-} CONDA_ENV_NAME=${CONDA_ENV_NAME:-torch} EPOCHS=${EPOCHS:-默认config} GRID_RESUME=${GRID_RESUME:-1}"

echo "==> [3/3] 调用 distilbert_autogrid/run_grid_bsub.sh ..."
exec bash "$PROJECT_DIR/distilbert_autogrid/run_grid_bsub.sh"
