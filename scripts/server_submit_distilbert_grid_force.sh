#!/usr/bin/env bash
# =============================================================================
# DistilBERT 全因子网格 — 强制全部重交（GRID_RESUME=0，不跳过已有结果）
# =============================================================================
# 与 server_submit_distilbert_grid.sh 相同，但固定 GRID_RESUME=0，适合覆盖重跑。
#
# 【推荐】在 tmux 里跑（SSH 断线后仍会继续 bsub）：
#   tmux new -s grid
#   cd ~/Manifold-Lora
#   export CONDA_ROOT="$HOME/miniconda3"    # 改成你的 conda 根（含 etc/profile.d/conda.sh）
#   bash scripts/server_submit_distilbert_grid_force.sh
#   # detach：Ctrl+B 松开后再按 D
#   # 重连：tmux attach -t grid
#
# 一行启动 tmux 并执行（路径按你改）：
#   tmux new -s grid "bash -lc 'cd ~/Manifold-Lora && export CONDA_ROOT=\"\$HOME/miniconda3\" && bash scripts/server_submit_distilbert_grid_force.sh'"
#
# 可选：CONDA_ENV_NAME、EPOCHS、QUEUE、RESULTS_ROOT、LORA_* 等，在运行前 export 即可。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export GRID_RESUME=0
export CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"

echo "==> GRID_RESUME=0（将重新提交所有组合，不跳过已有 test.csv）"
echo "==> CONDA_ROOT=$CONDA_ROOT"

exec bash "$SCRIPT_DIR/server_submit_distilbert_grid.sh"
