"""DeepSeek LoRA **细网格**超参（仅用于第二轮搜索，输出到 ``results_refine/``）。

依据仓库内 **第一轮 DeepSeek** ``deepseek_autogrid/results/summary.csv`` +
``deepseek_grid_analysis.md``（约 2026-04 一次完整 90 组）：

- 最优区集中在 **lr ≈ 2e-3 与 2e-4**，Top15 几乎被这两档占满；**2e-5** 明显更差，**2e-6 / 2e-7**
  均值很差，细搜不再展开到低 lr。
- **r=32、64** 的均值与 Top 组合显著优于 r=16；细搜以 **32、64** 为主。
- **alpha** 在 16–64 均有入榜；加 **48** 在 32 与 64 之间插值。
- **weight_decay** 仍保留 0 / 0.01（Top 里两档都出现）。

组合数：**8 × 2 × 4 × 2 = 128**（与 ``config.py`` 相同的训练默认值；仅 LR/R/ALPHA 与 ``RESULTS_ROOT`` 不同）。
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "deepseek_autogrid" / "results_refine"

# 在 2e-3、2e-4 峰值附近加密（第一轮中 2e-5…2e-7 明显更差，细搜不展开）
LR_LIST = [3e-3, 2e-3, 1.5e-3, 1e-3, 6e-4, 4e-4, 3e-4, 2e-4]

R_LIST = [32, 64]

# Top 组合里 16/32/64 均出现；在 32 两侧加 48 插值（共 4 档以控制总 job 数）
ALPHA_LIST = [16, 32, 48, 64]

WEIGHT_DECAY_LIST = [0.0, 0.01]

ADAM_BETA1_FIXED = 0.9
ADAM_BETA2_FIXED = 0.999

MAX_STEPS_DEFAULT = 500
EVAL_EVERY_DEFAULT = 100
SFT_PRESET_DEFAULT = "alpaca_train_1k"
SFT_VAL_RATIO_DEFAULT = 0.2
SFT_FORMAT_DEFAULT = "chat"
MODEL_NAME_DEFAULT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def lr_slug(lr: float) -> str:
    return f"{lr:.4e}".replace(".", "p").replace("+", "")


def wd_slug(wd: float) -> str:
    return f"{wd:.4e}".replace(".", "p").replace("+", "")


def run_dir_name(lr: float, r: int, alpha: float, max_steps: int, wd: float) -> str:
    a = int(alpha) if float(alpha).is_integer() else alpha
    return f"lr_{lr_slug(lr)}_r{r}_a{a}_st{max_steps}_wd_{wd_slug(wd)}"


def iter_grid():
    for lr in LR_LIST:
        for r in R_LIST:
            for alpha in ALPHA_LIST:
                for wd in WEIGHT_DECAY_LIST:
                    yield lr, r, alpha, wd


def grid_size() -> int:
    return len(LR_LIST) * len(R_LIST) * len(ALPHA_LIST) * len(WEIGHT_DECAY_LIST)
