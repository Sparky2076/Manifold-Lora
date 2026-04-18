from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "deepseek_autogrid" / "results"

# 粗略对数学习率（3 档，缩短总 job 数；需要更细再改回 5 档）
LR_LIST = [2e-4, 2e-5, 2e-6]
# r / alpha：各 3 档，覆盖中小容量
R_LIST = [16, 32, 64]
ALPHA_LIST = [16, 32, 64]
# 正则：2 档（去掉 0.1 以减组合数；需要再加回）
WEIGHT_DECAY_LIST = [0.0, 0.01]

ADAM_BETA1_FIXED = 0.9
ADAM_BETA2_FIXED = 0.999

# 单 job 时长：步数下调（约原 1500 的 1/3）；可用环境变量 MAX_STEPS 覆盖
MAX_STEPS_DEFAULT = 500
EVAL_EVERY_DEFAULT = 100
SFT_PRESET_DEFAULT = "alpaca_train_1k"
SFT_VAL_RATIO_DEFAULT = 0.2
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
