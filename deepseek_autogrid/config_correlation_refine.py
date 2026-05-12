"""Auto-generated refine grid from ``suggest_refine_from_summary.py`` (correlation-driven)."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "deepseek_autogrid" / "results_correlation_refine"

LR_LIST = [0.0002, 0.0002465693, 0.0003039822, 0.0003747635, 0.0004620259, 0.0005696072, 0.0007022383, 0.0008657523, 0.0010673398, 0.0013158664, 0.0016222617, 0.002]

R_LIST = [32, 64]

ALPHA_LIST = [16.0, 32.0]

WEIGHT_DECAY_LIST = [0.01]

ADAM_BETA1_FIXED = 0.9
ADAM_BETA2_FIXED = 0.999

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
