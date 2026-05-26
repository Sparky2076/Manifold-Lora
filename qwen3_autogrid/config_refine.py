"""Qwen3-0.6B refine grid: LR × rank × WD; alpha = 2*r; knowledge_mc_mix only."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "qwen3_autogrid" / "results_mmlu_refine"

LR_LIST = [3e-3, 2e-3, 1.5e-3, 1e-3, 6e-4, 4e-4, 3e-4, 2e-4]

R_LIST = [32, 64]

WEIGHT_DECAY_LIST = [0.0, 0.001, 0.01]

ADAM_BETA1_FIXED = 0.9
ADAM_BETA2_FIXED = 0.999

MAX_STEPS_DEFAULT = 500
EVAL_EVERY_DEFAULT = 100
SFT_PRESET_DEFAULT = "knowledge_mc_mix"
SFT_VAL_RATIO_DEFAULT = 0.1
SFT_FORMAT_DEFAULT = "chat"
MODEL_NAME_DEFAULT = "Qwen/Qwen3-0.6B"
LORA_TARGETS_DEFAULT = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
TRUST_REMOTE_CODE_DEFAULT = True


def lora_alpha_for_rank(r: int) -> float:
    return float(2 * r)


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
            alpha = lora_alpha_for_rank(r)
            for wd in WEIGHT_DECAY_LIST:
                yield lr, r, alpha, wd


def grid_size() -> int:
    return len(LR_LIST) * len(R_LIST) * len(WEIGHT_DECAY_LIST)
