# distilbert_autogrid — grid defaults (keep in sync with run_grid_bsub.sh)
from __future__ import annotations

from pathlib import Path

# Repository root: .../Manifold-Lora
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default output root for per-run metrics_dir (each run is a subfolder)
RESULTS_ROOT = PROJECT_ROOT / "distilbert_autogrid" / "results"

LR_LIST = [3e-3, 3e-4, 3e-5, 3e-6, 3e-7]
R_LIST = [8, 16, 32, 64, 128]
ALPHA_LIST = [8, 16, 32, 64, 128]

# AdamW: only weight decay is swept; betas fixed (PyTorch / HF default)
WEIGHT_DECAY_LIST = [0.0, 0.01, 0.1]
ADAM_BETA1_FIXED = 0.9
ADAM_BETA2_FIXED = 0.999

# Single epoch budget for the full grid (change here or via CLI on run_grid)
EPOCHS_DEFAULT = 3

MODEL_NAME_DEFAULT = "distilbert-base-uncased"


def lr_slug(lr: float) -> str:
    """Filesystem-safe token for learning rate, e.g. 3e-4 -> 3p0000e-04."""
    s = f"{lr:.4e}"
    return s.replace(".", "p").replace("+", "")


def wd_slug(wd: float) -> str:
    s = f"{wd:.4e}"
    return s.replace(".", "p").replace("+", "")


def run_dir_name(
    lr: float,
    r: int,
    alpha: float,
    epochs: int,
    weight_decay: float,
) -> str:
    a = int(alpha) if float(alpha).is_integer() else alpha
    return f"lr_{lr_slug(lr)}_r{r}_a{a}_ep{epochs}_wd_{wd_slug(weight_decay)}"


def iter_grid():
    """Yield (lr, r, alpha, weight_decay) for the full factorial grid."""
    for lr in LR_LIST:
        for r in R_LIST:
            for alpha in ALPHA_LIST:
                for weight_decay in WEIGHT_DECAY_LIST:
                    yield lr, r, alpha, weight_decay


def grid_size() -> int:
    return len(LR_LIST) * len(R_LIST) * len(ALPHA_LIST) * len(WEIGHT_DECAY_LIST)
