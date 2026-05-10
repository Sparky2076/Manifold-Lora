"""Resolve which DeepSeek autogrid config module to use (coarse vs refine)."""
from __future__ import annotations

import importlib
import os
from types import ModuleType

DEFAULT_CONFIG_MODULE = "deepseek_autogrid.config"


def resolve_config_module(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return os.environ.get("DEEPSEEK_GRID_CONFIG_MODULE", DEFAULT_CONFIG_MODULE)


def load_grid_config(explicit: str | None = None) -> ModuleType:
    return importlib.import_module(resolve_config_module(explicit))
