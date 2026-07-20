"""Load the YAML config and expose small helpers used across the pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


def _to_namespace(obj: Any) -> Any:
    """Recursively turn nested dicts into dot-accessible namespaces."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path: str | os.PathLike = "config/config.yaml") -> SimpleNamespace:
    """Read config.yaml into a dot-accessible namespace.

    Usage:
        cfg = load_config()
        print(cfg.embeddings.batch_size)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at {path.resolve()}. "
            "Run from the repo root or pass an explicit --config path."
        )
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch not needed for the cohort/preprocessing stages
        pass


def ensure_dirs(cfg: SimpleNamespace) -> None:
    """Create output directories if they don't exist."""
    for d in (cfg.paths.artifacts_dir, cfg.paths.models_dir, cfg.paths.results_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
