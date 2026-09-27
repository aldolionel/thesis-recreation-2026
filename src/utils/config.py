"""Project-wide configuration constants and reproducibility helpers."""

import random
from pathlib import Path

import numpy as np

SEED: int = 42

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

RAW_CHURN_PATH: Path = PROJECT_ROOT / "data" / "raw" / "training_clean.xlsx"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int = SEED) -> None:
    """Seed numpy and the stdlib random module for reproducibility.

    Args:
        seed: Seed value applied to both random number generators.
    """
    np.random.seed(seed)
    random.seed(seed)
