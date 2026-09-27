"""Loader for the public Taiwan bankruptcy dataset.

This is the fully reproducible, public counterpart to the private churn
panel: unlike the churn panel, it is cross-sectional (one row per company,
no repeated time-lag structure), which is exactly what makes it useful for
testing whether the churn study's findings are specific to temporal panel
data or general properties of the imbalance-handling strategies themselves.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils.config import PROJECT_ROOT

TAIWAN_URL: str = (
    "https://raw.githubusercontent.com/SayamAlt/Company-Bankruptcy-Prediction/main/data.csv"
)
DEFAULT_TAIWAN_PATH: Path = PROJECT_ROOT / "data" / "raw" / "taiwan_bankruptcy.csv"
TARGET_COL: str = "Bankrupt?"


def download_taiwan(dest: Path = DEFAULT_TAIWAN_PATH) -> Path:
    """Download the public Taiwan bankruptcy dataset if not already present.

    Args:
        dest: Destination path for the downloaded CSV.

    Returns:
        The destination path.
    """
    dest = Path(dest)
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(TAIWAN_URL, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_taiwan(path: Path = DEFAULT_TAIWAN_PATH) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the Taiwan bankruptcy dataset.

    Cross-sectional (one row per company), fully numeric, no missing values
    - unlike the churn panel, no per-fold imputation or categorical
    encoding is needed.

    Args:
        path: Path to the downloaded CSV.

    Returns:
        Tuple ``(X, y)``: all columns except the target, and the bankruptcy
        target as a NumPy array.

    Raises:
        FileNotFoundError: If no file exists at ``path``. The dataset is
            never fabricated; call ``download_taiwan()`` first.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Taiwan bankruptcy dataset not found at '{path}'. Call download_taiwan() first."
        )
    df = pd.read_csv(path)
    y = df[TARGET_COL].to_numpy()
    X = df.drop(columns=[TARGET_COL])
    return X, y
