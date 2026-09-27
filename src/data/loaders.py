"""Loaders and column helpers for the private churn panel dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import RAW_CHURN_PATH

SIGNALS: tuple[str, ...] = ("TAG", "STATUS_BAYAR", "GGN", "USAGE")
N_MONTHS: int = 12


def load_churn_panel(path: Path = RAW_CHURN_PATH) -> pd.DataFrame:
    """Load the raw churn panel from the configured Excel file.

    Args:
        path: Path to the raw Excel file.

    Returns:
        The raw churn panel as a DataFrame.

    Raises:
        FileNotFoundError: If no file exists at ``path``. The private dataset
            is never fabricated or substituted.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw churn panel not found at '{path}'. This dataset is private "
            "and not committed to the repository; place the file there before "
            "running this notebook."
        )
    return pd.read_excel(path)


def month_labels() -> list[str]:
    """Return the 12 panel month labels in chronological order.

    Returns:
        Labels ``["N-11", "N-10", ..., "N-1", "N"]``.
    """
    return [f"N-{offset}" if offset else "N" for offset in range(N_MONTHS - 1, -1, -1)]


def signal_columns(signal: str) -> list[str]:
    """Return the ordered lag columns for a signal, oldest to newest.

    Args:
        signal: One of ``TAG``, ``STATUS_BAYAR``, ``GGN``, ``USAGE``.

    Returns:
        Column names such as ``["USAGE_N-11", ..., "USAGE_N-1", "USAGE_N"]``.

    Raises:
        ValueError: If ``signal`` is not a recognized panel signal.
    """
    if signal not in SIGNALS:
        raise ValueError(f"Unknown signal '{signal}'. Expected one of {SIGNALS}.")
    return [f"{signal}_{month}" for month in month_labels()]


def signal_matrix(df: pd.DataFrame, signal: str) -> np.ndarray:
    """Extract a signal's 12-month values as a chronologically ordered matrix.

    Args:
        df: The churn panel.
        signal: One of ``TAG``, ``STATUS_BAYAR``, ``GGN``, ``USAGE``.

    Returns:
        Array of shape ``(n_rows, 12)`` ordered chronologically (``N-11`` to ``N``).
    """
    return df[signal_columns(signal)].to_numpy()
