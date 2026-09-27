"""Feature-window construction for the churn panel.

Builds model-ready feature matrices restricted to a chosen span of panel
months, so the honest modelling window can be compared against leaky
alternatives that include the contaminated ``N``/``N-1`` months.
"""

import numpy as np
import pandas as pd

from src.data.loaders import month_labels

ID_COLS: list[str] = ["SND", "PREFIX_ND"]
STATIC: list[str] = ["SEGMEN_ID", "UMUR_PLG", "PAKET_SPEEDY_ID", "WITEL"]
CAT: list[str] = ["SEGMEN_ID", "PAKET_SPEEDY_ID", "WITEL"]
SIGNALS: list[str] = ["TAG", "STATUS_BAYAR", "GGN", "USAGE"]
TARGET_COL: str = "CHURN"

_MONTHS = month_labels()

WINDOW_PRESETS: dict[str, list[str]] = {
    "ALL_12": _MONTHS[:12],
    "DROP_N": _MONTHS[:11],
    "HONEST": _MONTHS[:10],
    "DROP_N_N1_N2": _MONTHS[:9],
    "FAR": _MONTHS[:6],
}


def build_features(df: pd.DataFrame, months: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a one-hot-encoded feature matrix restricted to the given months.

    Args:
        df: The raw churn panel.
        months: Chronologically ordered month labels to include as lag
            features (typically a value from ``WINDOW_PRESETS``).

    Returns:
        Tuple ``(X, y)``: feature matrix (static columns + signal lag columns
        for ``months``, categoricals one-hot encoded, ``ID_COLS`` excluded)
        and the churn target array. Missing values are left as NaN.
    """
    signal_cols = [f"{signal}_{month}" for signal in SIGNALS for month in months]
    feature_cols = [col for col in STATIC if col not in ID_COLS] + signal_cols

    X = df[feature_cols].copy()
    X = pd.get_dummies(X, columns=CAT)
    y = df[TARGET_COL].to_numpy()
    return X, y


def build_features_coded(
    df: pd.DataFrame, months: list[str]
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    """Build a feature matrix with categoricals kept as integer codes.

    Same columns as ``build_features``, but ``CAT`` columns are encoded as
    integer category codes rather than one-hot columns. Required by
    SMOTENC, which needs to know which feature positions are categorical
    so it can interpolate numeric features but pick (not blend) categories.

    Args:
        df: The raw churn panel.
        months: Chronologically ordered month labels to include as lag
            features (typically a value from ``WINDOW_PRESETS``).

    Returns:
        Tuple ``(X, y, cat_idx)``: the coded feature matrix, the churn
        target array, and the column positions of ``CAT`` features in ``X``.
    """
    signal_cols = [f"{signal}_{month}" for signal in SIGNALS for month in months]
    feature_cols = [col for col in STATIC if col not in ID_COLS] + signal_cols

    X = df[feature_cols].copy()
    for col in CAT:
        X[col] = X[col].astype("category").cat.codes
    y = df[TARGET_COL].to_numpy()
    cat_idx = [X.columns.get_loc(col) for col in CAT]
    return X, y, cat_idx
