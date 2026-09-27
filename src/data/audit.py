"""Pure audit functions for the churn panel: tidy DataFrames/dicts, no plotting."""

import pandas as pd

from src.data.loaders import month_labels, signal_columns

CHURN_COL: str = "CHURN"


def _split_by_class(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the panel into churn and non-churn subsets."""
    churn = df.loc[df[CHURN_COL] == 1]
    nonchurn = df.loc[df[CHURN_COL] == 0]
    return churn, nonchurn


def class_balance(df: pd.DataFrame) -> dict:
    """Summarize class balance of the churn target.

    Args:
        df: The churn panel, must contain ``CHURN_COL``.

    Returns:
        Dict with ``n``, ``positives``, and ``positive_rate``.
    """
    n = len(df)
    positives = int(df[CHURN_COL].sum())
    return {
        "n": n,
        "positives": positives,
        "positive_rate": positives / n if n else float("nan"),
    }


def zero_fraction_by_month(df: pd.DataFrame, signal: str = "USAGE") -> pd.DataFrame:
    """Compute the per-month fraction of zero values, split by churn class.

    Args:
        df: The churn panel.
        signal: One of ``TAG``, ``STATUS_BAYAR``, ``GGN``, ``USAGE``.

    Returns:
        DataFrame with columns ``[month, churn_zero_frac, nonchurn_zero_frac]``.
    """
    churn, nonchurn = _split_by_class(df)
    rows = [
        {
            "month": month,
            "churn_zero_frac": float((churn[col] == 0).mean()),
            "nonchurn_zero_frac": float((nonchurn[col] == 0).mean()),
        }
        for month, col in zip(month_labels(), signal_columns(signal))
    ]
    return pd.DataFrame(rows)


def median_trajectory(df: pd.DataFrame, signal: str = "USAGE") -> pd.DataFrame:
    """Compute the per-month median of a signal, split by churn class.

    Args:
        df: The churn panel.
        signal: One of ``TAG``, ``STATUS_BAYAR``, ``GGN``, ``USAGE``.

    Returns:
        DataFrame with columns ``[month, churn_median, nonchurn_median]``.
    """
    churn, nonchurn = _split_by_class(df)
    rows = [
        {
            "month": month,
            "churn_median": float(churn[col].median()),
            "nonchurn_median": float(nonchurn[col].median()),
        }
        for month, col in zip(month_labels(), signal_columns(signal))
    ]
    return pd.DataFrame(rows)


def mean_by_month(df: pd.DataFrame, signal: str = "GGN") -> pd.DataFrame:
    """Compute the per-month mean of a signal, split by churn class.

    Args:
        df: The churn panel.
        signal: One of ``TAG``, ``STATUS_BAYAR``, ``GGN``, ``USAGE``.

    Returns:
        DataFrame with columns ``[month, churn_mean, nonchurn_mean]``.
    """
    churn, nonchurn = _split_by_class(df)
    rows = [
        {
            "month": month,
            "churn_mean": float(churn[col].mean()),
            "nonchurn_mean": float(nonchurn[col].mean()),
        }
        for month, col in zip(month_labels(), signal_columns(signal))
    ]
    return pd.DataFrame(rows)


def tenure_summary(df: pd.DataFrame, col: str = "UMUR_PLG") -> dict:
    """Compute median tenure, split by churn class.

    Args:
        df: The churn panel.
        col: Tenure column name.

    Returns:
        Dict with ``churn_median`` and ``nonchurn_median``.
    """
    churn, nonchurn = _split_by_class(df)
    return {
        "churn_median": float(churn[col].median()),
        "nonchurn_median": float(nonchurn[col].median()),
    }


def detect_artifact_months(
    df: pd.DataFrame, signal: str = "USAGE", ratio: float = 0.2
) -> list[str]:
    """Flag months whose all-customer median collapses relative to neighbours.

    A month is flagged when its all-customer median is below ``ratio`` times
    the mean of its immediate neighbours' medians, which surfaces data-artifact
    dips (e.g. a billing-cycle or migration effect) rather than genuine trend.

    Args:
        df: The churn panel.
        signal: One of ``TAG``, ``STATUS_BAYAR``, ``GGN``, ``USAGE``.
        ratio: Threshold multiplier applied to the neighbour-median mean.

    Returns:
        Flagged month labels, in chronological order.
    """
    months = month_labels()
    medians = [float(df[col].median()) for col in signal_columns(signal)]

    flagged = []
    for i, month in enumerate(months):
        neighbours = [medians[j] for j in (i - 1, i + 1) if 0 <= j < len(medians)]
        if not neighbours:
            continue
        neighbour_mean = sum(neighbours) / len(neighbours)
        if neighbour_mean > 0 and medians[i] < ratio * neighbour_mean:
            flagged.append(month)
    return flagged
