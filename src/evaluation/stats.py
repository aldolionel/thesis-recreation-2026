"""Paired statistical comparison between per-fold metric series."""

import numpy as np
from scipy import stats


def paired_compare(per_fold_a: np.ndarray, per_fold_b: np.ndarray) -> dict:
    """Paired comparison of two per-fold metric series (e.g. strategy vs ``none``).

    Args:
        per_fold_a: Per-fold metric values for the strategy being tested.
        per_fold_b: Per-fold metric values for the baseline, in the same
            fold order as ``per_fold_a``.

    Returns:
        Dict with ``n``, ``mean_delta`` (``a - b``), ``wilcoxon_p``, and
        ``ttest_p``. The p-values are NaN when there are too few paired
        observations (``n < 2``) or the deltas are degenerate (all zero).

    Raises:
        ValueError: If the two series have different lengths.
    """
    a = np.asarray(per_fold_a, dtype=float)
    b = np.asarray(per_fold_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Paired series must have equal length, got {len(a)} vs {len(b)}.")

    delta = a - b
    result = {"n": int(len(delta)), "mean_delta": float(delta.mean())}

    if len(delta) < 2 or np.allclose(delta, 0.0):
        result["wilcoxon_p"] = float("nan")
        result["ttest_p"] = float("nan")
        return result

    try:
        _, wilcoxon_p = stats.wilcoxon(a, b)
    except ValueError:
        wilcoxon_p = float("nan")
    _, ttest_p = stats.ttest_rel(a, b)

    result["wilcoxon_p"] = float(wilcoxon_p)
    result["ttest_p"] = float(ttest_p)
    return result
