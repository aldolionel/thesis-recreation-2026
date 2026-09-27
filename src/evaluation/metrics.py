"""Honest evaluation metrics. Pure functions on ``(y_true, y_score)``."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DEFAULT_BUDGET: float = 0.10


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the ROC-AUC.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.

    Returns:
        ROC-AUC.
    """
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the PR-AUC (average precision).

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.

    Returns:
        Average precision.
    """
    return float(average_precision_score(y_true, y_score))


def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute the Brier score (calibration-sensitive).

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted probabilities.

    Returns:
        Brier score.
    """
    return float(brier_score_loss(y_true, y_score))


def recall_at_budget(y_true: np.ndarray, y_score: np.ndarray, k: float = DEFAULT_BUDGET) -> float:
    """Compute recall among the top-``k`` fraction of highest-scored instances.

    Simulates a fixed intervention budget: rank all instances by score and
    check what fraction of actual positives fall in the top ``k`` share.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.
        k: Fraction of the population selected, in (0, 1].

    Returns:
        Recall among the selected top-``k`` fraction, or NaN if there are no
        positives in ``y_true``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    total_positives = y_true.sum()
    if total_positives == 0:
        return float("nan")

    budget = max(1, int(np.ceil(k * len(y_true))))
    order = np.argsort(-y_score, kind="stable")
    top_idx = order[:budget]
    return float(y_true[top_idx].sum() / total_positives)


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute the full honest-evaluation metric suite.

    Args:
        y_true: Binary ground-truth labels.
        y_score: Predicted scores or probabilities.

    Returns:
        Dict with ``roc_auc``, ``pr_auc``, ``brier``, and ``recall_at_10pct``.
    """
    return {
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "brier": brier(y_true, y_score),
        "recall_at_10pct": recall_at_budget(y_true, y_score, k=DEFAULT_BUDGET),
    }
