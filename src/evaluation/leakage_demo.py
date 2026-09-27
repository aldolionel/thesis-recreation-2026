"""Minimal demonstration of SMOTE data leakage: resample-before-split vs
resample-inside-fold, on a cross-sectional dataset.
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import pr_auc
from src.models.factory import make_gbm


def leaky_vs_honest_smote(
    X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> dict:
    """Compare SMOTE-before-split (leaky) vs SMOTE-inside-fold (honest) CV.

    The wrong protocol resamples the entire dataset once with SMOTE, then
    cross-validates on the resampled pool: synthetic points derived from
    what would otherwise be test-fold neighbours leak into training folds,
    and the "test" fold itself is partly synthetic. The honest protocol
    fits SMOTE on each training fold only and always scores an untouched,
    real-only test fold.

    Args:
        X: Feature matrix.
        y: Binary target array.
        n_splits: Number of stratified CV folds.
        seed: Random seed for SMOTE and the CV splitter.

    Returns:
        Dict with ``leaky_pr_auc``, ``honest_pr_auc``, and ``inflation``
        (``leaky_pr_auc - honest_pr_auc``).
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X_res, y_res = SMOTE(random_state=seed).fit_resample(X, y)
    leaky_scores = []
    for train_idx, test_idx in splitter.split(X_res, y_res):
        model = make_gbm()
        model.fit(X_res.iloc[train_idx], y_res[train_idx])
        scores = model.predict_proba(X_res.iloc[test_idx])[:, 1]
        leaky_scores.append(pr_auc(y_res[test_idx], scores))
    leaky_pr_auc = float(np.mean(leaky_scores))

    honest_scores = []
    for train_idx, test_idx in splitter.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        X_train_res, y_train_res = SMOTE(random_state=seed).fit_resample(X_train, y_train)

        model = make_gbm()
        model.fit(X_train_res, y_train_res)
        scores = model.predict_proba(X_test)[:, 1]
        honest_scores.append(pr_auc(y_test, scores))
    honest_pr_auc = float(np.mean(honest_scores))

    return {
        "leaky_pr_auc": leaky_pr_auc,
        "honest_pr_auc": honest_pr_auc,
        "inflation": leaky_pr_auc - honest_pr_auc,
    }
