"""Classifier two-sample tests for real vs. synthetic minority examples."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split

from src.data.loaders import SIGNALS
from src.models.factory import make_gbm
from src.utils.config import SEED

LOG1P_SIGNALS: tuple[str, ...] = ("TAG", "USAGE")


def _panel_feature_frame(
    df: pd.DataFrame,
    months: list[str],
    signals: tuple[str, ...] = SIGNALS,
) -> pd.DataFrame:
    """Build the flattened panel-lag representation used by C2ST.

    The feature space contains only panel signal lags, not static customer
    fields. ``TAG`` and ``USAGE`` are transformed with ``log1p`` to match the
    prior preprocessing choice for sparse/high-range activity signals; the
    bounded/status-like ``STATUS_BAYAR`` and ``GGN`` signals are left on their
    original scale.

    Args:
        df: Raw churn panel.
        months: Chronologically ordered month labels to include.
        signals: Signal names to include.

    Returns:
        DataFrame with columns named ``{signal}_{month}``.
    """
    columns = [f"{signal}_{month}" for signal in signals for month in months]
    X = df[columns].copy()
    for signal in signals:
        if signal in LOG1P_SIGNALS:
            signal_cols = [f"{signal}_{month}" for month in months]
            X[signal_cols] = np.log1p(X[signal_cols])
    return X


def _drop_rows_with_missing(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Drop rows with missing C2ST features before distance-based SMOTE."""
    missing_rows = X.isna().any(axis=1).to_numpy()
    return X.loc[~missing_rows].reset_index(drop=True), y[~missing_rows], int(missing_rows.sum())


def _as_feature_frame(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """Return a numeric feature DataFrame with stable column names."""
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True).copy()
    array = np.asarray(X)
    columns = [f"feature_{idx}" for idx in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def build_real_vs_synthetic_xy(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Build a balanced real-minority vs SMOTE-synthetic C2ST dataset.

    Args:
        X: Feature matrix for a binary classification dataset.
        y: Binary target array; the minority/positive class must be labeled
            ``1``.
        seed: Random seed for SMOTE and synthetic-row subsampling.

    Returns:
        Tuple ``(X_c2st, y_c2st, meta)`` where real minority rows are labeled
        ``0`` and newly synthesized SMOTE minority rows are labeled ``1``.
        The task is balanced by subsampling synthetic rows down to the real
        minority count.
    """
    X_all = _as_feature_frame(X)
    y_array = np.asarray(y)
    X_all, y_array, n_dropped_missing = _drop_rows_with_missing(X_all, y_array)

    real = X_all.loc[y_array == 1].reset_index(drop=True)
    if len(real) < 2:
        raise ValueError("C2ST requires at least two real minority examples.")

    k_neighbors = min(5, len(real) - 1)
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X_all, y_array)

    n_original = len(X_all)
    if not np.allclose(np.asarray(X_resampled.iloc[:n_original]), np.asarray(X_all)):
        raise AssertionError("SMOTE did not preserve original rows as the resampled prefix.")
    if not np.array_equal(np.asarray(y_resampled[:n_original]), y_array):
        raise AssertionError("SMOTE did not preserve original labels as the resampled prefix.")

    synthetic_all = X_resampled.iloc[n_original:].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    synthetic_idx = rng.choice(len(synthetic_all), size=len(real), replace=False)
    synthetic = synthetic_all.iloc[synthetic_idx].reset_index(drop=True)

    X_c2st = pd.concat([real, synthetic], ignore_index=True)
    y_c2st = np.r_[np.zeros(len(real), dtype=int), np.ones(len(synthetic), dtype=int)]
    groups = np.arange(len(X_c2st))
    meta = {
        "task": "real_vs_smote",
        "n_real": int(len(real)),
        "n_smote_synthetic_total": int(len(synthetic_all)),
        "n_smote_synthetic_used": int(len(synthetic)),
        "n_features": int(X_c2st.shape[1]),
        "n_dropped_missing": n_dropped_missing,
        "positive_rate": float(np.mean(y_c2st)),
        "smote_k_neighbors": int(k_neighbors),
    }
    X_c2st.attrs.update({**meta, "_groups": groups})
    return X_c2st, y_c2st, meta


def build_real_vs_duplicated_xy(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Build a balanced real-minority vs exact-copy C2ST sanity task.

    Args:
        X: Feature matrix for a binary classification dataset.
        y: Binary target array; the minority/positive class must be labeled
            ``1``.
        seed: Random seed, stored for reproducibility metadata.

    Returns:
        Tuple ``(X_c2st, y_c2st, meta)`` where real minority rows are labeled
        ``0`` and exact copies are labeled ``1``. Group IDs keep each
        original/copy pair in the same CV fold, making the expected AUC 0.5.
    """
    X_all = _as_feature_frame(X)
    y_array = np.asarray(y)
    X_all, y_array, n_dropped_missing = _drop_rows_with_missing(X_all, y_array)

    real = X_all.loc[y_array == 1].reset_index(drop=True)
    duplicated = real.copy()

    X_c2st = pd.concat([real, duplicated], ignore_index=True)
    y_c2st = np.r_[np.zeros(len(real), dtype=int), np.ones(len(duplicated), dtype=int)]
    groups = np.r_[np.arange(len(real)), np.arange(len(duplicated))]
    meta = {
        "task": "real_vs_duplicated",
        "n_real": int(len(real)),
        "n_duplicated": int(len(duplicated)),
        "n_features": int(X_c2st.shape[1]),
        "n_dropped_missing": n_dropped_missing,
        "positive_rate": float(np.mean(y_c2st)),
        "duplicate_policy": "one_exact_copy_per_real_row",
        "seed": int(seed),
    }
    X_c2st.attrs.update({**meta, "_groups": groups})
    return X_c2st, y_c2st, meta


def build_real_vs_synthetic(
    df: pd.DataFrame,
    months: list[str],
    signals: tuple[str, ...] = SIGNALS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a balanced real-minority vs SMOTE-synthetic C2ST dataset.

    The returned task is balanced by subsampling the newly synthesized SMOTE
    rows down to the number of real minority rows. Full pre-subsampling group
    sizes and preprocessing details are stored in ``X.attrs`` for reporting.

    Args:
        df: Raw churn panel with ``CHURN`` target and panel signal columns.
        months: Chronologically ordered month labels to include.
        signals: Signal names to include.
        seed: Random seed for SMOTE and synthetic-row subsampling.

    Returns:
        Tuple ``(X, y_c2st)`` where real minority rows are labeled ``0`` and
        SMOTE-synthetic rows are labeled ``1``.
    """
    X_all = _panel_feature_frame(df, months, signals)
    X_c2st, y_c2st, meta = build_real_vs_synthetic_xy(X_all, df["CHURN"].to_numpy(), seed=seed)
    X_c2st.attrs.update(
        {
            **meta,
            "task": "real_vs_smote",
            "months": list(months),
            "signals": list(signals),
            "log1p_signals": [signal for signal in signals if signal in LOG1P_SIGNALS],
        }
    )
    return X_c2st, y_c2st


def build_real_vs_duplicated(
    df: pd.DataFrame,
    months: list[str],
    signals: tuple[str, ...] = SIGNALS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a balanced real-minority vs duplicated-minority C2ST dataset.

    Duplicated rows are one exact copy of every real minority row. This is
    the sanity floor: the classifier should not be able to distinguish a row
    from an identical copy of that same row. Sampling duplicates with
    replacement would introduce finite-sample multiplicity artifacts, which
    are not the mechanism being tested here.

    Args:
        df: Raw churn panel with ``CHURN`` target and panel signal columns.
        months: Chronologically ordered month labels to include.
        signals: Signal names to include.
        seed: Random seed, stored for reproducibility metadata.

    Returns:
        Tuple ``(X, y_c2st)`` where real minority rows are labeled ``0`` and
        duplicated rows are labeled ``1``.
    """
    X_all = _panel_feature_frame(df, months, signals)
    X_c2st, y_c2st, meta = build_real_vs_duplicated_xy(X_all, df["CHURN"].to_numpy(), seed=seed)
    X_c2st.attrs.update(
        {
            **meta,
            "task": "real_vs_duplicated",
            "months": list(months),
            "signals": list(signals),
            "log1p_signals": [signal for signal in signals if signal in LOG1P_SIGNALS],
        }
    )
    return X_c2st, y_c2st


def _repeated_splits(
    X: pd.DataFrame,
    y_c2st: np.ndarray,
    n_splits: int,
    n_repeats: int,
    seed: int,
):
    """Yield repeated stratified splits, grouped when C2ST pairs are present."""
    groups = X.attrs.get("_groups")
    if groups is None:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=seed
        )
        yield from splitter.split(X, y_c2st)
        return

    groups = np.asarray(groups)
    for repeat in range(n_repeats):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        yield from splitter.split(X, y_c2st, groups=groups)


def c2st_auc(
    X: pd.DataFrame,
    y_c2st: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = SEED,
) -> dict:
    """Estimate C2ST ROC-AUC with repeated stratified cross-validation.

    Args:
        X: C2ST feature matrix.
        y_c2st: Binary C2ST labels, with synthetic/copy rows labeled ``1``.
        n_splits: Number of folds per repeat.
        n_repeats: Number of repeated CV rounds.
        seed: Random seed for the splitter and model.

    Returns:
        Dict with per-fold AUCs, mean, standard deviation, standard error,
        and normal-approximation 95% interval for the mean.
    """
    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(
        _repeated_splits(X, y_c2st, n_splits, n_repeats, seed)
    ):
        model = make_gbm(random_state=seed)
        model.fit(X.iloc[train_idx], y_c2st[train_idx])
        scores = model.predict_proba(X.iloc[test_idx])[:, 1]
        rows.append(
            {
                "repeat": int(fold_i // n_splits),
                "fold": int(fold_i % n_splits),
                "roc_auc": float(roc_auc_score(y_c2st[test_idx], scores)),
            }
        )

    fold_results = pd.DataFrame(rows)
    aucs = fold_results["roc_auc"].to_numpy()
    std = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")
    se = float(std / np.sqrt(len(aucs))) if len(aucs) > 1 else float("nan")
    mean = float(np.mean(aucs))
    return {
        "fold_results": fold_results,
        "aucs": aucs,
        "mean": mean,
        "std": std,
        "se": se,
        "ci_low": float(mean - 1.96 * se),
        "ci_high": float(mean + 1.96 * se),
        "n_folds": int(len(aucs)),
    }


def c2st_importances(
    X: pd.DataFrame,
    y_c2st: np.ndarray,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compute held-out permutation importances for the C2ST classifier.

    Args:
        X: C2ST feature matrix.
        y_c2st: Binary C2ST labels, with synthetic rows labeled ``1``.
        seed: Random seed for the train/test split, model, and permutations.

    Returns:
        DataFrame sorted by decreasing mean permutation importance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_c2st, test_size=0.25, stratify=y_c2st, random_state=seed
    )
    model = make_gbm(random_state=seed)
    model.fit(X_train, y_train)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=seed,
        n_jobs=1,
    )
    importances = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importances.sort_values("importance_mean", ascending=False).reset_index(drop=True)
