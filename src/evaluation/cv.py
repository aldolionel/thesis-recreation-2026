"""Leakage-safe repeated cross-validation for imbalance-strategy comparison.

Resampling (when a strategy uses one) is fit and applied on the training
fold only; the test fold is always scored untouched. No feature scaling is
performed anywhere in this module: the model is a histogram gradient
boosting classifier, which splits on raw feature values and is therefore
scale-invariant, matching the baseline established in the leakage-window
ablation.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from src.evaluation.metrics import evaluate
from src.evaluation.strategies import STRATEGIES
from src.features.windows import CAT, build_features, build_features_coded
from src.models.factory import make_gbm

OPERATING_THRESHOLD: float = 0.5


def _encode_categoricals_like_train(
    train_codes: pd.DataFrame, *frames_to_transform: pd.DataFrame
) -> list[np.ndarray]:
    """One-hot encode categorical-code frames using a fold-train-fit encoder.

    Args:
        train_codes: The fold's pre-resample training categorical codes
            (columns = ``CAT``); defines the encoder's category domain only.
        *frames_to_transform: Categorical-code frames to one-hot encode with
            that same fitted encoder (e.g. the resampled train set, the
            untouched test set).

    Returns:
        One dense one-hot array per frame in ``frames_to_transform``.
    """
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_codes)
    return [encoder.transform(frame) for frame in frames_to_transform]


def _impute_numeric_like_train(
    train_frame: pd.DataFrame, *other_frames: pd.DataFrame, columns: list[str] | None = None
) -> list[pd.DataFrame]:
    """Median-impute columns using statistics fit on the training fold only.

    Distance-based resamplers (SMOTE/ADASYN/SMOTENC) cannot accept NaN; the
    raw panel has a single stray missing value, so a fold-safe median
    imputation (fit on train, applied to train and test) is the minimal fix.

    Args:
        train_frame: The fold's training data; imputed and returned first.
        *other_frames: Additional frames (e.g. the test fold) transformed
            with the same fitted imputer.
        columns: Columns to impute; defaults to all of ``train_frame``'s
            columns.

    Returns:
        List of imputed frames: ``[imputed train_frame, *imputed other_frames]``.
    """
    cols = columns if columns is not None else list(train_frame.columns)
    imputer = SimpleImputer(strategy="median")
    result = [train_frame.copy()]
    result[0][cols] = imputer.fit_transform(train_frame[cols])
    for frame in other_frames:
        imputed = frame.copy()
        imputed[cols] = imputer.transform(frame[cols])
        result.append(imputed)
    return result


def _prepare_onehot_fold(
    X_onehot: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, spec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice a one-hot-representation fold and resample its training rows only."""
    X_train_raw = X_onehot.iloc[train_idx]
    X_test_raw = X_onehot.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if spec.requires_finite_input:
        X_train_raw, X_test_raw = _impute_numeric_like_train(X_train_raw, X_test_raw)
    X_test = X_test_raw.to_numpy(dtype=float)

    if spec.build_resampler is None:
        return X_train_raw.to_numpy(dtype=float), y_train, X_test, y_test

    resampler = spec.build_resampler()
    X_train_res, y_train_res = resampler.fit_resample(X_train_raw, y_train)
    X_train = np.asarray(X_train_res, dtype=float)
    return X_train, np.asarray(y_train_res), X_test, y_test


def _prepare_coded_fold(
    X_coded: pd.DataFrame,
    y: np.ndarray,
    cat_idx: list[int],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slice a coded-representation fold, resample (SMOTENC) train rows only,
    then one-hot encode categoricals with an encoder fit on the fold's
    original training categories.
    """
    numeric_cols = [col for col in X_coded.columns if col not in cat_cols]

    X_train_raw = X_coded.iloc[train_idx]
    X_test_raw = X_coded.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if spec.requires_finite_input:
        X_train_raw, X_test_raw = _impute_numeric_like_train(
            X_train_raw, X_test_raw, columns=numeric_cols
        )

    resampler = spec.build_resampler(cat_idx=cat_idx)
    X_res_raw, y_train_res = resampler.fit_resample(X_train_raw, y_train)

    cat_train_oh, cat_test_oh = _encode_categoricals_like_train(
        X_train_raw[cat_cols], X_res_raw[cat_cols], X_test_raw[cat_cols]
    )
    X_train = np.hstack([X_res_raw[numeric_cols].to_numpy(dtype=float), cat_train_oh])
    X_test = np.hstack([X_test_raw[numeric_cols].to_numpy(dtype=float), cat_test_oh])
    return X_train, np.asarray(y_train_res), X_test, y_test


def _repeated_cv_from_arrays(
    X_onehot: pd.DataFrame,
    y: np.ndarray,
    strategy: str,
    n_splits: int,
    n_repeats: int,
    seed: int,
    X_coded: pd.DataFrame | None = None,
    cat_idx: list[int] | None = None,
    cat_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Dataset-agnostic leakage-safe repeated CV core, given pre-built features.

    Shared by ``repeated_cv`` (churn panel, builds features from
    ``(df, months)``) and ``repeated_cv_xy`` (any pre-built ``(X, y)``, e.g.
    the Taiwan bankruptcy dataset). Resampling is fit and applied on the
    training fold only; the test fold is always scored untouched.

    Args:
        X_onehot: One-hot-representation feature matrix (used directly by
            every strategy except those needing ``"coded"``).
        y: Binary target array aligned with ``X_onehot``.
        strategy: Key into ``src.evaluation.strategies.STRATEGIES``.
        n_splits: Number of CV folds per repeat.
        n_repeats: Number of times the k-fold split is repeated.
        seed: Random seed for the CV splitter.
        X_coded: Coded-representation feature matrix, required only for
            strategies with ``representation == "coded"`` (e.g. SMOTENC).
        cat_idx: Column positions of the categorical features in
            ``X_coded``, required alongside ``X_coded``.
        cat_cols: Column names of the categorical features in ``X_coded``,
            required alongside ``X_coded``.

    Returns:
        DataFrame with one row per fold: ``repeat``, ``fold``, ranking
        metrics (``roc_auc``, ``pr_auc``, ``recall_at_10pct``, ``brier``),
        and operating-point metrics at threshold 0.5 (``precision_at_0.5``,
        ``recall_at_0.5``, ``f1_at_0.5``).
    """
    spec = STRATEGIES[strategy]
    if spec.representation == "coded" and (X_coded is None or cat_idx is None or cat_cols is None):
        raise ValueError(
            f"Strategy '{strategy}' needs a coded representation (X_coded, cat_idx, cat_cols)."
        )

    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(X_onehot, y)):
        if spec.representation == "coded":
            X_train, y_train, X_test, y_test = _prepare_coded_fold(
                X_coded, y, cat_idx, train_idx, test_idx, spec, cat_cols
            )
        else:
            X_train, y_train, X_test, y_test = _prepare_onehot_fold(
                X_onehot, y, train_idx, test_idx, spec
            )

        model_kwargs = {"class_weight": spec.class_weight} if spec.class_weight else {}
        model = make_gbm(**model_kwargs)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]

        ranking = evaluate(y_test, scores)
        preds = (scores >= OPERATING_THRESHOLD).astype(int)
        rows.append(
            {
                "repeat": fold_i // n_splits,
                "fold": fold_i % n_splits,
                **ranking,
                "precision_at_0.5": float(precision_score(y_test, preds, zero_division=0)),
                "recall_at_0.5": float(recall_score(y_test, preds, zero_division=0)),
                "f1_at_0.5": float(f1_score(y_test, preds, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def repeated_cv(
    df: pd.DataFrame,
    months: list[str],
    strategy: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Run leakage-safe repeated stratified CV for one imbalance strategy.

    Args:
        df: The raw churn panel.
        months: Chronologically ordered month labels defining the feature
            window (e.g. ``WINDOW_PRESETS["HONEST"]``).
        strategy: Key into ``src.evaluation.strategies.STRATEGIES``.
        n_splits: Number of CV folds per repeat.
        n_repeats: Number of times the k-fold split is repeated.
        seed: Random seed for the CV splitter.

    Returns:
        DataFrame with one row per fold: ``repeat``, ``fold``, ranking
        metrics (``roc_auc``, ``pr_auc``, ``recall_at_10pct``, ``brier``),
        and operating-point metrics at threshold 0.5 (``precision_at_0.5``,
        ``recall_at_0.5``, ``f1_at_0.5``).
    """
    spec = STRATEGIES[strategy]

    X_onehot, y = build_features(df, months)
    X_coded = cat_idx = None
    if spec.representation == "coded":
        X_coded, _, cat_idx = build_features_coded(df, months)

    return _repeated_cv_from_arrays(
        X_onehot,
        y,
        strategy,
        n_splits=n_splits,
        n_repeats=n_repeats,
        seed=seed,
        X_coded=X_coded,
        cat_idx=cat_idx,
        cat_cols=CAT if spec.representation == "coded" else None,
    )


def repeated_cv_xy(
    X: pd.DataFrame,
    y: np.ndarray,
    strategy: str,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Run leakage-safe repeated stratified CV on a pre-built ``(X, y)``.

    Dataset-agnostic entry point for any already-built feature matrix (e.g.
    the Taiwan bankruptcy dataset). Only supports strategies with
    ``representation == "onehot"`` (i.e. not ``smotenc``): a plain feature
    matrix has no categorical-code representation to resample against.

    Args:
        X: Feature matrix (any dataset).
        y: Binary target array aligned with ``X``.
        strategy: Key into ``src.evaluation.strategies.STRATEGIES``.
        n_splits: Number of CV folds per repeat.
        n_repeats: Number of times the k-fold split is repeated.
        seed: Random seed for the CV splitter.

    Returns:
        Same schema as ``repeated_cv``.

    Raises:
        ValueError: If ``strategy`` needs a coded categorical representation.
    """
    spec = STRATEGIES[strategy]
    if spec.representation == "coded":
        raise ValueError(
            f"Strategy '{strategy}' needs a coded categorical representation; "
            "repeated_cv_xy only supports plain feature matrices with no "
            "categorical columns."
        )
    return _repeated_cv_from_arrays(
        X, y, strategy, n_splits=n_splits, n_repeats=n_repeats, seed=seed
    )
