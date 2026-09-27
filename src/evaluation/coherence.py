"""Temporal-coherence diagnostics for real vs. synthetic panel trajectories.

These helpers preserve the candidate-mechanism probe from notebook 04. The
result was mostly negative: SMOTE did not make trajectories locally jagged,
and the per-signal coherence metrics did not cleanly explain the ranking
harm seen in notebook 02.

``roughness`` is kept for reference but demoted: it is a magnitude/local-
difference statistic that linear interpolation (SMOTE) mechanically shrinks
just by averaging two curves, regardless of whether the result is a
plausible single-entity trajectory. ``monotonicity`` (Spearman correlation
with the time index) is a rank/shape statistic instead, which is not
mechanically reduced by averaging two arbitrary sequences, making it a more
robust coherence proxy.
"""

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


def lag1_autocorrelation(sequence: np.ndarray) -> float:
    """Pearson lag-1 autocorrelation of a single 1D trajectory.

    Args:
        sequence: 1D array of length ``T``.

    Returns:
        Pearson correlation between ``sequence[:-1]`` and ``sequence[1:]``,
        or NaN if the sequence (or either shifted half) has zero variance.
    """
    sequence = np.asarray(sequence, dtype=float)
    if len(sequence) < 2 or np.std(sequence) == 0:
        return float("nan")

    earlier, later = sequence[:-1], sequence[1:]
    if np.std(earlier) == 0 or np.std(later) == 0:
        return float("nan")
    return float(np.corrcoef(earlier, later)[0, 1])


def roughness(sequence: np.ndarray) -> float:
    """Mean absolute second difference of a trajectory, scale-normalized.

    Args:
        sequence: 1D array of length ``T``.

    Returns:
        ``mean(abs(diff(sequence, n=2)))``, normalized by
        ``abs(mean(sequence)) + 1e-6`` so it is comparable across
        differently-scaled sequences.
    """
    sequence = np.asarray(sequence, dtype=float)
    second_diff = np.diff(sequence, n=2)
    scale = abs(np.mean(sequence)) + 1e-6
    return float(np.mean(np.abs(second_diff)) / scale)


def monotonicity(sequence: np.ndarray) -> float:
    """Spearman rank correlation between a trajectory and its time index.

    A coherence proxy that, unlike ``roughness``, is not mechanically
    reduced by linear interpolation: it measures whether the trajectory's
    overall shape follows a consistent trend over time (a rank/shape
    property), not the local magnitude of point-to-point changes (a
    magnitude property that naturally shrinks when averaging two curves).

    Args:
        sequence: 1D array of length ``T``.

    Returns:
        Spearman correlation between ``sequence`` and ``np.arange(T)``, or
        NaN if the sequence has zero variance (undefined rank correlation).
    """
    sequence = np.asarray(sequence, dtype=float)
    if len(sequence) < 2 or np.std(sequence) == 0:
        return float("nan")
    time_index = np.arange(len(sequence))
    correlation, _ = spearmanr(sequence, time_index)
    return float(correlation)


def _valid_mask(sequences: np.ndarray) -> np.ndarray:
    """Boolean mask of rows with a defined (non-NaN) lag-1 autocorrelation."""
    autocorrs = np.array([lag1_autocorrelation(row) for row in sequences])
    return ~np.isnan(autocorrs)


def valid_autocorrelations(sequences: np.ndarray) -> np.ndarray:
    """Per-row lag-1 autocorrelation, dropping rows where it is undefined.

    Args:
        sequences: Array of shape ``(n, T)``, one trajectory per row.

    Returns:
        1D array of autocorrelation values, NaN rows removed.
    """
    sequences = np.asarray(sequences, dtype=float)
    autocorrs = np.array([lag1_autocorrelation(row) for row in sequences])
    return autocorrs[~np.isnan(autocorrs)]


def valid_roughness(sequences: np.ndarray) -> np.ndarray:
    """Per-row roughness, restricted to rows with a defined autocorrelation.

    Kept on the same subset as ``valid_autocorrelations`` so the two metrics
    are always compared on identical rows.

    Args:
        sequences: Array of shape ``(n, T)``, one trajectory per row.

    Returns:
        1D array of roughness values, for the rows retained by
        ``valid_autocorrelations``.
    """
    sequences = np.asarray(sequences, dtype=float)
    valid = _valid_mask(sequences)
    roughnesses = np.array([roughness(row) for row in sequences])
    return roughnesses[valid]


def valid_monotonicities(sequences: np.ndarray) -> np.ndarray:
    """Per-row monotonicity, restricted to rows with a defined autocorrelation.

    Kept on the same subset as ``valid_autocorrelations`` so all metrics are
    always compared on identical rows.

    Args:
        sequences: Array of shape ``(n, T)``, one trajectory per row.

    Returns:
        1D array of monotonicity values, for the rows retained by
        ``valid_autocorrelations``.
    """
    sequences = np.asarray(sequences, dtype=float)
    valid = _valid_mask(sequences)
    monotonicities = np.array([monotonicity(row) for row in sequences])
    return monotonicities[valid]


def coherence_summary(sequences: np.ndarray) -> dict:
    """Summarize the temporal coherence of a group of trajectories.

    Rows whose lag-1 autocorrelation is undefined (constant trajectory) are
    dropped before averaging, for all three metrics, so they are computed on
    the same subset.

    Args:
        sequences: Array of shape ``(n, T)``, one trajectory per row.

    Returns:
        Dict with ``autocorr_mean``, ``autocorr_median``, ``roughness_mean``,
        ``roughness_median``, ``monotonicity_mean``, ``monotonicity_median``,
        ``n`` (rows retained), and ``n_dropped`` (rows excluded for
        undefined autocorrelation).
    """
    sequences = np.asarray(sequences, dtype=float)
    autocorrs = valid_autocorrelations(sequences)
    roughnesses = valid_roughness(sequences)
    monotonicities = valid_monotonicities(sequences)
    n_dropped = len(sequences) - len(autocorrs)

    return {
        "autocorr_mean": float(np.mean(autocorrs)) if len(autocorrs) else float("nan"),
        "autocorr_median": float(np.median(autocorrs)) if len(autocorrs) else float("nan"),
        "roughness_mean": float(np.mean(roughnesses)) if len(roughnesses) else float("nan"),
        "roughness_median": float(np.median(roughnesses)) if len(roughnesses) else float("nan"),
        "monotonicity_mean": float(np.mean(monotonicities))
        if len(monotonicities)
        else float("nan"),
        "monotonicity_median": float(np.median(monotonicities))
        if len(monotonicities)
        else float("nan"),
        "n": int(len(autocorrs)),
        "n_dropped": int(n_dropped),
    }


def median_bootstrap_ci(
    values: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Bootstrap 95% confidence interval for the median.

    Args:
        values: 1D array of values.
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Tuple ``(median, ci_low, ci_high)``: the sample median and the
        2.5th/97.5th percentiles of the bootstrap distribution of the
        median.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(values)

    boot_medians = np.empty(n_boot)
    max_draws_per_batch = 5_000_000
    batch_size = max(1, min(n_boot, max_draws_per_batch // n))

    start = 0
    while start < n_boot:
        stop = min(start + batch_size, n_boot)
        samples = rng.choice(values, size=(stop - start, n), replace=True)
        boot_medians[start:stop] = np.median(samples, axis=1)
        start = stop

    ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
    return float(np.median(values)), float(ci_low), float(ci_high)


def size_matched_test(
    group_a: np.ndarray,
    group_b: np.ndarray,
    n_boot: int = 1000,
    subsample: int | None = None,
    seed: int = 42,
) -> dict:
    """Power-fair two-sample comparison via repeated size-matched subsampling.

    When one group is far larger than the other, a single Mann-Whitney U
    test on the full samples conflates the true effect with the size
    mismatch's effect on power. This repeatedly subsamples both groups down
    to a common size and runs a two-sided Mann-Whitney U test each time,
    returning the distribution of p-values and rank-biserial effect sizes
    across repeats.

    Args:
        group_a: First group's per-row metric values.
        group_b: Second group's per-row metric values.
        n_boot: Number of subsampling repeats.
        subsample: Target sample size per group per repeat. Defaults to
            ``min(len(group_a), len(group_b))``.
        seed: Random seed.

    Returns:
        Dict with ``median_p``, ``frac_significant`` (fraction of repeats
        with ``p < 0.05``), ``p_values`` (array, one per repeat), and
        ``effect_sizes`` (array of rank-biserial correlations, one per
        repeat).
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    n = subsample if subsample is not None else min(len(group_a), len(group_b))
    rng = np.random.RandomState(seed)

    p_values = np.empty(n_boot)
    effect_sizes = np.empty(n_boot)
    for i in range(n_boot):
        sample_a = rng.choice(group_a, size=n, replace=len(group_a) < n)
        sample_b = rng.choice(group_b, size=n, replace=len(group_b) < n)
        u_stat, p_value = mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        p_values[i] = p_value
        effect_sizes[i] = 1 - (2 * u_stat) / (n * n)

    return {
        "median_p": float(np.median(p_values)),
        "frac_significant": float(np.mean(p_values < 0.05)),
        "p_values": p_values,
        "effect_sizes": effect_sizes,
    }
