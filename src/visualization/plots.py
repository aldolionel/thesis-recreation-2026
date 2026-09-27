"""Deterministic plotting helpers for the churn data audit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

_NONCHURN_COLOR = "#1f77b4"
_CHURN_COLOR = "#d62728"
_LEAKAGE_ZONE_MONTHS = ("N-1", "N")


def _shade_leakage_zone(ax: plt.Axes, months: list[str]) -> None:
    """Shade the N-1/N leakage zone on a month-indexed axis."""
    leakage_idx = [i for i, month in enumerate(months) if month in _LEAKAGE_ZONE_MONTHS]
    if leakage_idx:
        ax.axvspan(
            min(leakage_idx) - 0.5,
            max(leakage_idx) + 0.5,
            color="red",
            alpha=0.1,
            label="Leakage zone (N-1, N)",
        )


def plot_median_trajectory(traj_df: pd.DataFrame, signal: str, out_path: Path) -> None:
    """Plot the churn vs non-churn median trajectory across panel months.

    Args:
        traj_df: Output of ``median_trajectory``, columns
            ``[month, churn_median, nonchurn_median]``.
        signal: Signal name, used in axis labels and the title.
        out_path: Destination file path for the saved figure.
    """
    months = traj_df["month"].tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        months, traj_df["nonchurn_median"], marker="o", color=_NONCHURN_COLOR, label="Non-churn"
    )
    ax.plot(months, traj_df["churn_median"], marker="o", color=_CHURN_COLOR, label="Churn")
    _shade_leakage_zone(ax, months)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Median {signal}")
    ax.set_title(f"Median {signal} trajectory: churn vs non-churn")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_strategy_delta_bar(
    sig_df: pd.DataFrame, out_path: Path, baseline_name: str = "none", alpha: float = 0.05
) -> None:
    """Bar chart of mean PR-AUC delta vs a baseline strategy, per strategy.

    Bars are colored red where the strategy is significantly worse than the
    baseline (``mean_delta < 0`` and ``ttest_p < alpha``), gray otherwise.

    Args:
        sig_df: Paired-comparison results indexed by strategy name, with
            columns ``mean_delta`` and ``ttest_p`` (see
            ``src.evaluation.stats.paired_compare``).
        out_path: Destination file path for the saved figure.
        baseline_name: Name of the baseline strategy, used in the title.
        alpha: Significance threshold for the negative-delta highlight.
    """
    strategies = sig_df.index.tolist()
    deltas = sig_df["mean_delta"].to_numpy()
    significant_negative = (sig_df["mean_delta"] < 0) & (sig_df["ttest_p"] < alpha)
    colors = ["#d62728" if flag else "#7f7f7f" for flag in significant_negative]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(strategies, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Delta PR-AUC vs `{baseline_name}`")
    ax.set_title(f"PR-AUC change vs `{baseline_name}`, by imbalance strategy")
    ax.tick_params(axis="x", rotation=30)

    legend_handles = [
        Patch(facecolor="#d62728", label=f"significantly worse (p < {alpha})"),
        Patch(facecolor="#7f7f7f", label="not significant"),
    ]
    ax.legend(handles=legend_handles)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_zero_fraction(zero_df: pd.DataFrame, signal: str, out_path: Path) -> None:
    """Plot the fraction of zero values per month, churn vs non-churn.

    Args:
        zero_df: Output of ``zero_fraction_by_month``, columns
            ``[month, churn_zero_frac, nonchurn_zero_frac]``.
        signal: Signal name, used in axis labels and the title.
        out_path: Destination file path for the saved figure.
    """
    months = zero_df["month"].tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        months, zero_df["nonchurn_zero_frac"], marker="o", color=_NONCHURN_COLOR, label="Non-churn"
    )
    ax.plot(months, zero_df["churn_zero_frac"], marker="o", color=_CHURN_COLOR, label="Churn")
    _shade_leakage_zone(ax, months)
    ax.set_xlabel("Month")
    ax.set_ylabel(f"Fraction of {signal} == 0")
    ax.set_title(f"Zero-fraction of {signal} by month: churn vs non-churn")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coherence_distributions(
    groups: dict[str, np.ndarray], metric_name: str, out_path: Path
) -> None:
    """Overlaid histograms of a per-row coherence metric, one per group.

    Each group's median is marked with a dashed vertical line in the same
    color, since the mean and median can diverge sharply for skewed
    coherence distributions (this is exactly what happened for
    autocorrelation in the first pass of this diagnostic).

    Args:
        groups: Mapping of group label (e.g. ``"real"``, ``"duplicated"``,
            ``"smote_synthetic"``) to a 1D array of per-row metric values
            (already NaN-filtered).
        metric_name: Metric name, used in axis labels and the title (e.g.
            ``"lag-1 autocorrelation"``).
        out_path: Destination file path for the saved figure.
    """
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, values) in enumerate(groups.items()):
        color = colors[i % len(colors)]
        ax.hist(values, bins=40, density=True, alpha=0.5, label=label, color=color)
        ax.axvline(np.median(values), color=color, linestyle="--", linewidth=1.5)
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Density")
    ax.set_title(
        f"Distribution of {metric_name}: real vs duplicated vs SMOTE-synthetic\n"
        "(dashed lines = medians)"
    )
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_illustrative_trajectories(
    real_samples: np.ndarray, synthetic_samples: np.ndarray, out_path: Path
) -> None:
    """Side-by-side line plots of a handful of real vs synthetic trajectories.

    Args:
        real_samples: Array of shape ``(n_samples, T)``, real trajectories.
        synthetic_samples: Array of shape ``(n_samples, T)``, synthetic
            trajectories (e.g. SMOTE-generated).
        out_path: Destination file path for the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for sample in real_samples:
        axes[0].plot(sample, marker="o", alpha=0.8)
    axes[0].set_title("Real minority trajectories")
    axes[0].set_xlabel("Month index (honest window)")
    axes[0].set_ylabel("log1p(USAGE)")

    for sample in synthetic_samples:
        axes[1].plot(sample, marker="o", alpha=0.8)
    axes[1].set_title("SMOTE-synthetic minority trajectories")
    axes[1].set_xlabel("Month index (honest window)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_dataset_delta_bar(
    comparison_df: pd.DataFrame, out_path: Path, baseline_name: str = "none"
) -> None:
    """Grouped bar chart of mean PR-AUC delta vs baseline, strategy x dataset.

    Headline demarcation figure: shows the same strategy having a small
    (neutral) delta on one dataset and a large (harmful) delta on another,
    side by side, to make the structure-dependence of the effect visible.

    Args:
        comparison_df: One row per (dataset, strategy) pair, with columns
            ``dataset``, ``strategy``, ``mean_delta``.
        out_path: Destination file path for the saved figure.
        baseline_name: Name of the baseline strategy, used in the title.
    """
    strategies = sorted(comparison_df["strategy"].unique())
    datasets = sorted(comparison_df["dataset"].unique())
    x = np.arange(len(strategies))
    width = 0.8 / len(datasets)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, dataset in enumerate(datasets):
        subset = comparison_df.loc[comparison_df["dataset"] == dataset].set_index("strategy")
        values = [subset.loc[s, "mean_delta"] for s in strategies]
        offset = (i - (len(datasets) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=dataset, color=colors[i % len(colors)])

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylabel(f"Delta PR-AUC vs `{baseline_name}`")
    ax.set_title("Oversampling effect by dataset structure: neutral vs harmful")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
