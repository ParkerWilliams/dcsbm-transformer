"""Precision-recall curve visualization (PRCL-01, PRCL-03).

Plots AUPRC vs lookback distance per SVD metric, mirroring the
AUROC curve layout from auroc.py.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import PALETTE, THRESHOLD_COLOR


def plot_pr_curves(
    pr_results: dict[str, dict],
    r_value: int,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot AUPRC vs lookback distance for multiple metrics.

    Args:
        pr_results: Maps metric_name -> dict with:
            - 'auprc_by_lookback': list[float] of AUPRC values at j=1..r
            - 'prevalence': float, positive class prevalence (no-skill baseline)
        r_value: Maximum lookback distance.
        ax: Optional axes to plot on. Creates new figure if None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    lookbacks = np.arange(1, r_value + 1)

    # Compute mean prevalence across metrics for no-skill baseline
    prevalences = [
        d.get("prevalence", 0)
        for d in pr_results.values()
        if np.isfinite(d.get("prevalence", float("nan")))
    ]
    mean_prevalence = np.mean(prevalences) if prevalences else 0.0

    # No-skill baseline (prevalence rate)
    if mean_prevalence > 0:
        ax.axhline(
            mean_prevalence,
            color=THRESHOLD_COLOR,
            linestyle="--",
            alpha=0.7,
            label=f"No-skill ({mean_prevalence:.3f})",
        )

    # Plot each metric
    for i, (metric_name, data) in enumerate(sorted(pr_results.items())):
        auprc_values = data.get("auprc_by_lookback", [])
        if not auprc_values:
            continue

        color = PALETTE[i % len(PALETTE)]
        values = np.array(auprc_values[:r_value], dtype=float)
        plot_lookbacks = lookbacks[: len(values)]

        ax.plot(
            plot_lookbacks,
            values,
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label=metric_name,
        )

        # Annotate peak AUPRC
        finite_mask = np.isfinite(values)
        if np.any(finite_mask):
            peak_idx = np.nanargmax(values)
            peak_val = values[peak_idx]
            ax.annotate(
                f"{peak_val:.3f}",
                xy=(plot_lookbacks[peak_idx], peak_val),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Lookback distance j")
    ax.set_ylabel("AUPRC")
    ax.set_title(f"AUPRC vs lookback distance (r={r_value})")
    ax.set_xlim(0.5, r_value + 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7, loc="best")

    return fig
