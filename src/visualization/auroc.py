"""AUROC vs lookback distance curves (PLOT-03).

Plots AUROC at each lookback distance j (1..r) per SVD metric,
with threshold and chance reference lines, optional horizon markers
and bootstrap confidence bands.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import PALETTE, THRESHOLD_COLOR


def plot_auroc_curves(
    auroc_results: dict[str, dict],
    r_value: int,
    threshold: float = 0.75,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot AUROC vs lookback distance for multiple metrics.

    Args:
        auroc_results: Maps metric_name -> dict with:
            - 'auroc_by_lookback': list[float] of AUROC values at j=1..r
            - 'horizon': int, furthest j where AUROC > threshold
            - 'bootstrap_ci': optional list of (low, high) tuples
        r_value: Maximum lookback distance (length of curves).
        threshold: AUROC threshold for predictive horizon.
        ax: Optional axes to plot on. Creates new figure if None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    lookbacks = np.arange(1, r_value + 1)

    # Reference lines
    ax.axhline(
        threshold, color=THRESHOLD_COLOR, linestyle="--", alpha=0.7,
        label=f"Threshold ({threshold})",
    )
    ax.axhline(
        0.5, color="lightgray", linestyle=":", alpha=0.7, label="Chance (0.5)",
    )

    # Plot each metric
    for i, (metric_name, data) in enumerate(sorted(auroc_results.items())):
        auroc_values = data.get("auroc_by_lookback", [])
        if not auroc_values:
            continue

        color = PALETTE[i % len(PALETTE)]
        # Truncate or pad to r_value length
        values = np.array(auroc_values[:r_value], dtype=float)
        plot_lookbacks = lookbacks[:len(values)]

        ax.plot(
            plot_lookbacks, values, color=color, linewidth=1.5,
            marker="o", markersize=4, label=metric_name,
        )

        # Bootstrap confidence bands
        bootstrap_ci = data.get("bootstrap_ci")
        if bootstrap_ci and len(bootstrap_ci) == len(values):
            ci_low = [ci[0] if ci else np.nan for ci in bootstrap_ci]
            ci_high = [ci[1] if ci else np.nan for ci in bootstrap_ci]
            ax.fill_between(
                plot_lookbacks, ci_low, ci_high, alpha=0.15, color=color,
            )

        # Horizon marker
        horizon = data.get("horizon", 0)
        if horizon > 0 and horizon <= r_value:
            ax.axvline(
                horizon, color=color, linestyle=":", alpha=0.4, linewidth=1,
            )

    ax.set_xlabel("Lookback distance j")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs lookback distance")
    ax.set_xlim(0.5, r_value + 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=7, loc="best")

    return fig
