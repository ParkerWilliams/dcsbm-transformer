"""Calibration diagnostics visualization (PRCL-02, PRCL-03).

Reliability diagrams with ECE annotations and predicted probability
histograms. One diagram per metric, lookback distances as colored lines.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import PALETTE, THRESHOLD_COLOR


def plot_reliability_diagram(
    metric_name: str,
    lookback_data: list[dict],
    r_value: int,
) -> plt.Figure:
    """Plot reliability diagram with predicted probability histogram.

    Args:
        metric_name: Name of the SVD metric.
        lookback_data: List of dicts (one per lookback j=1..r), each with:
            - 'ece': float (may be NaN)
            - 'fraction_of_positives': list[float] (optional)
            - 'mean_predicted_value': list[float] (optional)
            - 'bin_counts': list[int] (optional)
        r_value: Maximum lookback distance.

    Returns:
        Matplotlib Figure with reliability diagram (top) and histogram (bottom).
    """
    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Perfect calibration diagonal
    ax_rel.plot(
        [0, 1], [0, 1],
        color=THRESHOLD_COLOR,
        linestyle="--",
        alpha=0.7,
        label="Perfect calibration",
    )

    # Plot each lookback distance
    first_valid_bin_counts = None
    for j_idx, data in enumerate(lookback_data):
        ece = data.get("ece", float("nan"))
        fop = data.get("fraction_of_positives", [])
        mpv = data.get("mean_predicted_value", [])
        bin_counts = data.get("bin_counts", [])

        if not fop or not mpv or not np.isfinite(ece):
            continue

        j = j_idx + 1
        color = PALETTE[j_idx % len(PALETTE)]

        ax_rel.plot(
            mpv, fop,
            color=color,
            linewidth=1.5,
            marker="s",
            markersize=4,
            label=f"j={j} (ECE={ece:.3f})",
        )

        if first_valid_bin_counts is None and bin_counts:
            first_valid_bin_counts = bin_counts

    ax_rel.set_ylabel("Fraction of positives")
    ax_rel.set_title(f"Reliability Diagram: {metric_name}")
    ax_rel.set_xlim(-0.05, 1.05)
    ax_rel.set_ylim(-0.05, 1.05)
    ax_rel.legend(fontsize=7, loc="best")

    # Histogram of predicted probabilities
    if first_valid_bin_counts is not None:
        n_bins = len(first_valid_bin_counts)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = 1.0 / n_bins * 0.8

        ax_hist.bar(
            bin_centers,
            first_valid_bin_counts,
            width=bar_width,
            color=PALETTE[0],
            alpha=0.6,
            edgecolor="gray",
        )

    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    return fig
