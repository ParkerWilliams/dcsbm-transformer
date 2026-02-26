"""Pre/post failure distribution comparison plots (PLOT-05).

Compares SVD metric value distributions for positions before
vs after the failure event using violin or box plots.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.visualization.style import CONTROL_COLOR, VIOLATION_COLOR


def plot_pre_post_distributions(
    metric_values: np.ndarray,
    failure_index: np.ndarray,
    window: int = 5,
    metric_name: str = "SVD metric",
    plot_type: str = "box",
) -> plt.Figure:
    """Plot pre-failure vs post-failure metric value distributions.

    For each sequence with a violation (failure_index >= 0):
      - Pre-failure: metric at positions [failure_index - window, failure_index)
      - Post-failure: metric at positions (failure_index, failure_index + window]

    Args:
        metric_values: Metric array, shape [n_sequences, max_steps].
        failure_index: First violation step per sequence, shape [n_sequences].
            -1 means no violation.
        window: Number of steps before/after to include.
        metric_name: Name for Y-axis label.
        plot_type: One of "box", "violin", "histogram".

    Returns:
        The matplotlib Figure containing the comparison plot.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Collect pre and post values
    pre_vals = []
    post_vals = []
    n_seq = metric_values.shape[0]
    max_steps = metric_values.shape[1]

    for seq_idx in range(n_seq):
        fi = int(failure_index[seq_idx])
        if fi < 0:
            continue  # No violation in this sequence

        # Pre-failure: positions [fi - window, fi)
        for pos in range(max(0, fi - window), fi):
            if pos < max_steps:
                val = metric_values[seq_idx, pos]
                if np.isfinite(val):
                    pre_vals.append(val)

        # Post-failure: positions (fi, fi + window]
        for pos in range(fi + 1, min(fi + window + 1, max_steps)):
            val = metric_values[seq_idx, pos]
            if np.isfinite(val):
                post_vals.append(val)

    if not pre_vals and not post_vals:
        ax.text(
            0.5, 0.5, "No violations detected",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="gray",
        )
        ax.set_title(f"Distribution comparison: {metric_name}")
        return fig

    # Build data for plotting
    data = []
    labels = []
    if pre_vals:
        data.append(np.array(pre_vals))
        labels.append("Pre-failure")
    if post_vals:
        data.append(np.array(post_vals))
        labels.append("Post-failure")

    colors = [CONTROL_COLOR, VIOLATION_COLOR][:len(data)]

    if plot_type == "violin" and len(data) >= 2:
        parts = ax.violinplot(data, positions=range(len(data)), showmeans=True)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    elif plot_type == "histogram":
        for i, (d, label) in enumerate(zip(data, labels)):
            ax.hist(d, bins=30, alpha=0.5, color=colors[i], label=label)
        ax.legend()
    else:
        # Default: box plot
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.6)

    ax.set_ylabel(metric_name)
    ax.set_title(f"Distribution comparison: {metric_name}")

    fig.tight_layout()
    return fig
