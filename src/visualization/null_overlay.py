"""Null distribution overlay on event-aligned SVD metric plots.

Renders a gray 95% CI band and solid gray median line representing the
null (jumper-free) Grassmannian drift distribution, overlaid on the
standard event-aligned violation/control plot.

Color scheme per CONTEXT.md: light gray band, solid gray median line.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.event_extraction import AnalysisEvent
from src.visualization.event_aligned import plot_event_aligned

# Null overlay colors (per CONTEXT.md locked decision)
NULL_BAND_COLOR = (0.7, 0.7, 0.7, 0.3)  # light gray with alpha
NULL_MEDIAN_COLOR = (0.5, 0.5, 0.5, 1.0)  # solid gray


def compute_null_distribution_stats(
    null_metric_array: np.ndarray,
    event_positions: list[int],
    window: int = 10,
) -> dict[int, dict[str, float]]:
    """Compute null distribution statistics at each position relative to events.

    For each relative position in [-window, +window], computes the median
    and 95% CI (2.5th and 97.5th percentiles) of the null metric values
    at the corresponding absolute positions.

    Args:
        null_metric_array: Shape [n_null_sequences, max_steps-1].
        event_positions: Absolute positions from violation events.
        window: Number of steps before/after event to include.

    Returns:
        Dict mapping relative position -> {"median": float, "ci_low": float, "ci_high": float}.
    """
    n_steps = null_metric_array.shape[1]
    result: dict[int, dict[str, float]] = {}

    for rel_pos in range(-window, window + 1):
        pooled_values = []

        for ev_pos in event_positions:
            abs_pos = ev_pos + rel_pos
            if 0 <= abs_pos < n_steps:
                col = null_metric_array[:, abs_pos]
                finite = col[np.isfinite(col)]
                if len(finite) > 0:
                    pooled_values.append(finite)

        if pooled_values:
            all_values = np.concatenate(pooled_values)
            if len(all_values) > 0:
                result[rel_pos] = {
                    "median": float(np.median(all_values)),
                    "ci_low": float(np.percentile(all_values, 2.5)),
                    "ci_high": float(np.percentile(all_values, 97.5)),
                }

    return result


def plot_event_aligned_with_null(
    metric_values: np.ndarray,
    events: list[AnalysisEvent],
    null_stats: dict[int, dict[str, float]],
    window: int = 10,
    metric_name: str = "SVD metric",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Event-aligned plot with null distribution overlay.

    Renders the standard violation/control traces from plot_event_aligned(),
    then overlays the null distribution as:
    - Gray 95% CI band (shaded)
    - Solid gray median line

    Args:
        metric_values: Metric array [n_sequences, max_steps].
        events: List of AnalysisEvent records.
        null_stats: Output from compute_null_distribution_stats().
        window: Number of steps before/after event.
        metric_name: Label for Y-axis.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    # Render base violation/control traces
    fig = plot_event_aligned(metric_values, events, window, metric_name, ax)
    plot_ax = fig.axes[0]

    # Build arrays from null_stats
    positions = list(range(-window, window + 1))
    medians = []
    ci_low = []
    ci_high = []

    for p in positions:
        if p in null_stats:
            medians.append(null_stats[p]["median"])
            ci_low.append(null_stats[p]["ci_low"])
            ci_high.append(null_stats[p]["ci_high"])
        else:
            medians.append(np.nan)
            ci_low.append(np.nan)
            ci_high.append(np.nan)

    positions_arr = np.array(positions)
    medians_arr = np.array(medians)
    ci_low_arr = np.array(ci_low)
    ci_high_arr = np.array(ci_high)

    # Overlay null 95% CI band
    plot_ax.fill_between(
        positions_arr,
        ci_low_arr,
        ci_high_arr,
        color=NULL_BAND_COLOR,
        label="Null 95% CI",
        zorder=1,
    )

    # Overlay null median line
    plot_ax.plot(
        positions_arr,
        medians_arr,
        color=NULL_MEDIAN_COLOR,
        linewidth=1.5,
        linestyle="-",
        label="Null median",
        zorder=2,
    )

    # Update legend
    plot_ax.legend(fontsize=8)

    return fig
