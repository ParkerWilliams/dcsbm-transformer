"""Event-aligned SVD metric plots (PLOT-01).

Plots SVD metrics aligned to failure events: position 0 = failure event
(resolution_step), negative positions = before failure, positive = after.
Includes confidence bands and separate traces for violations vs controls.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.event_extraction import AnalysisEvent
from src.evaluation.behavioral import RuleOutcome
from src.visualization.style import CONTROL_COLOR, VIOLATION_COLOR


def plot_event_aligned(
    metric_values: np.ndarray,
    events: list[AnalysisEvent],
    window: int = 10,
    metric_name: str = "SVD metric",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot SVD metric aligned to failure events.

    Position 0 = failure event (resolution_step).
    Negative positions = before failure.
    Positive positions = after failure.

    Args:
        metric_values: Metric array, shape [n_sequences, max_steps].
        events: List of AnalysisEvent records.
        window: Number of steps before/after to show.
        metric_name: Name for Y-axis label.
        ax: Optional axes to plot on. Creates new figure if None.

    Returns:
        The matplotlib Figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    violations = [e for e in events if e.outcome == RuleOutcome.VIOLATED]
    controls = [e for e in events if e.outcome == RuleOutcome.FOLLOWED]

    positions = np.arange(-window, window + 1)

    for group, label, color in [
        (violations, "Violation", VIOLATION_COLOR),
        (controls, "Control (baseline)", CONTROL_COLOR),
    ]:
        if not group:
            continue

        aligned = np.full((len(group), len(positions)), np.nan)
        for i, ev in enumerate(group):
            for j, pos in enumerate(positions):
                idx = ev.resolution_step + pos
                if 0 <= idx < metric_values.shape[1]:
                    aligned[i, j] = metric_values[ev.walk_idx, idx]

        mean = np.nanmean(aligned, axis=0)
        n_valid = np.sum(np.isfinite(aligned), axis=0).astype(float)
        std = np.nanstd(aligned, axis=0)
        se = np.where(n_valid > 1, std / np.sqrt(n_valid), 0.0)

        ax.plot(positions, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(
            positions,
            mean - 1.96 * se,
            mean + 1.96 * se,
            alpha=0.2,
            color=color,
        )

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7, label="Failure event")
    ax.set_xlabel("Position relative to failure event")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Event-aligned: {metric_name}")
    ax.legend(fontsize=8)

    return fig
