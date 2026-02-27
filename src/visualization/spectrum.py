"""Spectrum trajectory visualization: curvature timeseries and AUROC plots.

Phase 15: Advanced Analysis (SPEC-03).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import (
    PALETTE,
    THRESHOLD_COLOR,
    VIOLATION_COLOR,
    apply_style,
)


def plot_curvature_timeseries(
    curvature: np.ndarray,
    failure_indices: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot curvature time series with optional failure event markers.

    Args:
        curvature: Curvature values, shape [n_steps] or [n_sequences, n_steps].
            If 2D, plots mean +/- std across sequences.
        failure_indices: Optional failure step indices for vertical markers.
        ax: Optional axes to plot on.

    Returns:
        Matplotlib Figure.
    """
    apply_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if curvature.ndim == 2:
        mean_curv = np.nanmean(curvature, axis=0)
        std_curv = np.nanstd(curvature, axis=0)
        steps = np.arange(len(mean_curv))
        ax.plot(steps, mean_curv, color=PALETTE[0], linewidth=1.5, label="Mean curvature")
        ax.fill_between(
            steps,
            mean_curv - std_curv,
            mean_curv + std_curv,
            alpha=0.2,
            color=PALETTE[0],
        )
    else:
        steps = np.arange(len(curvature))
        ax.plot(steps, curvature, color=PALETTE[0], linewidth=1.0, label="Curvature")

    if failure_indices is not None:
        for fi in failure_indices:
            if fi >= 0:
                ax.axvline(x=fi, color=VIOLATION_COLOR, alpha=0.3, linewidth=0.8)
        # Add a single label for the legend
        ax.axvline(x=-1, color=VIOLATION_COLOR, alpha=0.3, linewidth=0.8, label="Violation")

    ax.set_xlabel("Step")
    ax.set_ylabel("Curvature")
    ax.set_title("Spectral Curvature Time Series")
    ax.legend(loc="upper right", fontsize=8)

    return fig


def plot_spectrum_auroc(
    by_metric: dict,
    r_value: int,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot AUROC vs lookback distance for curvature/torsion metrics.

    Args:
        by_metric: Dict mapping metric_name -> {auroc_by_lookback, peak_auroc, peak_lookback}.
        r_value: The r value for this group.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    apply_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for i, (metric_name, data) in enumerate(sorted(by_metric.items())):
        aurocs = data.get("auroc_by_lookback", [])
        if not aurocs:
            continue
        # Replace None with NaN for plotting
        aurocs_clean = [v if v is not None else np.nan for v in aurocs]
        lookbacks = np.arange(1, len(aurocs_clean) + 1)
        color = PALETTE[i % len(PALETTE)]

        # Shorten metric name for legend
        short_name = metric_name.split(".")[-1]
        layer_part = metric_name.split(".")[1] if "." in metric_name else ""
        label = f"{short_name} ({layer_part})" if layer_part else short_name

        ax.plot(lookbacks, aurocs_clean, marker="o", markersize=4,
                color=color, linewidth=1.5, label=label)

    # Chance level
    ax.axhline(y=0.5, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.0,
               label="Chance (0.5)")

    ax.set_xlabel("Lookback distance j")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Exploratory: Spectrum Geometry AUROC (r={r_value})")
    ax.legend(loc="best", fontsize=7)
    ax.set_ylim(0.3, 1.0)

    return fig


def plot_spectrum_trajectory_sample(
    spectra: np.ndarray,
    sequence_idx: int = 0,
    layer_idx: int = 0,
    top_k: int = 4,
) -> plt.Figure:
    """Plot top-k singular value trajectories over time for a single sequence.

    Args:
        spectra: Spectrum array, shape [n_sequences, n_steps, k].
        sequence_idx: Which sequence to plot.
        layer_idx: Layer index (for title only).
        top_k: Number of top singular values to show.

    Returns:
        Matplotlib Figure.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 4))

    if spectra.ndim == 3:
        sv_trajectory = spectra[sequence_idx]  # [n_steps, k]
    else:
        sv_trajectory = spectra  # [n_steps, k]

    n_steps, k_total = sv_trajectory.shape
    k_plot = min(top_k, k_total)
    steps = np.arange(n_steps)

    for i in range(k_plot):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(steps, sv_trajectory[:, i], color=color, linewidth=1.0,
                label=f"$\\sigma_{{{i+1}}}$")

    ax.set_xlabel("Step")
    ax.set_ylabel("Singular Value")
    ax.set_title(f"Spectrum Trajectory (seq={sequence_idx}, layer={layer_idx})")
    ax.legend(loc="best", fontsize=8)

    return fig
