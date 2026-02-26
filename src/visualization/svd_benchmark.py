"""SVD benchmark visualization (OVHD-01, OVHD-02, OVHD-03).

Grouped bar chart comparing SVD methods by target and scatter plot
showing accuracy-cost tradeoff.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import PALETTE


def plot_svd_benchmark_bars(benchmark_data: dict) -> plt.Figure:
    """Grouped bar chart: SVD computation cost by target and method.

    Args:
        benchmark_data: Dict with 'by_target' mapping target names
            to dicts with 'full_svd_ms', 'randomized_svd_ms',
            'values_only_ms'.

    Returns:
        Matplotlib Figure with grouped bar chart.
    """
    by_target = benchmark_data.get("by_target", {})
    targets = sorted(by_target.keys())
    n_targets = len(targets)

    methods = ["full_svd_ms", "randomized_svd_ms", "values_only_ms"]
    method_labels = ["Full SVD", "Randomized SVD", "Values-only SVD"]
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.25
    x = np.arange(n_targets)

    for i, (method_key, label) in enumerate(zip(methods, method_labels)):
        values = [by_target[t].get(method_key, 0) for t in targets]
        offset = (i - (n_methods - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values,
            width=bar_width,
            color=PALETTE[i % len(PALETTE)],
            label=label,
            edgecolor="gray",
            linewidth=0.5,
        )
        # Value labels on top of bars
        for bar_rect, val in zip(bars, values):
            ax.text(
                bar_rect.get_x() + bar_rect.get_width() / 2,
                bar_rect.get_height() + 0.01 * max(values) if max(values) > 0 else 0.001,
                f"{val:.2f}",
                ha="center", va="bottom",
                fontsize=7,
            )

    # Format target labels with matrix shape
    target_labels = []
    for t in targets:
        shape = by_target[t].get("matrix_shape", [])
        if shape:
            target_labels.append(f"{t.upper()}\n({shape[0]}x{shape[1]})")
        else:
            target_labels.append(t.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.set_ylabel("Time (ms per call)")
    ax.set_title("SVD Computation Cost by Target and Method")
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig


def plot_svd_accuracy_tradeoff(benchmark_data: dict) -> plt.Figure:
    """Scatter plot: accuracy vs cost for each (target, method) combination.

    Full SVD points are plotted at error=0 (reference). Randomized and
    values-only methods are plotted at their measured Frobenius error.

    Args:
        benchmark_data: Dict with 'by_target' containing timing and
            accuracy results.

    Returns:
        Matplotlib Figure with accuracy-cost scatter plot.
    """
    by_target = benchmark_data.get("by_target", {})
    targets = sorted(by_target.keys())

    fig, ax = plt.subplots(figsize=(8, 5))

    # Markers for methods
    method_markers = {"Full SVD": "o", "Randomized SVD": "s", "Values-only SVD": "^"}

    for t_idx, target in enumerate(targets):
        t_data = by_target[target]
        color = PALETTE[t_idx % len(PALETTE)]

        # Full SVD: reference point at error=0
        ax.scatter(
            t_data["full_svd_ms"], 0,
            color=color, marker="o", s=80,
            label=f"{target.upper()} - Full" if t_idx == 0 else None,
            zorder=3,
        )

        # Randomized SVD
        rand_error = t_data.get("randomized_frob_error", 0)
        if np.isfinite(rand_error):
            ax.scatter(
                t_data["randomized_svd_ms"], rand_error,
                color=color, marker="s", s=80,
                label=f"{target.upper()} - Randomized" if t_idx == 0 else None,
                zorder=3,
            )

        # Values-only: error is effectively 0 (same singular values)
        # Compute actual error from correlation
        vals_corr = t_data.get("values_only_sv_correlation", 1.0)
        vals_error = 1.0 - vals_corr if np.isfinite(vals_corr) else 0
        ax.scatter(
            t_data["values_only_ms"], vals_error,
            color=color, marker="^", s=80,
            label=f"{target.upper()} - Values-only" if t_idx == 0 else None,
            zorder=3,
        )

    # Annotate each point with target name
    for t_idx, target in enumerate(targets):
        t_data = by_target[target]
        color = PALETTE[t_idx % len(PALETTE)]

        ax.annotate(
            target.upper(),
            (t_data["full_svd_ms"], 0),
            textcoords="offset points", xytext=(5, 5),
            fontsize=7, color=color,
        )

    ax.set_xlabel("Time (ms per call)")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("SVD Accuracy-Cost Tradeoff")

    # Create custom legend for methods only
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markersize=8, label="Full SVD"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None",
               markersize=8, label="Randomized SVD"),
        Line2D([0], [0], marker="^", color="gray", linestyle="None",
               markersize=8, label="Values-only SVD"),
    ]
    # Add target colors
    for t_idx, target in enumerate(targets):
        color = PALETTE[t_idx % len(PALETTE)]
        legend_elements.append(
            Line2D([0], [0], marker="o", color=color, linestyle="None",
                   markersize=8, label=target.upper())
        )

    ax.legend(handles=legend_elements, fontsize=7, loc="best")

    fig.tight_layout()
    return fig
