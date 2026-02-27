"""Compliance curve visualization: dual-axis publication figure.

Phase 15: Advanced Analysis (COMP-02).
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import (
    PALETTE,
    THRESHOLD_COLOR,
    apply_style,
)


def plot_compliance_curve(
    compliance_data: dict,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Create dual-axis publication figure for compliance phase transition.

    Left y-axis: Rule compliance rate (0-1) with error bands.
    Right y-axis: Predictive horizon (steps) with error bands.
    Vertical dashed line at r/w = 1.0 (critical boundary).

    Args:
        compliance_data: Aggregated compliance curve dict from
            aggregate_compliance_curve, containing r_over_w_values,
            rule_compliance, edge_compliance, and predictive_horizon.
        ax: Optional axes (if provided, right axis is created via twinx).

    Returns:
        Matplotlib Figure.
    """
    apply_style()

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
    else:
        ax1 = ax
        fig = ax1.figure

    r_over_w = np.array(compliance_data["r_over_w_values"])
    rule_mean = np.array(compliance_data["rule_compliance"]["mean"])
    rule_std = np.array(compliance_data["rule_compliance"]["std"])
    edge_mean = np.array(compliance_data["edge_compliance"]["mean"])
    edge_std = np.array(compliance_data["edge_compliance"]["std"])

    # Left y-axis: Rule compliance
    color_rule = PALETTE[0]  # blue
    color_edge = PALETTE[4]  # lighter

    ax1.plot(r_over_w, rule_mean, color=color_rule, linewidth=2.0,
             marker="o", markersize=5, label="Rule compliance", zorder=3)
    ax1.fill_between(r_over_w, rule_mean - rule_std, rule_mean + rule_std,
                     alpha=0.15, color=color_rule, zorder=2)

    # Also show edge compliance as dashed
    ax1.plot(r_over_w, edge_mean, color=color_edge, linewidth=1.5,
             marker="s", markersize=4, linestyle="--", label="Edge compliance", zorder=2)
    ax1.fill_between(r_over_w, edge_mean - edge_std, edge_mean + edge_std,
                     alpha=0.1, color=color_edge, zorder=1)

    ax1.set_xlabel("r / w ratio", fontsize=11)
    ax1.set_ylabel("Compliance Rate", color=color_rule, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_rule)
    ax1.set_ylim(-0.05, 1.05)

    # Vertical line at r/w = 1.0
    ax1.axvline(x=1.0, color=THRESHOLD_COLOR, linestyle=":", linewidth=1.5,
                alpha=0.7, label="r = w boundary")

    # Right y-axis: Predictive horizon (if available)
    horizon_mean = compliance_data.get("predictive_horizon", {}).get("mean", [])
    horizon_std = compliance_data.get("predictive_horizon", {}).get("std", [])

    has_horizon = horizon_mean and any(h is not None for h in horizon_mean)

    if has_horizon:
        ax2 = ax1.twinx()
        color_horizon = PALETTE[1]  # orange

        # Filter out None values for plotting
        h_mean = np.array([h if h is not None else np.nan for h in horizon_mean])
        h_std = np.array([s if s is not None else np.nan for s in horizon_std])

        valid = ~np.isnan(h_mean)
        if valid.any():
            ax2.plot(r_over_w[valid], h_mean[valid], color=color_horizon,
                     linewidth=2.0, marker="^", markersize=5,
                     label="Predictive horizon", zorder=3)
            ax2.fill_between(r_over_w[valid],
                             h_mean[valid] - h_std[valid],
                             h_mean[valid] + h_std[valid],
                             alpha=0.15, color=color_horizon, zorder=2)

        ax2.set_ylabel("Predictive Horizon (steps)", color=color_horizon, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color_horizon)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc="center right", fontsize=8)
    else:
        ax1.legend(loc="best", fontsize=8)

    ax1.set_title("Compliance Phase Transition", fontsize=12)
    fig.tight_layout()

    return fig


def plot_compliance_scatter(
    raw_points: list[dict],
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Scatter plot of individual compliance points colored by seed.

    Args:
        raw_points: List of compliance point dicts with r_over_w,
            rule_compliance, and seed fields.
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    apply_style()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # Group by seed for coloring
    seeds = sorted(set(p["seed"] for p in raw_points))
    seed_to_color = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(seeds)}

    for p in raw_points:
        ax.scatter(
            p["r_over_w"],
            p["rule_compliance"],
            color=seed_to_color[p["seed"]],
            s=30,
            alpha=0.7,
            zorder=3,
        )

    # Legend for seeds
    for i, seed in enumerate(seeds):
        ax.scatter([], [], color=seed_to_color[seed], label=f"Seed {seed}")

    ax.axvline(x=1.0, color=THRESHOLD_COLOR, linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("r / w ratio")
    ax.set_ylabel("Rule Compliance Rate")
    ax.set_title("Individual Compliance Points by Seed")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    return fig
