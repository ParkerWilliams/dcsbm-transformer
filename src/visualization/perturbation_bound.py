"""Bound tightness visualization for softmax filtering bound experiments.

Generates scatter + envelope plots showing empirical spectral change ratios
vs the theoretical bound, and per-magnitude detail histograms.
"""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import PALETTE, apply_style, save_figure

log = logging.getLogger(__name__)

# Colors for adversarial vs random
ADVERSARIAL_COLOR = PALETTE[3]  # red/orange from colorblind palette
RANDOM_COLOR = PALETTE[0]  # blue from colorblind palette
BOUND_COLOR = PALETTE[7] if len(PALETTE) > 7 else "black"


def plot_bound_tightness(perturbation_results: dict[str, Any]) -> plt.Figure:
    """Main bound tightness visualization: scatter + envelope.

    Shows empirical/theoretical ratios for each perturbation magnitude,
    with the theoretical bound as a horizontal line at ratio=1.0.

    Args:
        perturbation_results: Dict from run_perturbation_experiment with
            by_magnitude, tightness_ratio, violation_rate, bound_verified.

    Returns:
        Matplotlib Figure.
    """
    apply_style()

    by_magnitude = perturbation_results.get("by_magnitude", {})
    tightness_ratio = perturbation_results.get("tightness_ratio", 0.0)
    violation_rate = perturbation_results.get("violation_rate", 0.0)
    bound_verified = perturbation_results.get("bound_verified", False)

    # Parse magnitudes and collect data
    magnitudes = sorted(float(k) for k in by_magnitude.keys())

    if not magnitudes:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, "No perturbation data available",
                transform=ax.transAxes, ha="center", va="center")
        return fig

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Collect all ratios for box plots
    adv_data_by_mag = []
    rand_data_by_mag = []
    positions = []

    for i, eps in enumerate(magnitudes):
        eps_key = str(eps)
        mag_data = by_magnitude.get(eps_key, {})
        adv = mag_data.get("adversarial", {})
        rand = mag_data.get("random", {})

        # For scatter plot, use mean and max ratios
        adv_mean = adv.get("mean_ratio", 0.0)
        adv_max = adv.get("max_ratio", 0.0)
        rand_mean = rand.get("mean_ratio", 0.0)
        rand_max = rand.get("max_ratio", 0.0)

        # Plot adversarial points
        ax.scatter(eps, adv_mean, color=ADVERSARIAL_COLOR, marker="D", s=100,
                   zorder=5, label="Adversarial (mean)" if i == 0 else None)
        ax.scatter(eps, adv_max, color=ADVERSARIAL_COLOR, marker="^", s=60,
                   alpha=0.6, zorder=4, label="Adversarial (max)" if i == 0 else None)

        # Plot random points
        ax.scatter(eps, rand_mean, color=RANDOM_COLOR, marker="o", s=80,
                   zorder=5, label="Random (mean)" if i == 0 else None)
        ax.scatter(eps, rand_max, color=RANDOM_COLOR, marker="v", s=60,
                   alpha=0.6, zorder=4, label="Random (max)" if i == 0 else None)

        # Error bars from mean to max
        ax.plot([eps, eps], [adv_mean, adv_max], color=ADVERSARIAL_COLOR,
                alpha=0.4, linewidth=1.5)
        ax.plot([eps, eps], [rand_mean, rand_max], color=RANDOM_COLOR,
                alpha=0.4, linewidth=1.5)

    # Theoretical bound line at ratio = 1.0
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=2,
               label="Theoretical bound", zorder=3)

    # Shade region above bound
    ax.axhspan(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 1.0 else 1.5,
               alpha=0.1, color="red", zorder=1)

    # Set y-axis to start from 0
    y_max = max(1.2, ax.get_ylim()[1])
    ax.set_ylim(0, y_max)

    # Re-shade after setting limits
    ax.axhspan(1.0, y_max, alpha=0.08, color="red", zorder=1,
               label="Bound exceeded zone")

    # Labels and title
    ax.set_xlabel(r"Perturbation magnitude ($\varepsilon$)", fontsize=12)
    ax.set_ylabel("Empirical / Theoretical bound", fontsize=12)
    ax.set_title("Softmax Filtering Bound: Empirical vs Theoretical", fontsize=14)

    # Annotation
    status_text = "VERIFIED" if bound_verified else "FAILED"
    status_color = "green" if bound_verified else "red"
    annotation = (
        f"Tightness ratio: {tightness_ratio:.4f}\n"
        f"Violation rate: {violation_rate:.4f}\n"
        f"Bound: {status_text}"
    )
    ax.text(0.02, 0.98, annotation, transform=ax.transAxes,
            verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor=status_color, alpha=0.8))

    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_bound_by_magnitude(perturbation_results: dict[str, Any]) -> plt.Figure:
    """Per-magnitude detail view with distribution of ratios.

    Creates one subplot per magnitude showing the distribution of
    empirical/theoretical ratios.

    Args:
        perturbation_results: Dict from run_perturbation_experiment.

    Returns:
        Matplotlib Figure.
    """
    apply_style()

    by_magnitude = perturbation_results.get("by_magnitude", {})
    magnitudes = sorted(float(k) for k in by_magnitude.keys())

    if not magnitudes:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, "No perturbation data available",
                transform=ax.transAxes, ha="center", va="center")
        return fig

    n_mags = len(magnitudes)
    fig, axes = plt.subplots(1, n_mags, figsize=(4 * n_mags, 5), squeeze=False)
    axes = axes[0]

    for i, eps in enumerate(magnitudes):
        ax = axes[i]
        eps_key = str(eps)
        mag_data = by_magnitude.get(eps_key, {})

        adv = mag_data.get("adversarial", {})
        rand = mag_data.get("random", {})

        # Collect ratio values for visualization
        adv_mean = adv.get("mean_ratio", 0.0)
        adv_max = adv.get("max_ratio", 0.0)
        rand_mean = rand.get("mean_ratio", 0.0)
        rand_max = rand.get("max_ratio", 0.0)
        n_adv = adv.get("n_total", 0)
        n_rand = rand.get("n_total", 0)

        # Bar chart showing mean and max for both types
        bar_width = 0.35
        x_positions = np.array([0, 1])

        bars_mean = ax.bar(x_positions - bar_width / 2,
                           [adv_mean, rand_mean],
                           bar_width, label="Mean ratio",
                           color=[ADVERSARIAL_COLOR, RANDOM_COLOR], alpha=0.7)

        bars_max = ax.bar(x_positions + bar_width / 2,
                          [adv_max, rand_max],
                          bar_width, label="Max ratio",
                          color=[ADVERSARIAL_COLOR, RANDOM_COLOR], alpha=0.4,
                          edgecolor=[ADVERSARIAL_COLOR, RANDOM_COLOR],
                          linewidth=1.5)

        # Theoretical bound line
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.5)

        # Value labels
        for bar in bars_mean:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        for bar in bars_max:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"Adv\n(n={n_adv})", f"Rand\n(n={n_rand})"])
        ax.set_ylabel("Ratio (empirical / bound)" if i == 0 else "")
        ax.set_title(rf"$\varepsilon$ = {eps}", fontsize=11)
        ax.set_ylim(0, max(1.2, adv_max * 1.3, rand_max * 1.3))
        ax.grid(True, alpha=0.3, axis="y")

        # Count exceeding
        n_exc_adv = adv.get("n_exceeding_bound", 0)
        n_exc_rand = rand.get("n_exceeding_bound", 0)
        ax.text(0.98, 0.02,
                f"Exceeding: {n_exc_adv + n_exc_rand}/{n_adv + n_rand}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                color="red" if (n_exc_adv + n_exc_rand) > 0 else "green")

    fig.suptitle("Perturbation Bound Detail by Magnitude", fontsize=14, y=1.02)
    fig.tight_layout()

    return fig
