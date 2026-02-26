"""Marchenko-Pastur density overlay on empirical QK^T singular value histogram.

Overlays the theoretical MP PDF curve on a histogram of empirical squared
singular values from QK^T, with KS test annotation.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.null_model import marchenko_pastur_pdf


def plot_mp_histogram(
    singular_values: np.ndarray,
    gamma: float,
    mp_result: dict,
    position_label: str = "event",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot empirical SV histogram with MP density overlay.

    Args:
        singular_values: 1D array of empirical singular values from QK^T.
        gamma: Aspect ratio w / d_model.
        mp_result: Dict from run_mp_ks_test() with sigma2, lambda_minus,
            lambda_plus, ks_statistic, ks_p_value.
        position_label: Label for the position (e.g., "event", "pre_event_5").
        ax: Optional axes.

    Returns:
        Matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    # Squared singular values (eigenvalue distribution)
    sv_squared = singular_values ** 2

    # Plot histogram of squared SVs
    ax.hist(
        sv_squared,
        bins=30,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="white",
        label="Empirical",
    )

    # Overlay MP PDF curve
    sigma2 = mp_result.get("sigma2", 1.0)
    lam_minus = mp_result.get("lambda_minus", sigma2 * (1 - np.sqrt(gamma)) ** 2)
    lam_plus = mp_result.get("lambda_plus", sigma2 * (1 + np.sqrt(gamma)) ** 2)

    # Evaluate MP PDF at 200 points across the support
    x_range = np.linspace(lam_minus + 1e-10, lam_plus - 1e-10, 200)
    mp_pdf_values = np.array(
        [marchenko_pastur_pdf(x, gamma, sigma2) for x in x_range]
    )

    ax.plot(
        x_range,
        mp_pdf_values,
        color="crimson",
        linewidth=2,
        label="MP density",
    )

    # KS test annotation
    ks_stat = mp_result.get("ks_statistic", float("nan"))
    ks_p = mp_result.get("ks_p_value", float("nan"))
    ax.text(
        0.97,
        0.95,
        f"KS = {ks_stat:.3f}, p = {ks_p:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    # Labels
    ax.set_xlabel("Squared singular value")
    ax.set_ylabel("Density")
    ax.set_title(f"QK^T SVs vs Marchenko-Pastur ({position_label})")
    ax.legend(fontsize=8)

    return fig
