"""Training convergence curve plots (PLOT-02).

Plots training loss per step and edge/rule compliance per epoch
with gate threshold reference lines.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import PALETTE, THRESHOLD_COLOR


# Default gate thresholds from training pipeline
DEFAULT_GATE_THRESHOLDS = {
    "edge_compliance": 0.95,
    "rule_compliance": 0.80,
}


def plot_training_curves(
    curves: dict[str, list[float]],
    gate_thresholds: dict[str, float] | None = None,
) -> plt.Figure:
    """Plot training loss and compliance curves.

    Creates a figure with two subplots:
    - Left: Training loss vs step number
    - Right: Edge and rule compliance vs epoch with gate thresholds

    Args:
        curves: Dict with keys 'train_loss', 'edge_compliance', 'rule_compliance'.
            Each value is a list of floats.
        gate_thresholds: Optional dict mapping compliance name to threshold.
            Defaults to edge=0.95, rule=0.80.

    Returns:
        The matplotlib Figure containing both subplots.
    """
    if gate_thresholds is None:
        gate_thresholds = DEFAULT_GATE_THRESHOLDS

    fig, (ax_loss, ax_compliance) = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: training loss
    train_loss = curves.get("train_loss", [])
    if train_loss:
        steps = np.arange(len(train_loss))
        ax_loss.plot(steps, train_loss, color=PALETTE[0], linewidth=0.5, alpha=0.7)
        # Add smoothed line if enough points
        if len(train_loss) > 50:
            kernel_size = max(len(train_loss) // 50, 1)
            smoothed = np.convolve(
                train_loss, np.ones(kernel_size) / kernel_size, mode="valid"
            )
            smooth_steps = np.arange(kernel_size - 1, kernel_size - 1 + len(smoothed))
            ax_loss.plot(
                smooth_steps, smoothed, color=PALETTE[0], linewidth=2, label="Smoothed"
            )
            ax_loss.legend(fontsize=8)
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Training loss")

    # Right subplot: compliance curves
    edge_compliance = curves.get("edge_compliance", [])
    rule_compliance = curves.get("rule_compliance", [])

    if edge_compliance:
        epochs = np.arange(1, len(edge_compliance) + 1)
        ax_compliance.plot(
            epochs, edge_compliance, color=PALETTE[0], linewidth=1.5,
            label="Edge compliance", marker="o", markersize=3,
        )
    if rule_compliance:
        epochs = np.arange(1, len(rule_compliance) + 1)
        ax_compliance.plot(
            epochs, rule_compliance, color=PALETTE[1], linewidth=1.5,
            label="Rule compliance", marker="s", markersize=3,
        )

    # Gate threshold reference lines
    for name, threshold in gate_thresholds.items():
        linestyle = "--" if "edge" in name else ":"
        ax_compliance.axhline(
            threshold, color=THRESHOLD_COLOR, linestyle=linestyle,
            alpha=0.7, label=f"{name} threshold ({threshold})",
        )

    ax_compliance.set_xlabel("Epoch")
    ax_compliance.set_ylabel("Compliance rate")
    ax_compliance.set_title("Training compliance")
    ax_compliance.set_ylim(0, 1.05)
    ax_compliance.legend(fontsize=8)

    fig.tight_layout()
    return fig
