"""Confusion matrix for 4-class behavioral outcomes (PLOT-04).

Renders a 2x2 heatmap of edge valid/invalid vs rule followed/violated,
excluding NOT_APPLICABLE steps. Shows both counts and percentages.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.evaluation.behavioral import RuleOutcome


def plot_confusion_matrix(
    edge_valid: np.ndarray,
    rule_outcome: np.ndarray,
) -> plt.Figure:
    """Plot 4-class behavioral confusion matrix.

    Rows: Edge valid / Edge invalid
    Cols: Rule followed / Rule violated
    Excludes NOT_APPLICABLE steps.

    Args:
        edge_valid: Boolean array, shape [n_steps] or [n_sequences, n_steps].
        rule_outcome: Integer array (RuleOutcome values), same shape.

    Returns:
        The matplotlib Figure containing the heatmap.
    """
    # Flatten if multi-dimensional
    ev = np.asarray(edge_valid).ravel()
    ro = np.asarray(rule_outcome).ravel()

    # Filter out NOT_APPLICABLE
    applicable_mask = ro != RuleOutcome.NOT_APPLICABLE
    ev = ev[applicable_mask]
    ro = ro[applicable_mask]

    # Build 2x2 counts
    # Rows: [valid, invalid], Cols: [followed, violated]
    matrix = np.zeros((2, 2), dtype=int)
    matrix[0, 0] = int(np.sum(ev & (ro == RuleOutcome.FOLLOWED)))     # valid + followed
    matrix[0, 1] = int(np.sum(ev & (ro == RuleOutcome.VIOLATED)))     # valid + violated
    matrix[1, 0] = int(np.sum(~ev & (ro == RuleOutcome.FOLLOWED)))    # invalid + followed
    matrix[1, 1] = int(np.sum(~ev & (ro == RuleOutcome.VIOLATED)))    # invalid + violated

    total = matrix.sum()

    # Annotation: count and percentage
    annot = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            count = matrix[i, j]
            pct = 100.0 * count / total if total > 0 else 0.0
            annot[i, j] = f"{count}\n({pct:.1f}%)"

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Rule followed", "Rule violated"],
        yticklabels=["Edge valid", "Edge invalid"],
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title("Behavioral outcome confusion matrix")
    ax.set_xlabel("Rule compliance")
    ax.set_ylabel("Edge validity")

    fig.tight_layout()
    return fig
