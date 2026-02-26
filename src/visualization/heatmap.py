"""Predictive horizon heatmap across (r, w) parameter grid (PLOT-06).

Renders a 2D heatmap showing predictive horizon values for each
(r, w) configuration. Handles sparse data gracefully with NaN masking.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_horizon_heatmap(
    horizon_data: dict[tuple[int, int], float],
    metric_name: str = "Best primary metric",
    threshold: float = 0.75,
) -> plt.Figure:
    """Plot predictive horizon heatmap across (r, w) parameter grid.

    Args:
        horizon_data: Maps (r_value, w_value) -> horizon value.
            Missing combinations are rendered as masked cells.
        metric_name: Name of the metric for the title.
        threshold: AUROC threshold used for horizon calculation.

    Returns:
        The matplotlib Figure containing the heatmap.
    """
    if not horizon_data:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(
            0.5, 0.5, "No horizon data available",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="gray",
        )
        ax.set_title(f"Predictive horizon: {metric_name}")
        return fig

    # Extract unique r and w values
    r_values = sorted(set(r for r, w in horizon_data.keys()))
    w_values = sorted(set(w for r, w in horizon_data.keys()))

    # Build 2D matrix (rows=r, cols=w), NaN for missing
    matrix = np.full((len(r_values), len(w_values)), np.nan)
    for (r, w), val in horizon_data.items():
        ri = r_values.index(r)
        wi = w_values.index(w)
        matrix[ri, wi] = val

    # Create mask for NaN cells
    mask = np.isnan(matrix)

    fig, ax = plt.subplots(figsize=(max(6, len(w_values) * 1.5), max(5, len(r_values) * 1.2)))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        mask=mask,
        xticklabels=[str(w) for w in w_values],
        yticklabels=[str(r) for r in r_values],
        cbar_kws={"label": "Predictive horizon (steps)"},
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Context window w")
    ax.set_ylabel("Jump length r")
    ax.set_title(f"Predictive horizon: {metric_name} (AUROC > {threshold})")

    fig.tight_layout()
    return fig


def render_horizon_heatmap(
    sweep_results_dir: str,
    metric_name: str = "Best primary metric",
    threshold: float = 0.75,
) -> plt.Figure | None:
    """Render heatmap from multiple experiment results in a sweep directory.

    Scans sweep_results_dir for subdirectories containing result.json,
    extracts (r, w) from config and horizon from metrics.

    Args:
        sweep_results_dir: Path to directory containing experiment subdirs.
        metric_name: Metric name for the title.
        threshold: AUROC threshold used for horizon calculation.

    Returns:
        Figure if data found, None otherwise.
    """
    import json
    from pathlib import Path

    sweep_dir = Path(sweep_results_dir)
    horizon_data: dict[tuple[int, int], float] = {}

    for result_path in sweep_dir.glob("*/result.json"):
        try:
            with open(result_path) as f:
                result = json.load(f)

            config = result.get("config", {})
            training_cfg = config.get("training", {})
            r_val = training_cfg.get("r")
            w_val = training_cfg.get("w")

            if r_val is None or w_val is None:
                continue

            # Try to find horizon in metrics
            metrics = result.get("metrics", {})
            pred_horizon = metrics.get("predictive_horizon", {})

            # Look for best horizon across primary metrics
            best_horizon = 0
            for r_key, r_data in pred_horizon.get("by_r_value", {}).items():
                for metric_key, metric_data in r_data.get("by_metric", {}).items():
                    h = metric_data.get("horizon", 0)
                    if h > best_horizon:
                        best_horizon = h

            horizon_data[(int(r_val), int(w_val))] = float(best_horizon)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    if not horizon_data:
        return None

    return plot_horizon_heatmap(horizon_data, metric_name=metric_name, threshold=threshold)
