"""Signal concentration analysis for multi-head ablation (Phase 16: MHAD-03, MHAD-04).

Measures whether the predictive SVD signal concentrates in specific attention
heads or distributes uniformly across all heads. Uses entropy and Gini
coefficient of per-head AUROC distributions as concentration metrics.

These are descriptive statistics, not hypothesis tests (see PITFALLS.md #13).
Report per-head AUROC with confidence intervals rather than p-values.
"""

import numpy as np
from typing import Any


def compute_auroc_entropy(per_head_aurocs: np.ndarray) -> float:
    """Compute normalized entropy of AUROC distribution across heads.

    High entropy means signal is distributed uniformly across heads.
    Low entropy means signal is concentrated in specific heads.

    Args:
        per_head_aurocs: Array of shape [n_heads] with max AUROC per head.
            Values should be in [0, 1]. NaN values are treated as 0.5 (chance).

    Returns:
        Normalized entropy in [0, 1]. 0 = signal concentrated in one head,
        1 = signal uniformly distributed across all heads.
        Returns NaN if n_heads < 2.
    """
    aurocs = np.array(per_head_aurocs, dtype=np.float64)
    n_heads = len(aurocs)
    if n_heads < 2:
        return float("nan")

    # Replace NaN with 0.5 (chance level -- no information)
    aurocs = np.where(np.isnan(aurocs), 0.5, aurocs)

    # Clamp to avoid log(0)
    aurocs = np.maximum(aurocs, 1e-10)

    # Normalize to probability distribution
    total = aurocs.sum()
    if total < 1e-10:
        return 1.0  # All effectively zero -> uniform (degenerate)
    p = aurocs / total

    # Shannon entropy
    H = -np.sum(p * np.log2(p))

    # Normalize by max entropy (log2(n_heads)) so result is in [0, 1]
    H_max = np.log2(n_heads)
    return float(H / H_max) if H_max > 0 else float("nan")


def compute_gini_coefficient(per_head_aurocs: np.ndarray) -> float:
    """Compute Gini coefficient of AUROC distribution across heads.

    Gini = 0 means perfectly equal distribution (all heads equally strong).
    Gini = 1 means maximum inequality (one head has all signal).

    Args:
        per_head_aurocs: Array of shape [n_heads] with max AUROC per head.
            NaN values are treated as 0.5 (chance).

    Returns:
        Gini coefficient in [0, 1].
        Returns NaN if n_heads < 2.
    """
    aurocs = np.array(per_head_aurocs, dtype=np.float64)
    n_heads = len(aurocs)
    if n_heads < 2:
        return float("nan")

    aurocs = np.where(np.isnan(aurocs), 0.5, aurocs)

    # Sort ascending
    sorted_aurocs = np.sort(aurocs)
    n = len(sorted_aurocs)
    mean_auroc = sorted_aurocs.mean()

    if mean_auroc < 1e-10:
        return 0.0  # All zero -> equal (degenerate)

    # Standard Gini formula using sorted values
    index = np.arange(1, n + 1)
    gini = (
        (2 * np.sum(index * sorted_aurocs) - (n + 1) * np.sum(sorted_aurocs))
        / (n * np.sum(sorted_aurocs))
    )
    return float(np.clip(gini, 0.0, 1.0))


def compute_signal_concentration(
    per_head_max_aurocs: dict[int, float],
    metric_name: str = "",
) -> dict[str, Any]:
    """Compute signal concentration metrics for a set of per-head AUROC values.

    Args:
        per_head_max_aurocs: Mapping of head_idx -> max AUROC value for that head.
        metric_name: Optional name of the SVD metric (for labeling in reports).

    Returns:
        Dict with keys:
            - metric_name: str
            - n_heads: int
            - per_head_aurocs: dict[int, float]
            - entropy: float (normalized, 0=concentrated, 1=distributed)
            - gini: float (0=equal, 1=unequal)
            - max_to_mean_ratio: float
            - dominant_head: int (head index with highest AUROC)
            - dominant_auroc: float
            - interpretation: str (human-readable summary)
    """
    n_heads = len(per_head_max_aurocs)
    sorted_heads = sorted(per_head_max_aurocs.keys())
    auroc_values = np.array([per_head_max_aurocs[h] for h in sorted_heads])

    entropy = compute_auroc_entropy(auroc_values)
    gini = compute_gini_coefficient(auroc_values)

    # Max-to-mean ratio
    valid = auroc_values[~np.isnan(auroc_values)]
    if len(valid) > 0 and valid.mean() > 1e-10:
        max_to_mean = float(valid.max() / valid.mean())
    else:
        max_to_mean = float("nan")

    # Dominant head
    if len(valid) > 0:
        dominant_idx = sorted_heads[int(np.nanargmax(auroc_values))]
        dominant_auroc = float(np.nanmax(auroc_values))
    else:
        dominant_idx = 0
        dominant_auroc = float("nan")

    # Human-readable interpretation
    if n_heads < 2:
        interp = "Single-head: concentration analysis not applicable"
    elif entropy < 0.5:
        interp = (
            f"Signal CONCENTRATED in head {dominant_idx} "
            f"(entropy={entropy:.3f}, Gini={gini:.3f})"
        )
    elif entropy > 0.9:
        interp = (
            f"Signal DISTRIBUTED across all heads "
            f"(entropy={entropy:.3f}, Gini={gini:.3f})"
        )
    else:
        interp = (
            f"Signal PARTIALLY concentrated "
            f"(entropy={entropy:.3f}, Gini={gini:.3f})"
        )

    return {
        "metric_name": metric_name,
        "n_heads": n_heads,
        "per_head_aurocs": dict(per_head_max_aurocs),
        "entropy": entropy,
        "gini": gini,
        "max_to_mean_ratio": max_to_mean,
        "dominant_head": dominant_idx,
        "dominant_auroc": dominant_auroc,
        "interpretation": interp,
    }


def compute_ablation_comparison(
    results_by_n_heads: dict[int, dict[str, float]],
) -> dict[str, Any]:
    """Compare signal strength across ablation configs (1h/2h/4h).

    Takes per-config best AUROC values and produces a comparison summary.

    Args:
        results_by_n_heads: Mapping of n_heads -> {metric_key: max_auroc}.
            Each entry represents one ablation config's best AUROC per metric.

    Returns:
        Dict with ablation comparison summary:
            - configs: list of n_heads values tested
            - per_config_best_auroc: {n_heads: {metric: max_auroc}}
            - aggregate_comparison: {metric: {n_heads: auroc}}
            - conclusion: str (human-readable summary)
    """
    configs = sorted(results_by_n_heads.keys())

    # Collect all metric names across configs
    all_metrics: set[str] = set()
    for n_heads_val in configs:
        all_metrics.update(results_by_n_heads[n_heads_val].keys())

    # Per-metric comparison across configs
    aggregate: dict[str, dict[int, float]] = {}
    for metric in sorted(all_metrics):
        aggregate[metric] = {}
        for n_heads_val in configs:
            aggregate[metric][n_heads_val] = results_by_n_heads[n_heads_val].get(
                metric, float("nan")
            )

    # Generate conclusions comparing multi-head to single-head
    conclusions: list[str] = []
    if 1 in configs:
        for metric in sorted(all_metrics):
            single_auroc = aggregate[metric].get(1, float("nan"))
            for n_heads_val in configs:
                if n_heads_val == 1:
                    continue
                multi_auroc = aggregate[metric].get(n_heads_val, float("nan"))
                if not np.isnan(single_auroc) and not np.isnan(multi_auroc):
                    if multi_auroc > single_auroc + 0.02:
                        conclusions.append(
                            f"{metric}: {n_heads_val}h improves over 1h "
                            f"({multi_auroc:.3f} vs {single_auroc:.3f})"
                        )
                    elif single_auroc > multi_auroc + 0.02:
                        conclusions.append(
                            f"{metric}: 1h outperforms {n_heads_val}h "
                            f"({single_auroc:.3f} vs {multi_auroc:.3f})"
                        )
                    else:
                        conclusions.append(
                            f"{metric}: comparable across 1h and {n_heads_val}h"
                        )

    return {
        "configs": configs,
        "per_config_best_auroc": dict(results_by_n_heads),
        "aggregate_comparison": aggregate,
        "conclusion": (
            "; ".join(conclusions) if conclusions else "Insufficient data for comparison"
        ),
    }
