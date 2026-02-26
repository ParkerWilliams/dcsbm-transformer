"""Statistical controls for predictive horizon analysis.

Implements Holm-Bonferroni multiple comparison correction, BCa bootstrap
confidence intervals on AUROC, Cohen's d effect sizes, correlation/redundancy
analysis, metric importance ranking, and headline QK^T vs AVWo comparison.

These statistical controls ensure the predictive horizon findings are rigorous
and publishable. Without correction for multiple comparisons, bootstrap CIs,
and redundancy analysis, the raw AUROC results could overstate significance
or miss that correlated metrics are measuring the same underlying signal.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import bootstrap, rankdata

from src.analysis.auroc_horizon import (
    PRIMARY_METRICS,
    _classify_event_count,
    _is_primary_metric,
    auroc_from_groups,
    compute_auroc_curve,
    compute_predictive_horizon,
    run_auroc_analysis,
)
from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
    stratify_by_r,
)
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


def holm_bonferroni(
    p_values: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Step-down procedure: sort p-values ascending, multiply by (m - rank + 1),
    enforce monotonicity, clip at 1.0, map back to original order.

    Args:
        p_values: Array of p-values to correct.
        alpha: Family-wise error rate.

    Returns:
        Tuple of (adjusted_p_values, reject_flags as bool array).
    """
    m = len(p_values)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # Multiply by (m - i) where i is the 0-based rank in sorted order
    adjusted = np.zeros(m)
    for i in range(m):
        adjusted[i] = sorted_p[i] * (m - i)

    # Enforce monotonicity (step-down: each adjusted p must be >= previous)
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    # Clip at 1.0
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result = np.zeros(m)
    result[sorted_idx] = adjusted

    reject = result <= alpha
    return result, reject


def auroc_with_bootstrap_ci(
    violation_vals: np.ndarray,
    control_vals: np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng: int | np.random.Generator = 42,
) -> tuple[float, float, float]:
    """Compute AUROC with BCa bootstrap confidence interval.

    Uses scipy.stats.bootstrap for BCa CIs. Falls back to percentile method
    if BCa produces NaN or raises an exception.

    Args:
        violation_vals: Metric values for violation events.
        control_vals: Metric values for control events.
        n_resamples: Number of bootstrap resamples.
        confidence_level: Confidence level (e.g. 0.95 for 95% CI).
        rng: Random seed or generator for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_low, ci_high).
    """
    point_estimate = auroc_from_groups(violation_vals, control_vals)

    if not np.isfinite(point_estimate):
        return float(point_estimate), np.nan, np.nan

    def auroc_stat(viol, ctrl, axis):
        """Vectorized AUROC statistic for scipy.stats.bootstrap."""
        n_v = viol.shape[axis]
        n_c = ctrl.shape[axis]
        combined = np.concatenate([viol, ctrl], axis=axis)
        ranks = np.apply_along_axis(rankdata, axis, combined)
        if axis == 0:
            viol_ranks = ranks[:n_v]
        else:
            viol_ranks = ranks[..., :n_v]
        rank_sum = viol_ranks.sum(axis=axis)
        return (rank_sum - n_v * (n_v + 1) / 2) / (n_v * n_c)

    # Try BCa first
    ci_low, ci_high = np.nan, np.nan
    for method in ("BCa", "percentile"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = bootstrap(
                    (violation_vals, control_vals),
                    auroc_stat,
                    n_resamples=n_resamples,
                    method=method,
                    confidence_level=confidence_level,
                    rng=rng,
                    vectorized=True,
                )
            ci_low = float(res.confidence_interval.low)
            ci_high = float(res.confidence_interval.high)
            if np.isfinite(ci_low) and np.isfinite(ci_high):
                break
        except Exception:
            continue

    return float(point_estimate), ci_low, ci_high


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size with pooled standard deviation.

    Positive d means group1 has higher values than group2.

    Args:
        group1: Values for group 1.
        group2: Values for group 2.

    Returns:
        Cohen's d, or NaN for insufficient samples or zero pooled_std.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return np.nan
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_cohens_d_by_lookback(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,
    r_value: int,
) -> np.ndarray:
    """Compute Cohen's d at each lookback distance j=1..r.

    For each lookback j, extracts violation and control metric values at
    step (resolution_step - j), then computes Cohen's d.

    Args:
        violation_events: Events with VIOLATED outcome.
        control_events: Events with FOLLOWED outcome.
        metric_array: Metric values, shape [n_sequences, max_steps-1].
        r_value: The r value for this group of events.

    Returns:
        Array of shape (r_value,) with Cohen's d at each lookback.
        NaN where insufficient valid data.
    """
    d_array = np.full(r_value, np.nan)
    n_steps = metric_array.shape[1]

    for j in range(1, r_value + 1):
        viol_vals = []
        ctrl_vals = []

        for ev in violation_events:
            idx = ev.resolution_step - j
            if 0 <= idx < n_steps:
                val = metric_array[ev.walk_idx, idx]
                if np.isfinite(val):
                    viol_vals.append(val)

        for ev in control_events:
            idx = ev.resolution_step - j
            if 0 <= idx < n_steps:
                val = metric_array[ev.walk_idx, idx]
                if np.isfinite(val):
                    ctrl_vals.append(val)

        if len(viol_vals) >= 2 and len(ctrl_vals) >= 2:
            d_array[j - 1] = cohens_d(
                np.array(viol_vals), np.array(ctrl_vals)
            )

    return d_array


def compute_correlation_matrix(
    metric_arrays: dict[str, np.ndarray],
    events: list[AnalysisEvent],
    mode: str,
) -> dict:
    """Compute correlation matrix across metrics.

    Two modes:
    - "measurement": Pool raw metric values at event positions, compute
      Pearson correlation. Measures measurement redundancy.
    - "predictive": Compute AUROC curves first, then correlate across
      lookback distances. Measures predictive redundancy.

    Args:
        metric_arrays: Dict mapping metric key -> array [n_sequences, max_steps-1].
        events: List of AnalysisEvent records.
        mode: One of "measurement" or "predictive".

    Returns:
        Dict with metric_names, matrix (2D list), redundant_pairs (|r|>0.9).
    """
    metric_names = sorted(metric_arrays.keys())
    n_metrics = len(metric_names)

    if n_metrics < 2:
        return {
            "metric_names": metric_names,
            "matrix": [[1.0]] if n_metrics == 1 else [],
            "redundant_pairs": [],
        }

    if mode == "measurement":
        # Pool raw metric values at event resolution positions
        vectors = []
        for name in metric_names:
            arr = metric_arrays[name]
            n_steps = arr.shape[1]
            vals = []
            for ev in events:
                # Use the resolution step - 1 as a representative position
                idx = ev.resolution_step - 1
                if 0 <= idx < n_steps:
                    val = arr[ev.walk_idx, idx]
                    if np.isfinite(val):
                        vals.append(val)
            vectors.append(np.array(vals) if vals else np.array([]))

        # Ensure all vectors have the same length for correlation
        # Use the minimum length across all metrics
        min_len = min(len(v) for v in vectors)
        if min_len < 3:
            matrix = np.eye(n_metrics).tolist()
            return {
                "metric_names": metric_names,
                "matrix": matrix,
                "redundant_pairs": [],
            }
        vectors = [v[:min_len] for v in vectors]
        data_matrix = np.array(vectors)
        corr_matrix = np.corrcoef(data_matrix)

    elif mode == "predictive":
        # Compute AUROC curves per metric, then correlate
        # Separate violations and controls
        violations = [e for e in events if e.outcome == RuleOutcome.VIOLATED]
        controls = [e for e in events if e.outcome == RuleOutcome.FOLLOWED]

        if len(violations) < 2 or len(controls) < 2:
            matrix = np.eye(n_metrics).tolist()
            return {
                "metric_names": metric_names,
                "matrix": matrix,
                "redundant_pairs": [],
            }

        # Determine r_value from events
        r_values = set(e.r_value for e in events)
        r_value = max(r_values) if r_values else 1

        auroc_curves = []
        for name in metric_names:
            arr = metric_arrays[name]
            viol_for_r = [e for e in violations if e.r_value == r_value]
            ctrl_for_r = [e for e in controls if e.r_value == r_value]
            curve = compute_auroc_curve(viol_for_r, ctrl_for_r, arr, r_value)
            # Replace NaN with 0.5 for correlation computation
            curve_clean = np.where(np.isfinite(curve), curve, 0.5)
            auroc_curves.append(curve_clean)

        data_matrix = np.array(auroc_curves)
        if data_matrix.shape[1] < 2:
            corr_matrix = np.eye(n_metrics)
        else:
            corr_matrix = np.corrcoef(data_matrix)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'measurement' or 'predictive'.")

    # Ensure matrix is well-formed
    if corr_matrix.ndim == 0:
        corr_matrix = np.array([[float(corr_matrix)]])
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Flag redundant pairs (|r| > 0.9)
    redundant_pairs = []
    for i in range(n_metrics):
        for j in range(i + 1, n_metrics):
            r_val = abs(corr_matrix[i, j])
            if r_val > 0.9:
                redundant_pairs.append(
                    (metric_names[i], metric_names[j], round(float(r_val), 4))
                )

    return {
        "metric_names": metric_names,
        "matrix": corr_matrix.tolist(),
        "redundant_pairs": redundant_pairs,
    }


def compute_metric_ranking(
    auroc_results_by_metric: dict[str, dict],
    primary_metric_names: list[str],
    redundant_pairs: list,
) -> dict:
    """Rank metrics by max AUROC descending with redundancy annotations.

    Args:
        auroc_results_by_metric: Dict mapping metric name -> dict with
            at least "max_auroc" and "horizon" keys.
        primary_metric_names: List of primary metric name strings.
        redundant_pairs: List of (metric_a, metric_b, correlation) tuples.

    Returns:
        Dict with "primary" (ranking of primary metrics only) and
        "all" (ranking of all metrics) lists.
    """
    # Build redundancy lookup
    redundancy_map: dict[str, list[str]] = {}
    for pair in redundant_pairs:
        a, b = pair[0], pair[1]
        redundancy_map.setdefault(a, []).append(b)
        redundancy_map.setdefault(b, []).append(a)

    primary_set = set(primary_metric_names)

    # Build ranking entries
    all_entries = []
    for metric_name, metric_data in auroc_results_by_metric.items():
        max_auroc = metric_data.get("max_auroc", 0.0)
        if not np.isfinite(max_auroc):
            max_auroc = 0.0
        entry = {
            "metric": metric_name,
            "max_auroc": max_auroc,
            "horizon": metric_data.get("horizon", 0),
            "is_primary": metric_name in primary_set,
            "redundant_with": redundancy_map.get(metric_name, []),
        }
        all_entries.append(entry)

    # Sort by max_auroc descending
    all_entries.sort(key=lambda x: x["max_auroc"], reverse=True)

    # Separate primary ranking
    primary_entries = [e for e in all_entries if e["is_primary"]]

    return {
        "primary": primary_entries,
        "all": all_entries,
    }


def compute_headline_comparison(
    auroc_results: dict,
    primary_metrics: list[str],
) -> dict:
    """Compare QK^T vs AVWo predictive horizons per r value.

    QK^T metrics = those starting with "qkt.".
    AVWo metrics = those starting with "avwo.".
    For each r, takes the max horizon across primary QK^T and primary AVWo metrics.

    Args:
        auroc_results: Dict with "by_r_value" key mapping r -> {by_metric: ...}.
        primary_metrics: List of primary metric name patterns
            (e.g. "qkt.grassmannian_distance").

    Returns:
        Dict with "description" and "by_r_value" mapping r -> comparison result.
    """
    primary_set = set(primary_metrics)

    by_r_value = {}
    for r_val, r_data in auroc_results.get("by_r_value", {}).items():
        by_metric = r_data.get("by_metric", {})

        qkt_horizons = []
        avwo_horizons = []

        for metric_key, metric_data in by_metric.items():
            # Check if this metric is primary by matching pattern
            parts = metric_key.split(".")
            if len(parts) == 3:
                target_metric = f"{parts[0]}.{parts[2]}"
            else:
                target_metric = metric_key

            is_primary = target_metric in primary_set or metric_data.get("is_primary", False)
            if not is_primary:
                continue

            horizon = metric_data.get("horizon", 0)

            if metric_key.startswith("qkt."):
                qkt_horizons.append(horizon)
            elif metric_key.startswith("avwo."):
                avwo_horizons.append(horizon)

        qkt_max = max(qkt_horizons) if qkt_horizons else 0
        avwo_max = max(avwo_horizons) if avwo_horizons else 0

        by_r_value[r_val] = {
            "qkt_max_horizon": qkt_max,
            "avwo_max_horizon": avwo_max,
            "qkt_leads": qkt_max > avwo_max,
            "gap": qkt_max - avwo_max,
        }

    return {
        "description": "QK^T vs AVWo predictive horizon comparison (descriptive)",
        "by_r_value": by_r_value,
    }


def apply_statistical_controls(
    auroc_results: dict | None,
    eval_data: dict,
    jumper_map: dict[int, JumperInfo],
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    bootstrap_rng: int = 42,
) -> dict:
    """Top-level orchestrator for all statistical controls.

    Takes the output from run_auroc_analysis (Plan 07-01) or computes it,
    then enriches it with:
    1. BCa bootstrap CIs on each AUROC value (for strata with 10+ events)
    2. Holm-Bonferroni correction on the 5 primary metrics' p-values
    3. Cohen's d at each lookback
    4. Both correlation matrices (measurement + predictive)
    5. Metric importance ranking per layer
    6. Headline QK^T vs AVWo comparison

    Args:
        auroc_results: Output from run_auroc_analysis, or None to compute.
        eval_data: Dict with generated, rule_outcome, failure_index, etc.
        jumper_map: Mapping from vertex_id to JumperInfo.
        n_bootstrap: Number of bootstrap resamples.
        confidence_level: Bootstrap confidence level.
        bootstrap_rng: Random seed for bootstrap reproducibility.

    Returns:
        Enriched dict ready for result.json.
    """
    # Determine metric keys from eval_data
    non_metric_keys = {
        "generated", "rule_outcome", "failure_index", "sequence_lengths",
        "edge_valid", "walk_id",
    }
    metric_keys = [k for k in eval_data.keys() if k not in non_metric_keys]

    # Compute AUROC results if not provided
    if auroc_results is None:
        auroc_results = run_auroc_analysis(
            eval_result_data=eval_data,
            jumper_map=jumper_map,
            metric_keys=metric_keys,
            n_shuffle=100,  # Use fewer shuffles when called from controls
        )

    # Extract events for statistical controls
    generated = eval_data["generated"]
    rule_outcome = eval_data["rule_outcome"]
    failure_index = eval_data["failure_index"]

    all_events = extract_events(generated, rule_outcome, failure_index, jumper_map)
    filtered_events, _ = filter_contaminated_events(all_events)
    by_r = stratify_by_r(filtered_events)

    # Prepare metric arrays
    metric_arrays = {k: eval_data[k] for k in metric_keys}

    # ----- 1. Bootstrap CIs and Cohen's d per r-value, per metric -----
    primary_p_values = {}  # metric_key -> p_value for Holm-Bonferroni

    for r_val in sorted(auroc_results.get("by_r_value", {}).keys()):
        r_data = auroc_results["by_r_value"][r_val]
        r_events = by_r.get(r_val, [])

        violations = [e for e in r_events if e.outcome == RuleOutcome.VIOLATED]
        controls = [e for e in r_events if e.outcome == RuleOutcome.FOLLOWED]

        tier = r_data.get("event_tier", _classify_event_count(len(violations), len(controls)))

        for metric_key in list(r_data.get("by_metric", {}).keys()):
            metric_data = r_data["by_metric"][metric_key]

            if metric_key not in eval_data:
                continue

            metric_array = eval_data[metric_key]

            # Determine event tier for bootstrap
            metric_data["event_tier_for_bootstrap"] = tier

            # Bootstrap CIs (only for full tier: 10+ events per class)
            if tier == "full":
                # Compute bootstrap CI on the max AUROC lookback
                max_lookback = metric_data.get("max_auroc_lookback", 1)
                n_steps = metric_array.shape[1]

                # Extract values at the max AUROC lookback distance
                viol_vals = []
                ctrl_vals = []
                for ev in violations:
                    idx = ev.resolution_step - max_lookback
                    if 0 <= idx < n_steps:
                        val = metric_array[ev.walk_idx, idx]
                        if np.isfinite(val):
                            viol_vals.append(float(val))
                for ev in controls:
                    idx = ev.resolution_step - max_lookback
                    if 0 <= idx < n_steps:
                        val = metric_array[ev.walk_idx, idx]
                        if np.isfinite(val):
                            ctrl_vals.append(float(val))

                if len(viol_vals) >= 3 and len(ctrl_vals) >= 3:
                    _, ci_low, ci_high = auroc_with_bootstrap_ci(
                        np.array(viol_vals),
                        np.array(ctrl_vals),
                        n_resamples=n_bootstrap,
                        confidence_level=confidence_level,
                        rng=bootstrap_rng,
                    )
                    metric_data["bootstrap_ci"] = [
                        round(float(ci_low), 6),
                        round(float(ci_high), 6),
                    ]

                    # For primary metrics, compute p-value (AUROC > 0.5)
                    # Approximate: if CI lower bound > 0.5, p < alpha
                    if _is_primary_metric(metric_key):
                        # Use the shuffle p-value if available
                        p_val = metric_data.get("p_value")
                        if p_val is not None and np.isfinite(p_val):
                            primary_p_values[metric_key] = p_val
                else:
                    metric_data["bootstrap_ci"] = [np.nan, np.nan]

            # Cohen's d by lookback
            d_by_lookback = compute_cohens_d_by_lookback(
                violations, controls, metric_array, r_val
            )
            metric_data["cohens_d_by_lookback"] = [
                round(float(d), 6) if np.isfinite(d) else None
                for d in d_by_lookback
            ]

    # ----- 2. Holm-Bonferroni correction on primary metrics -----
    hb_result = {"metric_names": [], "adjusted_p_values": {}, "reject": {}}

    if primary_p_values:
        hb_names = sorted(primary_p_values.keys())
        hb_pvals = np.array([primary_p_values[n] for n in hb_names])
        adjusted, reject = holm_bonferroni(hb_pvals, alpha=0.05)

        hb_result = {
            "metric_names": hb_names,
            "adjusted_p_values": {
                name: round(float(adj), 6)
                for name, adj in zip(hb_names, adjusted)
            },
            "reject": {
                name: bool(rej)
                for name, rej in zip(hb_names, reject)
            },
        }

        # Annotate each primary metric result with adjusted p-value
        for i, name in enumerate(hb_names):
            for r_val_key, r_data in auroc_results["by_r_value"].items():
                if name in r_data.get("by_metric", {}):
                    r_data["by_metric"][name]["p_value_adjusted"] = round(
                        float(adjusted[i]), 6
                    )
                    r_data["by_metric"][name]["reject_corrected"] = bool(
                        reject[i]
                    )

    # ----- 3. Correlation matrices -----
    correlation_matrices = {}
    if filtered_events and metric_arrays:
        correlation_matrices["measurement_redundancy"] = compute_correlation_matrix(
            metric_arrays, filtered_events, mode="measurement"
        )
        correlation_matrices["predictive_redundancy"] = compute_correlation_matrix(
            metric_arrays, filtered_events, mode="predictive"
        )

    # ----- 4. Metric importance ranking per layer -----
    # Collect all metric results across r-values (use max AUROC across r-values)
    metric_best: dict[str, dict] = {}
    for r_val_key, r_data in auroc_results.get("by_r_value", {}).items():
        for metric_key, metric_data in r_data.get("by_metric", {}).items():
            max_auroc = metric_data.get("max_auroc", 0.0)
            if not np.isfinite(max_auroc):
                max_auroc = 0.0
            current = metric_best.get(metric_key, {"max_auroc": 0.0, "horizon": 0})
            if max_auroc > current["max_auroc"]:
                metric_best[metric_key] = {
                    "max_auroc": max_auroc,
                    "horizon": metric_data.get("horizon", 0),
                }

    # Get redundant pairs from measurement correlation
    redundant_pairs = correlation_matrices.get(
        "measurement_redundancy", {}
    ).get("redundant_pairs", [])

    primary_names_full = [
        k for k in metric_best.keys() if _is_primary_metric(k)
    ]

    metric_ranking = {}
    # Group by layer
    layer_metrics: dict[str, dict[str, dict]] = {}
    for metric_key, metric_data in metric_best.items():
        parts = metric_key.split(".")
        if len(parts) == 3:
            layer_key = parts[1]  # e.g. "layer_0"
        else:
            layer_key = "default"
        layer_metrics.setdefault(layer_key, {})[metric_key] = metric_data

    for layer_key, metrics in layer_metrics.items():
        layer_primary = [k for k in metrics.keys() if _is_primary_metric(k)]
        metric_ranking[layer_key] = compute_metric_ranking(
            metrics, layer_primary, redundant_pairs
        )

    # ----- 5. Headline comparison -----
    headline = compute_headline_comparison(
        auroc_results, list(PRIMARY_METRICS)
    )

    # ----- Assemble final result -----
    result = dict(auroc_results)
    result["holm_bonferroni"] = hb_result
    result["correlation_matrices"] = correlation_matrices
    result["metric_ranking"] = metric_ranking
    result["headline_comparison"] = headline

    return result
