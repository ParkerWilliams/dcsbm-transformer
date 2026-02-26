"""AUROC computation, predictive horizon, and shuffle controls.

Computes AUROC at each lookback distance j (1 to r) for each SVD metric,
determines predictive horizons, and validates with shuffle controls.
Implements the rank-based AUROC method equivalent to Mann-Whitney U statistic.
"""

from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
    stratify_by_r,
)
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo

# Pre-registered primary metrics (Holm-Bonferroni corrected).
# These are the 5 metrics from CONTEXT.md locked decisions.
# Metric keys in the NPZ follow: target.layer_N.metric_name
# These patterns match the metric_name part (without layer prefix).
PRIMARY_METRICS: frozenset[str] = frozenset({
    "qkt.grassmannian_distance",
    "qkt.spectral_gap_1_2",
    "qkt.spectral_entropy",
    "avwo.stable_rank",
    "avwo.grassmannian_distance",
})


def auroc_from_groups(violations: np.ndarray, controls: np.ndarray) -> float:
    """AUROC via rank-based method: P(X_viol > X_ctrl).

    Mathematically equivalent to sklearn.metrics.roc_auc_score and
    to U / (n1 * n0) from Mann-Whitney U test. Handles ties via midrank.

    Args:
        violations: Metric values for violation events.
        controls: Metric values for control events.

    Returns:
        AUROC value, or NaN if either group is empty.
    """
    n_v, n_c = len(violations), len(controls)
    if n_v == 0 or n_c == 0:
        return np.nan
    combined = np.concatenate([violations, controls])
    ranks = rankdata(combined)
    rank_sum = ranks[:n_v].sum()
    return (rank_sum - n_v * (n_v + 1) / 2) / (n_v * n_c)


def compute_auroc_curve(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,
    r_value: int,
    min_per_class: int = 2,
) -> np.ndarray:
    """Compute AUROC at each lookback distance j=1..r.

    For lookback j:
      - violation_values = metric_array[event.walk_idx, event.resolution_step - j]
        for each violation event (filtering NaN)
      - control_values = same for control events
      - AUROC via rank-based method

    Args:
        violation_events: Events with VIOLATED outcome.
        control_events: Events with FOLLOWED outcome.
        metric_array: Metric values, shape [n_sequences, max_steps-1].
        r_value: The r value for this group of events.
        min_per_class: Minimum events per class to compute AUROC.

    Returns:
        Array of shape (r_value,) with AUROC at each lookback j=1..r.
        NaN where insufficient valid data.
    """
    aurocs = np.full(r_value, np.nan)
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

        if len(viol_vals) >= min_per_class and len(ctrl_vals) >= min_per_class:
            aurocs[j - 1] = auroc_from_groups(
                np.array(viol_vals), np.array(ctrl_vals)
            )

    return aurocs


def compute_predictive_horizon(
    auroc_curve: np.ndarray,
    threshold: float = 0.75,
) -> int:
    """Find the furthest lookback j where AUROC exceeds threshold.

    Scans from largest j to smallest. Returns 0 if none exceed.

    Args:
        auroc_curve: AUROC values at each lookback j=1..r, shape (r,).
        threshold: AUROC threshold for predictive signal.

    Returns:
        Largest j (1-based) where auroc_curve[j-1] > threshold, or 0.
    """
    r = len(auroc_curve)
    for j in range(r, 0, -1):
        val = auroc_curve[j - 1]
        if np.isfinite(val) and val > threshold:
            return j
    return 0


def run_shuffle_control(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,
    r_value: int,
    n_permutations: int = 10_000,
    flag_threshold: float = 0.6,
    rng: int | np.random.Generator = 42,
) -> dict:
    """Shuffle violation/control labels and recompute max AUROC.

    Tests whether the observed AUROC signal is a positional artifact
    rather than a genuine class-label signal.

    Args:
        violation_events: Events with VIOLATED outcome.
        control_events: Events with FOLLOWED outcome.
        metric_array: Metric values, shape [n_sequences, max_steps-1].
        r_value: The r value for this group of events.
        n_permutations: Number of label shuffles.
        flag_threshold: Flag if p95 shuffled AUROC exceeds this.
        rng: Random seed or generator for reproducibility.

    Returns:
        Dict with shuffle_auroc_mean, shuffle_auroc_p95, shuffle_flag,
        p_value (fraction of shuffles with max AUROC >= observed).
    """
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # Compute observed max AUROC
    observed_curve = compute_auroc_curve(
        violation_events, control_events, metric_array, r_value
    )
    observed_max = np.nanmax(observed_curve) if np.any(np.isfinite(observed_curve)) else np.nan

    # Combine all events for shuffling
    all_events = list(violation_events) + list(control_events)
    n_viol = len(violation_events)
    n_total = len(all_events)

    shuffle_max_aurocs = np.zeros(n_permutations)

    for perm in range(n_permutations):
        # Shuffle indices
        perm_indices = rng.permutation(n_total)
        shuffled_viol = [all_events[i] for i in perm_indices[:n_viol]]
        shuffled_ctrl = [all_events[i] for i in perm_indices[n_viol:]]

        shuffled_curve = compute_auroc_curve(
            shuffled_viol, shuffled_ctrl, metric_array, r_value
        )
        max_val = np.nanmax(shuffled_curve) if np.any(np.isfinite(shuffled_curve)) else np.nan
        shuffle_max_aurocs[perm] = max_val

    valid_shuffles = shuffle_max_aurocs[np.isfinite(shuffle_max_aurocs)]
    shuffle_mean = float(np.mean(valid_shuffles)) if len(valid_shuffles) > 0 else np.nan
    shuffle_p95 = float(np.percentile(valid_shuffles, 95)) if len(valid_shuffles) > 0 else np.nan

    # p-value: fraction of shuffles with max AUROC >= observed
    if np.isfinite(observed_max) and len(valid_shuffles) > 0:
        p_value = float(np.mean(valid_shuffles >= observed_max))
    else:
        p_value = np.nan

    return {
        "shuffle_auroc_mean": shuffle_mean,
        "shuffle_auroc_p95": shuffle_p95,
        "shuffle_flag": bool(np.isfinite(shuffle_p95) and shuffle_p95 > flag_threshold),
        "p_value": p_value,
    }


def _is_primary_metric(metric_key: str) -> bool:
    """Check if a metric key matches a pre-registered primary metric.

    Metric keys follow the pattern: target.layer_N.metric_name
    Primary metrics are identified by target.metric_name (without layer).

    Args:
        metric_key: Full metric key like 'qkt.layer_0.grassmannian_distance'.

    Returns:
        True if this metric is in the primary set.
    """
    parts = metric_key.split(".")
    if len(parts) == 3:
        # target.layer_N.metric_name -> target.metric_name
        target_metric = f"{parts[0]}.{parts[2]}"
        return target_metric in PRIMARY_METRICS
    return False


def _classify_event_count(n_violations: int, n_controls: int) -> str:
    """Classify the event count tier per RESEARCH.md edge case handling.

    Returns:
        One of: 'skip', 'low_n', 'moderate_n', 'full'
    """
    min_n = min(n_violations, n_controls)
    if min_n <= 1:
        return "skip"
    elif min_n <= 4:
        return "low_n"
    elif min_n <= 9:
        return "moderate_n"
    else:
        return "full"


def run_auroc_analysis(
    eval_result_data: dict,
    jumper_map: dict[int, JumperInfo],
    metric_keys: list[str],
    horizon_threshold: float = 0.75,
    shuffle_flag_threshold: float = 0.6,
    n_shuffle: int = 10_000,
    min_events_per_class: int = 5,
) -> dict:
    """Orchestrate the full AUROC analysis pipeline.

    Steps:
    1. Extract events from generated sequences
    2. Apply contamination filter
    3. Stratify by r value
    4. For each r value, for each metric, compute AUROC curve and horizon
    5. Run shuffle controls for metrics with enough events

    Args:
        eval_result_data: Dict with keys: 'generated', 'rule_outcome',
            'failure_index', 'sequence_lengths', and metric arrays.
        jumper_map: Mapping from vertex_id to JumperInfo.
        metric_keys: List of metric key strings to analyze.
        horizon_threshold: AUROC threshold for predictive horizon.
        shuffle_flag_threshold: Flag threshold for shuffle controls.
        n_shuffle: Number of shuffle permutations.
        min_events_per_class: Minimum events for full analysis.

    Returns:
        Nested dict matching result.json predictive_horizon schema:
        - config: analysis parameters
        - contamination_audit: filtering statistics
        - by_r_value: per-r results with AUROC curves and horizons
    """
    # Step 1: Extract events
    generated = eval_result_data["generated"]
    rule_outcome = eval_result_data["rule_outcome"]
    failure_index = eval_result_data["failure_index"]

    all_events = extract_events(generated, rule_outcome, failure_index, jumper_map)

    # Step 2: Apply contamination filter
    filtered_events, contamination_audit = filter_contaminated_events(all_events)

    # Step 3: Stratify by r value
    by_r = stratify_by_r(filtered_events)

    # Build config block
    primary_list = sorted(PRIMARY_METRICS)
    config = {
        "horizon_threshold": horizon_threshold,
        "shuffle_flag_threshold": shuffle_flag_threshold,
        "n_shuffle": n_shuffle,
        "min_events_per_class": min_events_per_class,
        "primary_metrics": primary_list,
    }

    # Step 4 & 5: Analyze each r value
    by_r_value_results: dict[int, dict] = {}

    for r_val in sorted(by_r.keys()):
        r_events = by_r[r_val]

        # Separate violations and controls
        violations = [e for e in r_events if e.outcome == RuleOutcome.VIOLATED]
        controls = [e for e in r_events if e.outcome == RuleOutcome.FOLLOWED]

        n_violations = len(violations)
        n_controls = len(controls)
        tier = _classify_event_count(n_violations, n_controls)

        r_result: dict = {
            "n_violations": n_violations,
            "n_controls": n_controls,
            "event_tier": tier,
            "by_metric": {},
        }

        if tier == "skip":
            by_r_value_results[r_val] = r_result
            continue

        # Analyze each metric
        for metric_key in metric_keys:
            if metric_key not in eval_result_data:
                continue

            metric_array = eval_result_data[metric_key]
            is_primary = _is_primary_metric(metric_key)

            # Compute AUROC curve
            auroc_curve = compute_auroc_curve(
                violations, controls, metric_array, r_val
            )

            # Compute predictive horizon
            horizon = compute_predictive_horizon(auroc_curve, threshold=horizon_threshold)

            # Max AUROC and its lookback
            finite_mask = np.isfinite(auroc_curve)
            if np.any(finite_mask):
                max_auroc = float(np.nanmax(auroc_curve))
                max_auroc_lookback = int(np.nanargmax(auroc_curve)) + 1  # 1-based
            else:
                max_auroc = float("nan")
                max_auroc_lookback = 0

            # Count valid events per lookback
            n_valid_by_lookback = []
            n_steps = metric_array.shape[1]
            for j in range(1, r_val + 1):
                n_valid = 0
                for ev in violations + controls:
                    idx = ev.resolution_step - j
                    if 0 <= idx < n_steps:
                        val = metric_array[ev.walk_idx, idx]
                        if np.isfinite(val):
                            n_valid += 1
                n_valid_by_lookback.append(n_valid)

            metric_result: dict = {
                "auroc_by_lookback": auroc_curve.tolist(),
                "horizon": horizon,
                "max_auroc": max_auroc,
                "max_auroc_lookback": max_auroc_lookback,
                "n_valid_by_lookback": n_valid_by_lookback,
                "is_primary": is_primary,
            }

            # Shuffle controls (only for full tier or moderate_n with primary)
            if tier == "full" or (tier == "moderate_n" and is_primary):
                shuffle_result = run_shuffle_control(
                    violations, controls, metric_array, r_val,
                    n_permutations=n_shuffle,
                    flag_threshold=shuffle_flag_threshold,
                    rng=42,
                )
                metric_result["shuffle_auroc_mean"] = shuffle_result["shuffle_auroc_mean"]
                metric_result["shuffle_auroc_p95"] = shuffle_result["shuffle_auroc_p95"]
                metric_result["shuffle_flag"] = shuffle_result["shuffle_flag"]
                metric_result["p_value"] = shuffle_result["p_value"]
            else:
                metric_result["shuffle_flag"] = None
                metric_result["p_value"] = None

            r_result["by_metric"][metric_key] = metric_result

        by_r_value_results[r_val] = r_result

    return {
        "config": config,
        "contamination_audit": contamination_audit,
        "by_r_value": by_r_value_results,
    }
