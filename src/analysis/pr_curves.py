"""Precision-recall curves and AUPRC computation.

Computes precision-recall curves and AUPRC at each lookback distance j (1 to r)
for each SVD metric, using the same event extraction infrastructure as the
existing AUROC pipeline. Mirrors the structure of auroc_horizon.py.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.analysis.auroc_horizon import auroc_from_groups
from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
    stratify_by_r,
)
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


def _gather_values_at_lookback(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather metric values at a specific lookback distance for violations and controls.

    For each event, extracts the metric value at position (resolution_step - lookback).
    Filters out NaN values.

    Args:
        violation_events: Events with VIOLATED outcome.
        control_events: Events with FOLLOWED outcome.
        metric_array: Metric values, shape [n_sequences, max_steps-1].
        lookback: Lookback distance j.

    Returns:
        Tuple of (violation_values, control_values) as numpy arrays.
    """
    n_steps = metric_array.shape[1]
    viol_vals: list[float] = []
    ctrl_vals: list[float] = []

    for ev in violation_events:
        idx = ev.resolution_step - lookback
        if 0 <= idx < n_steps:
            val = metric_array[ev.walk_idx, idx]
            if np.isfinite(val):
                viol_vals.append(float(val))

    for ev in control_events:
        idx = ev.resolution_step - lookback
        if 0 <= idx < n_steps:
            val = metric_array[ev.walk_idx, idx]
            if np.isfinite(val):
                ctrl_vals.append(float(val))

    return np.array(viol_vals), np.array(ctrl_vals)


def compute_pr_at_lookback(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,
    r_value: int,
    lookback: int,
    min_per_class: int = 2,
) -> dict:
    """Compute precision-recall curve metrics at a specific lookback distance.

    Determines score direction from AUROC (higher score = more like violation),
    then computes AUPRC using sklearn.

    Args:
        violation_events: Events with VIOLATED outcome.
        control_events: Events with FOLLOWED outcome.
        metric_array: Metric values, shape [n_sequences, max_steps-1].
        r_value: The r value for this group of events.
        lookback: Lookback distance j.
        min_per_class: Minimum events per class to compute PR curve.

    Returns:
        Dict with auprc, prevalence, n_violations, n_controls.
        NaN auprc if insufficient data.
    """
    viol_vals, ctrl_vals = _gather_values_at_lookback(
        violation_events, control_events, metric_array, lookback
    )

    n_v = len(viol_vals)
    n_c = len(ctrl_vals)

    if n_v < min_per_class or n_c < min_per_class:
        return {
            "auprc": float("nan"),
            "prevalence": float("nan"),
            "n_violations": n_v,
            "n_controls": n_c,
        }

    # Determine score direction using AUROC
    auroc = auroc_from_groups(viol_vals, ctrl_vals)

    # Build labels and scores
    labels = np.concatenate([np.ones(n_v), np.zeros(n_c)])
    scores = np.concatenate([viol_vals, ctrl_vals])

    # If AUROC < 0.5, higher metric values predict controls, not violations.
    # Negate scores so higher always means "more like violation".
    if np.isfinite(auroc) and auroc < 0.5:
        scores = -scores

    auprc = average_precision_score(labels, scores)
    prevalence = n_v / (n_v + n_c)

    return {
        "auprc": float(auprc),
        "prevalence": float(prevalence),
        "n_violations": n_v,
        "n_controls": n_c,
    }


def run_pr_analysis(
    eval_result_data: dict,
    jumper_map: dict[int, JumperInfo],
    metric_keys: list[str],
    min_events_per_class: int = 5,
) -> dict:
    """Orchestrate the full PR curve analysis pipeline.

    Mirrors run_auroc_analysis from auroc_horizon.py:
    1. Extract events from generated sequences
    2. Apply contamination filter
    3. Stratify by r value
    4. For each r value, for each metric, compute AUPRC at each lookback j=1..r

    Args:
        eval_result_data: Dict with keys: 'generated', 'rule_outcome',
            'failure_index', 'sequence_lengths', and metric arrays.
        jumper_map: Mapping from vertex_id to JumperInfo.
        metric_keys: List of metric key strings to analyze.
        min_events_per_class: Minimum events for analysis.

    Returns:
        Nested dict matching result.json pr_curves schema:
        - config: analysis parameters
        - by_r_value: per-r results with AUPRC per lookback per metric
    """
    # Step 1: Extract events
    generated = eval_result_data["generated"]
    rule_outcome = eval_result_data["rule_outcome"]
    failure_index = eval_result_data["failure_index"]

    all_events = extract_events(generated, rule_outcome, failure_index, jumper_map)

    # Step 2: Apply contamination filter
    filtered_events, _ = filter_contaminated_events(all_events)

    # Step 3: Stratify by r value
    by_r = stratify_by_r(filtered_events)

    # Build config block
    config = {
        "min_events_per_class": min_events_per_class,
    }

    # Step 4: Analyze each r value
    by_r_value_results: dict[int, dict] = {}

    for r_val in sorted(by_r.keys()):
        r_events = by_r[r_val]

        # Separate violations and controls
        violations = [e for e in r_events if e.outcome == RuleOutcome.VIOLATED]
        controls = [e for e in r_events if e.outcome == RuleOutcome.FOLLOWED]

        n_violations = len(violations)
        n_controls = len(controls)

        r_result: dict = {
            "n_violations": n_violations,
            "n_controls": n_controls,
            "by_metric": {},
        }

        if min(n_violations, n_controls) <= 1:
            by_r_value_results[r_val] = r_result
            continue

        # Analyze each metric
        for metric_key in metric_keys:
            if metric_key not in eval_result_data:
                continue

            metric_array = eval_result_data[metric_key]

            # Compute AUPRC at each lookback j=1..r
            auprc_by_lookback: list[float] = []
            prevalence = float("nan")

            for j in range(1, r_val + 1):
                pr_result = compute_pr_at_lookback(
                    violations, controls, metric_array, r_val, j,
                    min_per_class=min(min_events_per_class, 2),
                )
                auprc_by_lookback.append(pr_result["auprc"])
                if np.isfinite(pr_result["prevalence"]):
                    prevalence = pr_result["prevalence"]

            r_result["by_metric"][metric_key] = {
                "auprc_by_lookback": auprc_by_lookback,
                "prevalence": float(prevalence),
            }

        by_r_value_results[r_val] = r_result

    return {
        "config": config,
        "by_r_value": by_r_value_results,
    }
