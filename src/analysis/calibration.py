"""Calibration diagnostics: reliability diagrams and Expected Calibration Error (ECE).

Computes calibration metrics for violation prediction at each lookback distance
using the same event extraction infrastructure as AUROC and PR curves.
Pseudo-probabilities are derived via empirical CDF (rank-based conversion).
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve

from src.analysis.auroc_horizon import auroc_from_groups
from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
    stratify_by_r,
)
from src.analysis.pr_curves import _gather_values_at_lookback
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


def metric_to_pseudo_probability(scores: np.ndarray) -> np.ndarray:
    """Convert raw metric values to pseudo-probabilities using empirical CDF.

    Uses rank-based conversion: the value at percentile p is treated as
    predicting P(violation) = p. Non-parametric and doesn't overfit.

    Args:
        scores: Raw metric values, 1D array.

    Returns:
        Array of pseudo-probabilities in (0, 1], same shape as input.
    """
    if len(scores) == 0:
        return np.array([])
    return rankdata(scores) / len(scores)


def compute_ece(
    fraction_of_positives: np.ndarray,
    mean_predicted_value: np.ndarray,
    bin_counts: np.ndarray,
) -> float:
    """Compute Expected Calibration Error.

    ECE = sum_b (|B_b| / N) * |acc(b) - conf(b)|

    where B_b is the set of samples in bin b, N is total samples,
    acc(b) is fraction of positives in bin b, and conf(b) is mean
    predicted probability in bin b.

    Args:
        fraction_of_positives: Fraction of positive samples per bin.
        mean_predicted_value: Mean predicted probability per bin.
        bin_counts: Number of samples per bin.

    Returns:
        ECE value in [0, 1], or NaN if no valid bins.
    """
    total = bin_counts.sum()
    if total == 0:
        return float("nan")

    ece = 0.0
    for i in range(len(fraction_of_positives)):
        if bin_counts[i] > 0:
            weight = bin_counts[i] / total
            ece += weight * abs(fraction_of_positives[i] - mean_predicted_value[i])

    return float(ece)


def compute_calibration_at_lookback(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,
    r_value: int,
    lookback: int,
    n_bins: int = 10,
    min_per_class: int = 2,
) -> dict:
    """Compute calibration metrics at a specific lookback distance.

    Converts SVD metric values to pseudo-probabilities via empirical CDF,
    then computes calibration curve and ECE.

    Args:
        violation_events: Events with VIOLATED outcome.
        control_events: Events with FOLLOWED outcome.
        metric_array: Metric values, shape [n_sequences, max_steps-1].
        r_value: The r value for this group of events.
        lookback: Lookback distance j.
        n_bins: Number of equal-width bins for calibration.
        min_per_class: Minimum events per class.

    Returns:
        Dict with ece, fraction_of_positives, mean_predicted_value, bin_counts, n_bins.
    """
    viol_vals, ctrl_vals = _gather_values_at_lookback(
        violation_events, control_events, metric_array, lookback
    )

    n_v = len(viol_vals)
    n_c = len(ctrl_vals)

    if n_v < min_per_class or n_c < min_per_class:
        return {
            "ece": float("nan"),
            "fraction_of_positives": [],
            "mean_predicted_value": [],
            "bin_counts": [],
            "n_bins": n_bins,
        }

    # Determine score direction
    auroc = auroc_from_groups(viol_vals, ctrl_vals)
    labels = np.concatenate([np.ones(n_v), np.zeros(n_c)])
    scores = np.concatenate([viol_vals, ctrl_vals])

    # Negate if lower values predict violation
    if np.isfinite(auroc) and auroc < 0.5:
        scores = -scores

    # Convert to pseudo-probabilities
    prob_pred = metric_to_pseudo_probability(scores)

    # Compute calibration curve
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, prob_pred, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        # sklearn may raise if too few unique predicted values
        return {
            "ece": float("nan"),
            "fraction_of_positives": [],
            "mean_predicted_value": [],
            "bin_counts": [],
            "n_bins": n_bins,
        }

    # Compute bin counts
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_counts = np.histogram(prob_pred, bins=bin_edges)[0]

    # Map calibration_curve output bins to histogram bins
    # calibration_curve returns only non-empty bins, so we need to align
    # Compute ECE using the calibration_curve output directly
    # (fraction_of_positives and mean_predicted_value already correspond to non-empty bins)
    non_empty_mask = bin_counts > 0
    non_empty_counts = bin_counts[non_empty_mask]

    ece = compute_ece(fraction_of_positives, mean_predicted_value, non_empty_counts)

    return {
        "ece": float(ece),
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "bin_counts": bin_counts.tolist(),
        "n_bins": n_bins,
    }


def run_calibration_analysis(
    eval_result_data: dict,
    jumper_map: dict[int, JumperInfo],
    metric_keys: list[str],
    n_bins: int = 10,
    min_events_per_class: int = 5,
) -> dict:
    """Orchestrate the full calibration analysis pipeline.

    Mirrors run_pr_analysis and run_auroc_analysis structure:
    1. Extract events from generated sequences
    2. Apply contamination filter
    3. Stratify by r value
    4. For each r value, for each metric, compute ECE at each lookback j=1..r

    Args:
        eval_result_data: Dict with keys: 'generated', 'rule_outcome',
            'failure_index', 'sequence_lengths', and metric arrays.
        jumper_map: Mapping from vertex_id to JumperInfo.
        metric_keys: List of metric key strings to analyze.
        n_bins: Number of calibration bins.
        min_events_per_class: Minimum events for analysis.

    Returns:
        Nested dict matching result.json calibration schema.
    """
    # Extract and filter events
    generated = eval_result_data["generated"]
    rule_outcome = eval_result_data["rule_outcome"]
    failure_index = eval_result_data["failure_index"]

    all_events = extract_events(generated, rule_outcome, failure_index, jumper_map)
    filtered_events, _ = filter_contaminated_events(all_events)
    by_r = stratify_by_r(filtered_events)

    config = {
        "n_bins": n_bins,
        "min_events_per_class": min_events_per_class,
        "probability_method": "empirical_cdf",
    }

    by_r_value_results: dict[int, dict] = {}

    for r_val in sorted(by_r.keys()):
        r_events = by_r[r_val]
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

        for metric_key in metric_keys:
            if metric_key not in eval_result_data:
                continue

            metric_array = eval_result_data[metric_key]
            ece_by_lookback: list[float] = []

            for j in range(1, r_val + 1):
                cal_result = compute_calibration_at_lookback(
                    violations,
                    controls,
                    metric_array,
                    r_val,
                    j,
                    n_bins=n_bins,
                    min_per_class=min(min_events_per_class, 2),
                )
                ece_by_lookback.append(cal_result["ece"])

            r_result["by_metric"][metric_key] = {
                "ece_by_lookback": ece_by_lookback,
            }

        by_r_value_results[r_val] = r_result

    return {
        "config": config,
        "by_r_value": by_r_value_results,
    }
