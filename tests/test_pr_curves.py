"""Tests for precision-recall curves and AUPRC computation.

All tests use synthetic data with fabricated metric arrays and events.
No model or real evaluation data needed. Mirrors test_auroc_horizon.py patterns.
"""

import numpy as np
import pytest

from src.analysis.event_extraction import AnalysisEvent
from src.analysis.pr_curves import (
    _gather_values_at_lookback,
    compute_pr_at_lookback,
    run_pr_analysis,
)
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


def _make_events(walk_indices, resolution_steps, r_value, outcome):
    """Helper to create AnalysisEvent instances."""
    return [
        AnalysisEvent(
            walk_idx=w,
            encounter_step=rs - r_value,
            resolution_step=rs,
            r_value=r_value,
            outcome=outcome,
            is_first_violation=(outcome == RuleOutcome.VIOLATED),
        )
        for w, rs in zip(walk_indices, resolution_steps)
    ]


class TestGatherValuesAtLookback:
    """Tests for _gather_values_at_lookback helper."""

    def test_gather_basic(self):
        """Gather values at lookback=1 from known positions."""
        r_value = 3
        metric_array = np.arange(100, dtype=float).reshape(10, 10)

        violations = _make_events([0, 1], [5, 5], r_value, RuleOutcome.VIOLATED)
        controls = _make_events([2, 3], [5, 5], r_value, RuleOutcome.FOLLOWED)

        viol_vals, ctrl_vals = _gather_values_at_lookback(
            violations, controls, metric_array, lookback=1
        )

        # lookback=1, resolution_step=5 -> index 4
        assert len(viol_vals) == 2
        assert len(ctrl_vals) == 2
        assert viol_vals[0] == metric_array[0, 4]
        assert viol_vals[1] == metric_array[1, 4]

    def test_gather_filters_nan(self):
        """NaN values are excluded from gathered arrays."""
        metric_array = np.ones((10, 10))
        metric_array[0, 4] = np.nan  # walk 0, lookback=1 from step 5

        violations = _make_events([0, 1], [5, 5], 3, RuleOutcome.VIOLATED)
        controls = _make_events([2], [5], 3, RuleOutcome.FOLLOWED)

        viol_vals, ctrl_vals = _gather_values_at_lookback(
            violations, controls, metric_array, lookback=1
        )

        # walk 0's value is NaN so should be excluded
        assert len(viol_vals) == 1
        assert len(ctrl_vals) == 1

    def test_gather_out_of_bounds(self):
        """Events with lookback going out of bounds are excluded."""
        metric_array = np.ones((10, 10))

        # resolution_step=2, lookback=5 -> index=-3 (out of bounds)
        violations = _make_events([0], [2], 5, RuleOutcome.VIOLATED)
        controls = _make_events([1], [2], 5, RuleOutcome.FOLLOWED)

        viol_vals, ctrl_vals = _gather_values_at_lookback(
            violations, controls, metric_array, lookback=5
        )

        assert len(viol_vals) == 0
        assert len(ctrl_vals) == 0


class TestComputePrAtLookback:
    """Tests for PR curve computation at a single lookback distance."""

    def test_pr_perfect_separation(self):
        """Violations have clearly higher metric values than controls.
        AUPRC should be close to 1.0.
        """
        rng = np.random.default_rng(42)
        n_walks = 40
        max_steps = 30
        r_value = 5

        metric_array = rng.standard_normal((n_walks, max_steps))

        # Inject clear separation at lookback=1 (index=resolution_step-1=14)
        violations = _make_events(
            list(range(20)), [15] * 20, r_value, RuleOutcome.VIOLATED
        )
        controls = _make_events(
            list(range(20, 40)), [15] * 20, r_value, RuleOutcome.FOLLOWED
        )

        for ev in violations:
            metric_array[ev.walk_idx, 14] = 10.0 + rng.normal(0, 0.1)
        for ev in controls:
            metric_array[ev.walk_idx, 14] = -10.0 + rng.normal(0, 0.1)

        result = compute_pr_at_lookback(
            violations, controls, metric_array, r_value, lookback=1
        )

        assert result["auprc"] > 0.95
        assert result["n_violations"] == 20
        assert result["n_controls"] == 20
        assert 0.0 < result["prevalence"] < 1.0

    def test_pr_random_data(self):
        """Random metric values. AUPRC should be close to prevalence (no-skill)."""
        rng = np.random.default_rng(42)
        n_walks = 200
        max_steps = 30
        r_value = 5

        metric_array = rng.standard_normal((n_walks, max_steps))

        n_viol = 40
        n_ctrl = 160

        violations = _make_events(
            list(range(n_viol)), [15] * n_viol, r_value, RuleOutcome.VIOLATED
        )
        controls = _make_events(
            list(range(n_viol, n_viol + n_ctrl)),
            [15] * n_ctrl,
            r_value,
            RuleOutcome.FOLLOWED,
        )

        result = compute_pr_at_lookback(
            violations, controls, metric_array, r_value, lookback=1
        )

        prevalence = n_viol / (n_viol + n_ctrl)
        # AUPRC should be near prevalence for random data, within reasonable tolerance
        assert abs(result["auprc"] - prevalence) < 0.15
        assert abs(result["prevalence"] - prevalence) < 0.01

    def test_pr_insufficient_events(self):
        """Fewer than min_per_class events. Should return NaN AUPRC."""
        metric_array = np.ones((10, 20))

        violations = _make_events([0], [10], 3, RuleOutcome.VIOLATED)
        controls = _make_events([1], [10], 3, RuleOutcome.FOLLOWED)

        result = compute_pr_at_lookback(
            violations, controls, metric_array, r_value=3, lookback=1, min_per_class=2
        )

        assert np.isnan(result["auprc"])
        assert result["n_violations"] == 1
        assert result["n_controls"] == 1

    def test_pr_score_direction_handling(self):
        """When violations have LOWER metric values, score negation should kick in.
        AUPRC should still be high (not near prevalence).
        """
        rng = np.random.default_rng(42)
        n_walks = 40
        max_steps = 30
        r_value = 5

        metric_array = rng.standard_normal((n_walks, max_steps))

        violations = _make_events(
            list(range(20)), [15] * 20, r_value, RuleOutcome.VIOLATED
        )
        controls = _make_events(
            list(range(20, 40)), [15] * 20, r_value, RuleOutcome.FOLLOWED
        )

        # Violations have LOWER values (reversed direction)
        for ev in violations:
            metric_array[ev.walk_idx, 14] = -10.0 + rng.normal(0, 0.1)
        for ev in controls:
            metric_array[ev.walk_idx, 14] = 10.0 + rng.normal(0, 0.1)

        result = compute_pr_at_lookback(
            violations, controls, metric_array, r_value, lookback=1
        )

        # Despite reversed direction, AUPRC should still be high
        # because score negation corrects for it
        assert result["auprc"] > 0.9


class TestRunPrAnalysis:
    """Integration test for the full PR analysis pipeline."""

    def test_run_pr_analysis_structure(self):
        """Build synthetic eval_result_data, run full pipeline.
        Verify output structure matches schema.
        """
        rng = np.random.default_rng(42)
        n_sequences = 40
        max_steps = 50
        r_value = 5

        # Create jumper map
        jumper = JumperInfo(vertex_id=5, source_block=0, target_block=1, r=r_value)
        jumper_map = {5: jumper}

        # Generated array with jumper at step 20
        generated = np.zeros((n_sequences, max_steps), dtype=np.int64)
        generated[:, 20] = 5

        # Rule outcome at resolution step 24 (20+5-1)
        rule_outcome = np.full(
            (n_sequences, max_steps - 1), RuleOutcome.NOT_APPLICABLE, dtype=np.int32
        )
        for w in range(15):
            rule_outcome[w, 24] = RuleOutcome.VIOLATED
        for w in range(15, 40):
            rule_outcome[w, 24] = RuleOutcome.FOLLOWED

        failure_index = np.full(n_sequences, -1, dtype=np.int32)
        for w in range(15):
            failure_index[w] = 24

        sequence_lengths = np.full(n_sequences, max_steps, dtype=np.int32)

        metric_key = "qkt.layer_0.grassmannian_distance"
        metric_array = rng.standard_normal((n_sequences, max_steps - 1)).astype(
            np.float32
        )

        eval_data = {
            "generated": generated,
            "rule_outcome": rule_outcome,
            "failure_index": failure_index,
            "sequence_lengths": sequence_lengths,
            metric_key: metric_array,
        }

        result = run_pr_analysis(
            eval_result_data=eval_data,
            jumper_map=jumper_map,
            metric_keys=[metric_key],
            min_events_per_class=5,
        )

        # Verify top-level structure
        assert "config" in result
        assert "by_r_value" in result
        assert result["config"]["min_events_per_class"] == 5

        # Verify r value present
        assert r_value in result["by_r_value"]

        r_block = result["by_r_value"][r_value]
        assert "n_violations" in r_block
        assert "n_controls" in r_block
        assert "by_metric" in r_block

        # Verify metric result
        assert metric_key in r_block["by_metric"]
        metric_result = r_block["by_metric"][metric_key]
        assert "auprc_by_lookback" in metric_result
        assert "prevalence" in metric_result

        # AUPRC curve should have length r_value
        assert len(metric_result["auprc_by_lookback"]) == r_value

        # All AUPRC values should be finite (we have enough events)
        for auprc_val in metric_result["auprc_by_lookback"]:
            assert np.isfinite(auprc_val), f"AUPRC should be finite, got {auprc_val}"

    def test_run_pr_analysis_no_events(self):
        """No jumper encounters at all. Should return empty by_r_value."""
        n_sequences = 10
        max_steps = 20

        generated = np.zeros((n_sequences, max_steps), dtype=np.int64)
        rule_outcome = np.full(
            (n_sequences, max_steps - 1), RuleOutcome.NOT_APPLICABLE, dtype=np.int32
        )
        failure_index = np.full(n_sequences, -1, dtype=np.int32)

        eval_data = {
            "generated": generated,
            "rule_outcome": rule_outcome,
            "failure_index": failure_index,
            "sequence_lengths": np.full(n_sequences, max_steps, dtype=np.int32),
        }

        result = run_pr_analysis(
            eval_result_data=eval_data,
            jumper_map={},
            metric_keys=["qkt.layer_0.stable_rank"],
        )

        assert "by_r_value" in result
        assert len(result["by_r_value"]) == 0
