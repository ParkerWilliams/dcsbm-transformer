"""Tests for calibration diagnostics: reliability diagrams and ECE.

All tests use synthetic data with fabricated metric arrays and events.
"""

import numpy as np
import pytest

from src.analysis.calibration import (
    compute_calibration_at_lookback,
    compute_ece,
    metric_to_pseudo_probability,
    run_calibration_analysis,
)
from src.analysis.event_extraction import AnalysisEvent
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


class TestMetricToPseudoProbability:
    """Tests for rank-based probability conversion."""

    def test_basic_conversion(self):
        """[1, 2, 3, 4] -> [0.25, 0.5, 0.75, 1.0]."""
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        probs = metric_to_pseudo_probability(scores)
        expected = np.array([0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(probs, expected)

    def test_ties(self):
        """Tied values get midranks."""
        scores = np.array([1.0, 1.0, 3.0, 4.0])
        probs = metric_to_pseudo_probability(scores)
        # Ties at 1.0: ranks 1,2 -> midrank 1.5
        # Probabilities: 1.5/4, 1.5/4, 3/4, 4/4
        expected = np.array([0.375, 0.375, 0.75, 1.0])
        np.testing.assert_array_almost_equal(probs, expected)

    def test_empty_array(self):
        """Empty input returns empty output."""
        probs = metric_to_pseudo_probability(np.array([]))
        assert len(probs) == 0

    def test_single_value(self):
        """Single value gets probability 1.0."""
        probs = metric_to_pseudo_probability(np.array([5.0]))
        np.testing.assert_array_almost_equal(probs, [1.0])

    def test_range_in_0_1(self):
        """All probabilities should be in (0, 1]."""
        rng = np.random.default_rng(42)
        scores = rng.standard_normal(100)
        probs = metric_to_pseudo_probability(scores)
        assert np.all(probs > 0)
        assert np.all(probs <= 1)


class TestComputeEce:
    """Tests for Expected Calibration Error computation."""

    def test_perfect_calibration(self):
        """When fraction_of_positives == mean_predicted_value, ECE = 0."""
        fop = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        mpv = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        counts = np.array([20, 20, 20, 20, 20])
        ece = compute_ece(fop, mpv, counts)
        assert ece == pytest.approx(0.0, abs=1e-10)

    def test_worst_calibration(self):
        """When predictions are completely opposite, ECE should be high."""
        fop = np.array([0.0, 1.0])
        mpv = np.array([1.0, 0.0])
        counts = np.array([50, 50])
        ece = compute_ece(fop, mpv, counts)
        assert ece == pytest.approx(1.0, abs=1e-10)

    def test_ece_range(self):
        """ECE should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            n_bins = 5
            fop = rng.random(n_bins)
            mpv = rng.random(n_bins)
            counts = rng.integers(1, 100, size=n_bins)
            ece = compute_ece(fop, mpv, counts)
            assert 0 <= ece <= 1

    def test_empty_bins(self):
        """All zero bin counts -> NaN."""
        ece = compute_ece(np.array([]), np.array([]), np.array([]))
        assert np.isnan(ece)


class TestComputeCalibrationAtLookback:
    """Tests for calibration at a single lookback distance."""

    def test_calibration_basic(self):
        """Compute calibration on synthetic data with enough events."""
        rng = np.random.default_rng(42)
        n_walks = 100
        max_steps = 30
        r_value = 5

        metric_array = rng.standard_normal((n_walks, max_steps))

        violations = _make_events(
            list(range(30)), [15] * 30, r_value, RuleOutcome.VIOLATED
        )
        controls = _make_events(
            list(range(30, 100)), [15] * 100, r_value, RuleOutcome.FOLLOWED
        )
        # Trim controls to actual count
        controls = controls[:70]

        result = compute_calibration_at_lookback(
            violations, controls, metric_array, r_value, lookback=1, n_bins=10
        )

        assert "ece" in result
        assert "fraction_of_positives" in result
        assert "mean_predicted_value" in result
        assert "bin_counts" in result
        assert result["n_bins"] == 10
        assert np.isfinite(result["ece"])
        assert 0.0 <= result["ece"] <= 1.0

    def test_calibration_insufficient_events(self):
        """Fewer than min_per_class -> NaN ECE."""
        metric_array = np.ones((10, 20))

        violations = _make_events([0], [10], 3, RuleOutcome.VIOLATED)
        controls = _make_events([1], [10], 3, RuleOutcome.FOLLOWED)

        result = compute_calibration_at_lookback(
            violations, controls, metric_array, r_value=3, lookback=1, min_per_class=5
        )

        assert np.isnan(result["ece"])


class TestRunCalibrationAnalysis:
    """Integration test for the full calibration analysis pipeline."""

    def test_run_calibration_analysis_structure(self):
        """Build synthetic eval_result_data, verify output structure."""
        rng = np.random.default_rng(42)
        n_sequences = 40
        max_steps = 50
        r_value = 5

        jumper = JumperInfo(vertex_id=5, source_block=0, target_block=1, r=r_value)
        jumper_map = {5: jumper}

        generated = np.zeros((n_sequences, max_steps), dtype=np.int64)
        generated[:, 20] = 5

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

        metric_key = "qkt.layer_0.grassmannian_distance"
        metric_array = rng.standard_normal((n_sequences, max_steps - 1)).astype(
            np.float32
        )

        eval_data = {
            "generated": generated,
            "rule_outcome": rule_outcome,
            "failure_index": failure_index,
            "sequence_lengths": np.full(n_sequences, max_steps, dtype=np.int32),
            metric_key: metric_array,
        }

        result = run_calibration_analysis(
            eval_result_data=eval_data,
            jumper_map=jumper_map,
            metric_keys=[metric_key],
            n_bins=10,
            min_events_per_class=5,
        )

        assert "config" in result
        assert result["config"]["n_bins"] == 10
        assert result["config"]["probability_method"] == "empirical_cdf"
        assert "by_r_value" in result

        assert r_value in result["by_r_value"]
        r_block = result["by_r_value"][r_value]
        assert "by_metric" in r_block
        assert metric_key in r_block["by_metric"]

        metric_result = r_block["by_metric"][metric_key]
        assert "ece_by_lookback" in metric_result
        assert len(metric_result["ece_by_lookback"]) == r_value

        # All ECE values should be finite
        for ece_val in metric_result["ece_by_lookback"]:
            assert np.isfinite(ece_val)
            assert 0.0 <= ece_val <= 1.0
