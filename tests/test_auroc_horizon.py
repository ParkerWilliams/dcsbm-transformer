"""Tests for AUROC computation, predictive horizon, and shuffle controls.

All tests use synthetic data with fabricated metric arrays and events.
No model or real evaluation data needed.
"""

import numpy as np
import pytest

from src.analysis.auroc_horizon import (
    PRIMARY_METRICS,
    auroc_from_groups,
    compute_auroc_curve,
    compute_predictive_horizon,
    run_auroc_analysis,
    run_shuffle_control,
)
from src.analysis.event_extraction import AnalysisEvent
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


class TestAurocFromGroups:
    """Tests for the core AUROC rank-based computation."""

    def test_auroc_from_groups_perfect_separation(self):
        """violations=[5,6,7], controls=[1,2,3]. AUROC must equal 1.0."""
        violations = np.array([5.0, 6.0, 7.0])
        controls = np.array([1.0, 2.0, 3.0])
        assert auroc_from_groups(violations, controls) == 1.0

    def test_auroc_from_groups_no_separation(self):
        """violations=[1,2,3], controls=[1,2,3]. AUROC must equal 0.5."""
        violations = np.array([1.0, 2.0, 3.0])
        controls = np.array([1.0, 2.0, 3.0])
        assert auroc_from_groups(violations, controls) == pytest.approx(0.5)

    def test_auroc_from_groups_reversed(self):
        """violations=[1,2,3], controls=[5,6,7]. AUROC must equal 0.0."""
        violations = np.array([1.0, 2.0, 3.0])
        controls = np.array([5.0, 6.0, 7.0])
        assert auroc_from_groups(violations, controls) == 0.0

    def test_auroc_from_groups_empty_returns_nan(self):
        """Either group empty. Returns NaN."""
        violations = np.array([1.0, 2.0])
        empty = np.array([])
        assert np.isnan(auroc_from_groups(violations, empty))
        assert np.isnan(auroc_from_groups(empty, violations))


class TestComputeAurocCurve:
    """Tests for AUROC curve computation at each lookback distance."""

    def _make_events(self, walk_indices, resolution_steps, r_value, outcome):
        """Helper to create events."""
        return [
            AnalysisEvent(
                walk_idx=w, encounter_step=rs - r_value, resolution_step=rs,
                r_value=r_value, outcome=outcome, is_first_violation=(outcome == RuleOutcome.VIOLATED),
            )
            for w, rs in zip(walk_indices, resolution_steps)
        ]

    def test_compute_auroc_curve_shape(self):
        """For r=8, verify returned array has shape (8,) with AUROC at each lookback j=1..8."""
        r_value = 8
        n_walks = 20
        max_steps = 50

        # Create metric array with random data
        rng = np.random.default_rng(42)
        metric_array = rng.standard_normal((n_walks, max_steps))

        # Create violation and control events
        violation_events = self._make_events(
            walk_indices=list(range(0, 10)),
            resolution_steps=[30] * 10,
            r_value=r_value, outcome=RuleOutcome.VIOLATED,
        )
        control_events = self._make_events(
            walk_indices=list(range(10, 20)),
            resolution_steps=[30] * 10,
            r_value=r_value, outcome=RuleOutcome.FOLLOWED,
        )

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value)
        assert curve.shape == (8,)

    def test_compute_auroc_curve_known_signal(self):
        """Create a synthetic metric array where violation events have consistently
        higher values at lookback j=1..3 and similar values at j=4..8.
        Verify AUROC > 0.8 for j<=3 and ~0.5 for j>3.
        """
        r_value = 8
        n_walks = 40
        max_steps = 50
        rng = np.random.default_rng(42)

        metric_array = rng.standard_normal((n_walks, max_steps))

        # Create events at resolution_step=30
        violation_events = self._make_events(
            walk_indices=list(range(0, 20)),
            resolution_steps=[30] * 20,
            r_value=r_value, outcome=RuleOutcome.VIOLATED,
        )
        control_events = self._make_events(
            walk_indices=list(range(20, 40)),
            resolution_steps=[30] * 20,
            r_value=r_value, outcome=RuleOutcome.FOLLOWED,
        )

        # Inject signal: for violation walks at lookback j=1..3 (indices 29, 28, 27),
        # set high values. For control walks at same positions, set low values.
        for ev in violation_events:
            for j in range(1, 4):
                metric_array[ev.walk_idx, ev.resolution_step - j] = 5.0 + rng.normal(0, 0.1)
        for ev in control_events:
            for j in range(1, 4):
                metric_array[ev.walk_idx, ev.resolution_step - j] = 1.0 + rng.normal(0, 0.1)

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value)

        # j=1..3 should have high AUROC (clear separation)
        for j_idx in range(3):
            assert curve[j_idx] > 0.8, f"AUROC at j={j_idx+1} should be > 0.8, got {curve[j_idx]}"

        # j=4..8 should be near 0.5 (random data, no signal)
        for j_idx in range(3, 8):
            assert 0.1 < curve[j_idx] < 0.9, f"AUROC at j={j_idx+1} should be ~0.5, got {curve[j_idx]}"

    def test_compute_auroc_curve_nan_handling(self):
        """Some metric values are NaN (warmup positions). Verify those lookback
        distances get NaN AUROC (not crash).
        """
        r_value = 5
        n_walks = 10
        max_steps = 20

        metric_array = np.ones((n_walks, max_steps))

        # Create events at resolution_step=5
        # lookback j=5 => index 0, which is valid
        # But we'll set j=4,5 positions to NaN for all walks
        for w in range(n_walks):
            metric_array[w, 0] = np.nan  # lookback j=5 from resolution_step 5
            metric_array[w, 1] = np.nan  # lookback j=4

        violation_events = self._make_events(
            walk_indices=list(range(0, 5)),
            resolution_steps=[5] * 5,
            r_value=r_value, outcome=RuleOutcome.VIOLATED,
        )
        control_events = self._make_events(
            walk_indices=list(range(5, 10)),
            resolution_steps=[5] * 5,
            r_value=r_value, outcome=RuleOutcome.FOLLOWED,
        )

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value)

        # j=4 and j=5 should be NaN (all values at those positions are NaN)
        assert np.isnan(curve[3]), f"AUROC at j=4 should be NaN, got {curve[3]}"
        assert np.isnan(curve[4]), f"AUROC at j=5 should be NaN, got {curve[4]}"

    def test_compute_auroc_curve_min_events(self):
        """Fewer than 2 events per class at some lookback j. Verify NaN returned."""
        r_value = 3
        n_walks = 4
        max_steps = 20

        metric_array = np.ones((n_walks, max_steps))

        # Only 1 violation event, 1 control event => fewer than 2 per class
        violation_events = self._make_events(
            walk_indices=[0],
            resolution_steps=[10],
            r_value=r_value, outcome=RuleOutcome.VIOLATED,
        )
        control_events = self._make_events(
            walk_indices=[1],
            resolution_steps=[10],
            r_value=r_value, outcome=RuleOutcome.FOLLOWED,
        )

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value)

        # With only 1 event per class, all lookback should be NaN
        for j_idx in range(r_value):
            assert np.isnan(curve[j_idx]), f"AUROC at j={j_idx+1} should be NaN with < 2 events per class"


class TestPredictiveHorizon:
    """Tests for predictive horizon computation."""

    def test_predictive_horizon_basic(self):
        """AUROC curve [0.5, 0.6, 0.8, 0.9, 0.85, 0.7, 0.5, 0.5] with threshold 0.75.
        Horizon = 5 (furthest j where AUROC > 0.75, j indexed 1-based).
        """
        auroc_curve = np.array([0.5, 0.6, 0.8, 0.9, 0.85, 0.7, 0.5, 0.5])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 5

    def test_predictive_horizon_none_exceeds(self):
        """All AUROC < 0.75. Horizon = 0."""
        auroc_curve = np.array([0.5, 0.6, 0.65, 0.7, 0.6, 0.5, 0.4, 0.3])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 0


class TestShuffleControl:
    """Tests for shuffle permutation controls."""

    def test_shuffle_control_no_signal(self):
        """Random metric values for both groups. Verify shuffle_flag=False.
        Uses large samples (100 per group) with r=2 to keep max-AUROC
        distribution well below 0.6 even with multiple lookback distances.
        """
        rng = np.random.default_rng(42)

        r_value = 2
        n_walks = 200
        max_steps = 30

        metric_array = rng.standard_normal((n_walks, max_steps))

        violation_events = [
            AnalysisEvent(
                walk_idx=w, encounter_step=15 - r_value, resolution_step=15,
                r_value=r_value, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
            )
            for w in range(100)
        ]
        control_events = [
            AnalysisEvent(
                walk_idx=w, encounter_step=15 - r_value, resolution_step=15,
                r_value=r_value, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            )
            for w in range(100, 200)
        ]

        result = run_shuffle_control(
            violation_events, control_events, metric_array, r_value,
            n_permutations=500, flag_threshold=0.6, rng=42,
        )

        assert result["shuffle_flag"] is False
        assert result["shuffle_auroc_mean"] < 0.6

    def test_shuffle_control_positional_artifact(self):
        """Metric values correlated with position, not class label.
        Verify shuffle_flag=True (shuffled AUROC exceeds 0.6 because
        signal is positional).
        """
        rng = np.random.default_rng(42)

        r_value = 5
        n_walks = 40
        max_steps = 30

        # Create metric array where values depend on walk position,
        # not on violation/control label
        metric_array = np.zeros((n_walks, max_steps))
        for w in range(n_walks):
            # Position-dependent signal: walks 0-19 get higher values
            # at the lookback positions, walks 20-39 get lower.
            # This is a positional artifact because even after shuffling labels,
            # the first 20 walks always have higher values.
            if w < 20:
                metric_array[w, :] = 3.0 + rng.normal(0, 0.1, max_steps)
            else:
                metric_array[w, :] = 1.0 + rng.normal(0, 0.1, max_steps)

        violation_events = [
            AnalysisEvent(
                walk_idx=w, encounter_step=15 - r_value, resolution_step=15,
                r_value=r_value, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
            )
            for w in range(20)
        ]
        control_events = [
            AnalysisEvent(
                walk_idx=w, encounter_step=15 - r_value, resolution_step=15,
                r_value=r_value, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            )
            for w in range(20, 40)
        ]

        result = run_shuffle_control(
            violation_events, control_events, metric_array, r_value,
            n_permutations=500, flag_threshold=0.6, rng=42,
        )

        # Even after shuffling labels, walks 0-19 always have higher values,
        # so shuffled AUROC should be high (positional artifact).
        assert result["shuffle_flag"] is True
        assert result["shuffle_auroc_p95"] > 0.6


class TestRunAurocAnalysis:
    """Integration test for the full analysis pipeline."""

    def test_run_auroc_analysis_full_pipeline(self):
        """Create synthetic EvaluationResult-like arrays, fake jumper_map,
        run full pipeline. Verify output dict structure matches result.json
        schema from RESEARCH.md.
        """
        rng = np.random.default_rng(42)
        n_sequences = 40
        max_steps = 50
        r_value = 8

        # Create jumper map with one jumper vertex (id=5, r=8)
        jumper = JumperInfo(vertex_id=5, source_block=0, target_block=1, r=r_value)
        jumper_map = {5: jumper}

        # Create generated array with jumper at known positions
        generated = np.zeros((n_sequences, max_steps), dtype=np.int64)
        # Place jumper vertex 5 at step 20 for all walks
        generated[:, 20] = 5

        # rule_outcome: mark resolution at step 27 (20 + 8 - 1)
        rule_outcome = np.full((n_sequences, max_steps - 1), RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
        # First 15 walks: VIOLATED
        for w in range(15):
            rule_outcome[w, 27] = RuleOutcome.VIOLATED
        # Next 25 walks: FOLLOWED
        for w in range(15, 40):
            rule_outcome[w, 27] = RuleOutcome.FOLLOWED

        # failure_index: for violated walks, step 27
        failure_index = np.full(n_sequences, -1, dtype=np.int32)
        for w in range(15):
            failure_index[w] = 27

        sequence_lengths = np.full(n_sequences, max_steps, dtype=np.int32)

        # Metric arrays: use one primary metric key
        metric_key = "qkt.layer_0.grassmannian_distance"
        metric_array = rng.standard_normal((n_sequences, max_steps - 1)).astype(np.float32)

        # Inject signal at lookback j=1..3 for violations
        for w in range(15):
            for j in range(1, 4):
                metric_array[w, 28 - j] = 5.0 + rng.normal(0, 0.1)
        for w in range(15, 40):
            for j in range(1, 4):
                metric_array[w, 28 - j] = 1.0 + rng.normal(0, 0.1)

        eval_data = {
            "generated": generated,
            "rule_outcome": rule_outcome,
            "failure_index": failure_index,
            "sequence_lengths": sequence_lengths,
            metric_key: metric_array,
        }

        result = run_auroc_analysis(
            eval_result_data=eval_data,
            jumper_map=jumper_map,
            metric_keys=[metric_key],
            horizon_threshold=0.75,
            shuffle_flag_threshold=0.6,
            n_shuffle=100,  # Small for test speed
            min_events_per_class=5,
        )

        # Verify top-level structure
        assert "config" in result
        assert "contamination_audit" in result
        assert "by_r_value" in result

        # Verify config
        assert result["config"]["horizon_threshold"] == 0.75
        assert result["config"]["shuffle_flag_threshold"] == 0.6

        # Verify r value present
        assert r_value in result["by_r_value"] or str(r_value) in result["by_r_value"]

        r_block = result["by_r_value"][r_value]
        assert "n_violations" in r_block
        assert "n_controls" in r_block
        assert r_block["n_violations"] == 15
        assert r_block["n_controls"] == 25

        # Verify metric result
        assert "by_metric" in r_block
        assert metric_key in r_block["by_metric"]

        metric_result = r_block["by_metric"][metric_key]
        assert "auroc_by_lookback" in metric_result
        assert "horizon" in metric_result
        assert "max_auroc" in metric_result
        assert "shuffle_flag" in metric_result
        assert "n_valid_by_lookback" in metric_result
        assert "is_primary" in metric_result

        # AUROC curve should have length r_value
        assert len(metric_result["auroc_by_lookback"]) == r_value

        # Horizon should be > 0 since we injected signal
        assert metric_result["horizon"] >= 1

        # Verify primary metric identification
        assert metric_result["is_primary"] is True

    def test_primary_metrics_constant(self):
        """Verify PRIMARY_METRICS contains the 5 pre-registered metrics."""
        assert len(PRIMARY_METRICS) == 5
        assert "qkt.grassmannian_distance" in PRIMARY_METRICS
        assert "qkt.spectral_gap_1_2" in PRIMARY_METRICS
        assert "qkt.spectral_entropy" in PRIMARY_METRICS
        assert "avwo.stable_rank" in PRIMARY_METRICS
        assert "avwo.grassmannian_distance" in PRIMARY_METRICS
