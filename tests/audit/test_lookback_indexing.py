"""Audit tests for lookback indexing fence-post correctness (AUROC-02).

Verifies that compute_auroc_curve correctly retrieves metric values at
resolution_step - j using planted-signal metric arrays. A distinctive value
(999.0) at exactly one position detects off-by-one errors: if indexing is
wrong by +/-1, the AUROC will be 0.5 (chance) instead of 1.0 at the
planted position. Also verifies compute_predictive_horizon logic.
"""

import numpy as np

from src.analysis.auroc_horizon import compute_auroc_curve, compute_predictive_horizon
from src.analysis.event_extraction import AnalysisEvent
from src.evaluation.behavioral import RuleOutcome


class TestPlantedSignalRetrieval:
    """Core fence-post test: plant distinctive values and verify correct retrieval.

    Indexing chain:
      j=1 -> metric_array[:, resolution_step - 1]  (the resolution step metric)
      j=2 -> metric_array[:, resolution_step - 2]  (one step before resolution)
      j=3 -> metric_array[:, resolution_step - 3]  (two steps before resolution)

    j=1 retrieves the metric at index resolution_step - 1, which is the attention
    pattern when predicting token at position resolution_step from tokens 0..resolution_step-1.
    This is the LAST metric value before the outcome is determined.
    """

    def test_basic_planted_signal(self) -> None:
        """Plant 999.0 at column 7 for violations, -999.0 for controls.
        resolution_step=8, r_value=3.
        j=1 retrieves index 8-1=7 (planted signal) -> AUROC=1.0 (perfect separation).
        j=2 retrieves index 8-2=6 (zero) -> AUROC=0.5 (no separation).
        j=3 retrieves index 8-3=5 (zero) -> AUROC=0.5 (no separation).
        """
        metric_array = np.zeros((4, 20))
        # Plant signal at column 7
        metric_array[0, 7] = 999.0
        metric_array[1, 7] = 999.0
        metric_array[2, 7] = -999.0
        metric_array[3, 7] = -999.0

        violation_events = [
            AnalysisEvent(walk_idx=0, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
            AnalysisEvent(walk_idx=1, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
        ]
        control_events = [
            AnalysisEvent(walk_idx=2, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
            AnalysisEvent(walk_idx=3, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
        ]

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value=3)

        # j=1 (index 7): planted signal -> perfect separation
        assert curve[0] == 1.0, (
            f"j=1 should retrieve planted signal at index 7, got AUROC={curve[0]}"
        )
        # j=2 (index 6): all zeros -> chance level
        assert curve[1] == 0.5, (
            f"j=2 should be chance level (all zeros at index 6), got AUROC={curve[1]}"
        )
        # j=3 (index 5): all zeros -> chance level
        assert curve[2] == 0.5, (
            f"j=3 should be chance level (all zeros at index 5), got AUROC={curve[2]}"
        )

    def test_fence_post_sensitivity(self) -> None:
        """Shift planted signal to column 6 instead of 7.
        resolution_step=8, r_value=3.
        j=1 retrieves index 7 (zero) -> AUROC=0.5.
        j=2 retrieves index 6 (planted signal) -> AUROC=1.0.
        This confirms a +/-1 error in indexing would be caught.
        """
        metric_array = np.zeros((4, 20))
        # Plant signal at column 6 (one position earlier)
        metric_array[0, 6] = 999.0
        metric_array[1, 6] = 999.0
        metric_array[2, 6] = -999.0
        metric_array[3, 6] = -999.0

        violation_events = [
            AnalysisEvent(walk_idx=0, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
            AnalysisEvent(walk_idx=1, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
        ]
        control_events = [
            AnalysisEvent(walk_idx=2, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
            AnalysisEvent(walk_idx=3, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
        ]

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value=3)

        # j=1 (index 7): all zeros -> chance level
        assert curve[0] == 0.5, (
            f"j=1 retrieves index 7 (no signal), expected 0.5, got AUROC={curve[0]}"
        )
        # j=2 (index 6): planted signal -> perfect separation
        assert curve[1] == 1.0, (
            f"j=2 retrieves index 6 (planted signal), expected 1.0, got AUROC={curve[1]}"
        )
        # j=3 (index 5): all zeros -> chance level
        assert curve[2] == 0.5, (
            f"j=3 retrieves index 5 (no signal), expected 0.5, got AUROC={curve[2]}"
        )

    def test_j1_semantic_meaning(self) -> None:
        """Verify j=1 retrieves metric_array[:, resolution_step - 1].

        j=1 is the metric at the resolution step itself -- the attention pattern when
        predicting token at position resolution_step from tokens 0..resolution_step-1.
        This is the LAST metric value before the outcome is determined, not one step
        before resolution. j=0 is excluded because it would be post-hoc (not predictive).

        Test: resolution_step=8.
          j=1 -> index 7 (planted 100.0 / -100.0)
          j=2 -> index 6 (planted 200.0 / -200.0)
          j=3 -> index 5 (planted 300.0 / -300.0)
        All three should yield AUROC=1.0 with distinct planted values confirming
        each j retrieves from the correct, unique position.
        """
        metric_array = np.zeros((4, 20))
        # Plant distinct values at each lookback position
        for walk_idx in [0, 1]:  # violations
            metric_array[walk_idx, 7] = 100.0   # j=1 position
            metric_array[walk_idx, 6] = 200.0   # j=2 position
            metric_array[walk_idx, 5] = 300.0   # j=3 position
        for walk_idx in [2, 3]:  # controls
            metric_array[walk_idx, 7] = -100.0  # j=1 position
            metric_array[walk_idx, 6] = -200.0  # j=2 position
            metric_array[walk_idx, 5] = -300.0  # j=3 position

        violation_events = [
            AnalysisEvent(walk_idx=0, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
            AnalysisEvent(walk_idx=1, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
        ]
        control_events = [
            AnalysisEvent(walk_idx=2, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
            AnalysisEvent(walk_idx=3, encounter_step=5, resolution_step=8,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
        ]

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value=3)

        # All three lookback positions have signal -> all should be 1.0
        assert curve[0] == 1.0, f"j=1 (index 7): expected 1.0, got {curve[0]}"
        assert curve[1] == 1.0, f"j=2 (index 6): expected 1.0, got {curve[1]}"
        assert curve[2] == 1.0, f"j=3 (index 5): expected 1.0, got {curve[2]}"

    def test_multiple_resolution_steps(self) -> None:
        """Events with different resolution steps retrieve from different columns.
        Event A: resolution_step=10, j=1 -> index 9.
        Event B: resolution_step=15, j=1 -> index 14.
        Plant signal at both positions to verify vectorized indexing handles
        heterogeneous resolution steps correctly.
        """
        metric_array = np.zeros((4, 20))
        # Violations: plant positive signal at their respective j=1 positions
        metric_array[0, 9] = 999.0    # walk 0, resolution_step=10, j=1 -> idx 9
        metric_array[1, 14] = 999.0   # walk 1, resolution_step=15, j=1 -> idx 14
        # Controls: plant negative signal
        metric_array[2, 9] = -999.0
        metric_array[3, 14] = -999.0

        violation_events = [
            AnalysisEvent(walk_idx=0, encounter_step=7, resolution_step=10,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
            AnalysisEvent(walk_idx=1, encounter_step=12, resolution_step=15,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
        ]
        control_events = [
            AnalysisEvent(walk_idx=2, encounter_step=7, resolution_step=10,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
            AnalysisEvent(walk_idx=3, encounter_step=12, resolution_step=15,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
        ]

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value=3)

        # j=1 retrieves from both planted positions -> perfect separation
        assert curve[0] == 1.0, f"j=1 should find signal at both positions, got {curve[0]}"
        # j=2 retrieves from indices 8 and 13 (zeros) -> chance level
        assert curve[1] == 0.5, f"j=2 should be chance level, got {curve[1]}"


class TestMetricArrayShapeOffset:
    """Verify metric_array has max_steps-1 columns and bounds are respected.

    Per CONTEXT.md: metric_array has max_steps-1 columns because the metric at
    position i represents the attention pattern predicting token i+1 from tokens
    0..i. There are max_steps tokens but only max_steps-1 transitions.
    """

    def test_boundary_resolution_step(self) -> None:
        """metric_array shape (2,19) for max_steps=20.
        resolution_step=19 (the maximum valid for this array size).
        j=1 retrieves index 18 (last column, valid).
        j=2 retrieves index 17 (valid).
        No IndexError should occur.
        """
        metric_array = np.zeros((4, 19))
        # Plant signal at last column
        metric_array[0, 18] = 999.0
        metric_array[1, 18] = 999.0
        metric_array[2, 18] = -999.0
        metric_array[3, 18] = -999.0

        violation_events = [
            AnalysisEvent(walk_idx=0, encounter_step=16, resolution_step=19,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
            AnalysisEvent(walk_idx=1, encounter_step=16, resolution_step=19,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
        ]
        control_events = [
            AnalysisEvent(walk_idx=2, encounter_step=16, resolution_step=19,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
            AnalysisEvent(walk_idx=3, encounter_step=16, resolution_step=19,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
        ]

        # Should not raise IndexError
        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value=3)

        # j=1 (index 18): planted signal -> 1.0
        assert curve[0] == 1.0, f"j=1 at boundary should find signal, got {curve[0]}"
        # j=2 (index 17): zeros -> 0.5
        assert curve[1] == 0.5, f"j=2 should be chance level, got {curve[1]}"
        # j=3 (index 16): zeros -> 0.5
        assert curve[2] == 0.5, f"j=3 should be chance level, got {curve[2]}"

    def test_out_of_bounds_handling(self) -> None:
        """resolution_step=20 with metric_array of 19 columns.
        j=1 needs index 19, but metric_array only has columns 0..18.
        The bounds check (viol_idx < n_steps) should filter it out -> NaN.
        j=2 needs index 18 (valid, last column).
        j=3 needs index 17 (valid).
        """
        metric_array = np.zeros((4, 19))

        violation_events = [
            AnalysisEvent(walk_idx=0, encounter_step=17, resolution_step=20,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
            AnalysisEvent(walk_idx=1, encounter_step=17, resolution_step=20,
                          r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
        ]
        control_events = [
            AnalysisEvent(walk_idx=2, encounter_step=17, resolution_step=20,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
            AnalysisEvent(walk_idx=3, encounter_step=17, resolution_step=20,
                          r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
        ]

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value=3)

        # j=1 (index 19): out of bounds -> NaN
        assert np.isnan(curve[0]), f"j=1 out of bounds should be NaN, got {curve[0]}"
        # j=2 (index 18): valid (zeros) -> 0.5
        assert curve[1] == 0.5, f"j=2 in bounds should be 0.5, got {curve[1]}"
        # j=3 (index 17): valid (zeros) -> 0.5
        assert curve[2] == 0.5, f"j=3 in bounds should be 0.5, got {curve[2]}"

    def test_j_exceeding_resolution_step(self) -> None:
        """resolution_step=2, r_value=5.
        j=1 -> index 1 (valid).
        j=2 -> index 0 (valid).
        j=3 -> index -1 (filtered by viol_idx >= 0 check -> NaN).
        j=4 -> index -2 (filtered -> NaN).
        j=5 -> index -3 (filtered -> NaN).
        """
        metric_array = np.zeros((4, 19))
        # Plant signal at index 1 (j=1 position for resolution_step=2)
        metric_array[0, 1] = 999.0
        metric_array[1, 1] = 999.0
        metric_array[2, 1] = -999.0
        metric_array[3, 1] = -999.0

        violation_events = [
            AnalysisEvent(walk_idx=0, encounter_step=0, resolution_step=2,
                          r_value=5, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
            AnalysisEvent(walk_idx=1, encounter_step=0, resolution_step=2,
                          r_value=5, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
        ]
        control_events = [
            AnalysisEvent(walk_idx=2, encounter_step=0, resolution_step=2,
                          r_value=5, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
            AnalysisEvent(walk_idx=3, encounter_step=0, resolution_step=2,
                          r_value=5, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
        ]

        curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value=5)

        # j=1 (index 1): planted signal -> 1.0
        assert curve[0] == 1.0, f"j=1 should find signal, got {curve[0]}"
        # j=2 (index 0): zeros -> 0.5
        assert curve[1] == 0.5, f"j=2 should be chance (zeros), got {curve[1]}"
        # j=3,4,5 (indices -1,-2,-3): filtered -> NaN
        assert np.isnan(curve[2]), f"j=3 (index -1) should be NaN, got {curve[2]}"
        assert np.isnan(curve[3]), f"j=4 (index -2) should be NaN, got {curve[3]}"
        assert np.isnan(curve[4]), f"j=5 (index -3) should be NaN, got {curve[4]}"


class TestComputePredictiveHorizon:
    """Verify predictive horizon finds the largest j where AUROC > threshold."""

    def test_basic_horizon(self) -> None:
        """auroc_curve=[0.8, 0.9, 0.6, 0.5] with threshold=0.75.
        j=1 (curve[0]=0.8) > 0.75 yes.
        j=2 (curve[1]=0.9) > 0.75 yes.
        j=3 (curve[2]=0.6) > 0.75 no.
        j=4 (curve[3]=0.5) > 0.75 no.
        Largest j above threshold = 2.
        """
        auroc_curve = np.array([0.8, 0.9, 0.6, 0.5])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 2, f"Expected horizon=2, got {horizon}"

    def test_no_horizon(self) -> None:
        """All AUROC values below threshold -> returns 0.
        No lookback distance has sufficient predictive power.
        """
        auroc_curve = np.array([0.5, 0.6, 0.4, 0.3])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 0, f"Expected horizon=0, got {horizon}"

    def test_all_above_threshold(self) -> None:
        """All values above threshold -> returns r (the largest j).
        Predictive signal persists across the entire lookback range.
        """
        auroc_curve = np.array([0.9, 0.85, 0.8, 0.76])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 4, f"Expected horizon=4 (all above 0.75), got {horizon}"

    def test_nan_handling(self) -> None:
        """NaN values mixed in are skipped. Horizon from finite values only.
        auroc_curve=[0.8, NaN, 0.9, NaN].
        j=3 (curve[2]=0.9) > 0.75 -> horizon = 3.
        """
        auroc_curve = np.array([0.8, np.nan, 0.9, np.nan])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        # j=4 is NaN (skipped), j=3 is 0.9 > 0.75 -> horizon = 3
        assert horizon == 3, f"Expected horizon=3 (NaN skipped), got {horizon}"

    def test_threshold_boundary_strict_inequality(self) -> None:
        """AUROC exactly equal to threshold (0.75) is NOT included.
        compute_predictive_horizon uses val > threshold (strict inequality).
        auroc_curve=[0.75, 0.76, 0.74] with threshold=0.75.
        j=1 (0.75): NOT > 0.75, excluded.
        j=2 (0.76): > 0.75, included.
        Horizon = 2.
        """
        auroc_curve = np.array([0.75, 0.76, 0.74])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 2, (
            f"Expected horizon=2 (0.75 not included, strict >), got {horizon}"
        )

    def test_only_at_threshold_returns_zero(self) -> None:
        """If all AUROC values are exactly at threshold, horizon = 0.
        Strict inequality means 0.75 == 0.75 is not above threshold.
        """
        auroc_curve = np.array([0.75, 0.75, 0.75])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 0, (
            f"Expected horizon=0 (strict > excludes 0.75), got {horizon}"
        )

    def test_empty_curve(self) -> None:
        """Empty AUROC curve -> returns 0 (no lookback distances)."""
        auroc_curve = np.array([])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 0, f"Expected horizon=0 for empty curve, got {horizon}"

    def test_all_nan_curve(self) -> None:
        """All-NaN AUROC curve -> returns 0 (no finite values above threshold)."""
        auroc_curve = np.array([np.nan, np.nan, np.nan])
        horizon = compute_predictive_horizon(auroc_curve, threshold=0.75)
        assert horizon == 0, f"Expected horizon=0 for all-NaN curve, got {horizon}"
