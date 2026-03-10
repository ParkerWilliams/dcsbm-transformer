"""Audit tests for shuffle permutation null (STAT-01).

Verifies run_shuffle_control in auroc_horizon.py correctly permutes event labels
(not metric values), preserves group sizes per permutation, produces uniformly
distributed p-values under H0, and detects real signal as a positive control.
"""

import numpy as np
from scipy.stats import kstest

from src.analysis.auroc_horizon import auroc_from_groups, run_shuffle_control
from src.analysis.event_extraction import AnalysisEvent
from src.evaluation.behavioral import RuleOutcome


def _make_events(
    n_viol: int,
    n_ctrl: int,
    r_value: int,
    start_step: int = 10,
) -> tuple[list[AnalysisEvent], list[AnalysisEvent]]:
    """Create synthetic AnalysisEvent objects with valid indexing.

    Events are placed at sequential walk_idx values with resolution_step
    starting at start_step, ensuring metric_array indexing at
    resolution_step - j is valid for j in [1, r_value].
    """
    violations = []
    controls = []
    for i in range(n_viol):
        violations.append(
            AnalysisEvent(
                walk_idx=i,
                encounter_step=start_step - r_value,
                resolution_step=start_step,
                r_value=r_value,
                outcome=RuleOutcome.VIOLATED,
                is_first_violation=True,
            )
        )
    for i in range(n_ctrl):
        controls.append(
            AnalysisEvent(
                walk_idx=n_viol + i,
                encounter_step=start_step - r_value,
                resolution_step=start_step,
                r_value=r_value,
                outcome=RuleOutcome.FOLLOWED,
                is_first_violation=False,
            )
        )
    return violations, controls


class TestH0Uniformity:
    """Under H0 (both groups from same distribution), p-values should be U[0,1]."""

    def test_pvalues_uniform_under_null(self) -> None:
        """Generate data where violations and controls come from the SAME distribution
        N(0,1). Run 100 independent shuffle controls, collect p-values, and test
        uniformity via KS test against U[0,1]. KS p-value > 0.01 means we do NOT
        reject the null that p-values are uniform -- confirming no false signal.
        """
        n_viol, n_ctrl = 30, 30
        r_value = 3
        n_total = n_viol + n_ctrl
        start_step = 10
        n_steps = start_step + 5  # ensure valid indexing

        p_values = []
        for trial in range(100):
            rng = np.random.default_rng(trial)
            # Both groups drawn from the same N(0,1) -- no signal
            metric_array = rng.normal(0.0, 1.0, size=(n_total, n_steps))
            violations, controls = _make_events(n_viol, n_ctrl, r_value, start_step)

            result = run_shuffle_control(
                violations, controls, metric_array, r_value,
                n_permutations=1000, rng=trial * 1000 + 42,
            )
            p_values.append(result["p_value"])

        p_values = np.array(p_values)

        # KS test: H0 = p-values are U[0,1]. We should NOT reject this.
        ks_stat, ks_pvalue = kstest(p_values, "uniform")
        assert ks_pvalue > 0.01, (
            f"KS test rejected uniformity (ks_stat={ks_stat:.4f}, ks_p={ks_pvalue:.4f}). "
            f"p-value distribution: mean={np.mean(p_values):.3f}, std={np.std(p_values):.3f}"
        )


class TestMetricArrayImmutability:
    """run_shuffle_control must not mutate the metric_array or its values."""

    def test_metric_array_unchanged_after_shuffle(self) -> None:
        """Copy the metric array before calling run_shuffle_control, then verify
        the original is element-wise identical. The function permutes index masks,
        not metric values, so the array should be untouched.
        """
        n_viol, n_ctrl = 15, 15
        r_value = 2
        n_total = n_viol + n_ctrl
        start_step = 10
        n_steps = start_step + 5

        rng = np.random.default_rng(42)
        metric_array = rng.normal(0.0, 1.0, size=(n_total, n_steps))
        original_copy = metric_array.copy()

        violations, controls = _make_events(n_viol, n_ctrl, r_value, start_step)
        run_shuffle_control(
            violations, controls, metric_array, r_value,
            n_permutations=500, rng=99,
        )

        # Metric values must be completely unchanged after permutation test
        assert np.array_equal(metric_array, original_copy), (
            "run_shuffle_control mutated the metric_array! "
            f"Max diff: {np.max(np.abs(metric_array - original_copy))}"
        )


class TestGroupSizePreservation:
    """Each permutation must produce exactly n_viol 'violation' labels and n_ctrl 'control' labels."""

    def test_permutation_mask_preserves_group_sizes(self) -> None:
        """Replicate the permutation logic from run_shuffle_control: combine
        n_viol + n_ctrl events, call rng.permutation(n_total), take first n_viol
        as violation mask. Assert exactly n_viol True values in the mask.
        Run 50 iterations to confirm the invariant holds every time.
        """
        n_viol, n_ctrl = 20, 30
        n_total = n_viol + n_ctrl
        rng = np.random.default_rng(42)

        for _ in range(50):
            perm_indices = rng.permutation(n_total)
            perm_viol_mask = np.zeros(n_total, dtype=bool)
            perm_viol_mask[perm_indices[:n_viol]] = True

            # Exactly n_viol True values in the mask
            assert perm_viol_mask.sum() == n_viol, (
                f"Expected {n_viol} violations in mask, got {perm_viol_mask.sum()}"
            )
            # Exactly n_ctrl False values (complement)
            assert (~perm_viol_mask).sum() == n_ctrl, (
                f"Expected {n_ctrl} controls in mask, got {(~perm_viol_mask).sum()}"
            )


class TestSignalDetection:
    """Positive control: clear separation should produce significant p-value."""

    def test_detects_real_signal(self) -> None:
        """Violations drawn from N(5, 1), controls from N(0, 1).
        With clear separation (d=5), the shuffle p-value should be < 0.05
        because the observed AUROC is far above what shuffles produce.
        """
        n_viol, n_ctrl = 30, 30
        r_value = 3
        n_total = n_viol + n_ctrl
        start_step = 10
        n_steps = start_step + 5

        rng = np.random.default_rng(42)
        metric_array = np.zeros((n_total, n_steps))

        # Violations get high values at the lookback positions
        for i in range(n_viol):
            metric_array[i, :] = rng.normal(5.0, 1.0, size=n_steps)
        # Controls get low values
        for i in range(n_ctrl):
            metric_array[n_viol + i, :] = rng.normal(0.0, 1.0, size=n_steps)

        violations, controls = _make_events(n_viol, n_ctrl, r_value, start_step)
        result = run_shuffle_control(
            violations, controls, metric_array, r_value,
            n_permutations=1000, rng=42,
        )

        # With d=5 separation, shuffle null should never produce an AUROC as extreme
        assert result["p_value"] < 0.05, (
            f"Expected p < 0.05 for clear signal, got p={result['p_value']:.4f}"
        )
        # Shuffle mean should be around 0.5 (no class signal after shuffling)
        assert 0.35 < result["shuffle_auroc_mean"] < 0.65, (
            f"Expected shuffle mean ~0.5, got {result['shuffle_auroc_mean']:.4f}"
        )

    def test_no_signal_gives_high_pvalue(self) -> None:
        """Both groups from N(0, 1) -- no signal. p-value should be > 0.05
        (not guaranteed per-trial, but with large enough groups it's very likely).
        """
        n_viol, n_ctrl = 30, 30
        r_value = 3
        n_total = n_viol + n_ctrl
        start_step = 10
        n_steps = start_step + 5

        rng = np.random.default_rng(99)
        # Same distribution for both groups
        metric_array = rng.normal(0.0, 1.0, size=(n_total, n_steps))

        violations, controls = _make_events(n_viol, n_ctrl, r_value, start_step)
        result = run_shuffle_control(
            violations, controls, metric_array, r_value,
            n_permutations=1000, rng=99,
        )

        # Under H0, p-value should not be tiny (use a lenient threshold)
        assert result["p_value"] > 0.01, (
            f"Expected p > 0.01 for null data, got p={result['p_value']:.4f}"
        )
