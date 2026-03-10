"""Audit tests for exploratory/confirmatory split assignment (STAT-06).

Verifies that assign_split produces balanced 50/50 proportions within each
stratum (violation vs non-violation), preserves input violation ratios in
each split, is exactly deterministic with the same seed, and handles all
edge cases correctly (empty, single walk, all-violation, all-non-violation,
odd-count groups).

Pre-registration: docs/pre-registration.md Section 5.1 specifies SPLIT_SEED=2026,
50% exploratory / 50% confirmatory, stratified by violation status, deterministic
via np.random.default_rng.
"""

import numpy as np

from src.evaluation.split import (
    CONFIRMATORY,
    EXPLORATORY,
    SPLIT_SEED,
    assign_split,
)


class TestProportionVerification:
    """Verify 50/50 split within each stratum and overall."""

    def test_balanced_strata_40v_60nv(self) -> None:
        """40 violations (failure_index >= 0) and 60 non-violations (failure_index = -1).
        Each stratum should be split exactly in half via floor division:
        - Violations: 40 // 2 = 20 exploratory, 20 confirmatory
        - Non-violations: 60 // 2 = 30 exploratory, 30 confirmatory
        - Overall: 50 exploratory, 50 confirmatory
        """
        failure_index = np.concatenate([
            np.arange(40, dtype=int),      # 40 violations (values 0..39)
            np.full(60, -1, dtype=int),     # 60 non-violations
        ])

        result = assign_split(failure_index)

        assert len(result) == 100

        # Count within violation stratum
        violation_mask = failure_index >= 0
        viol_splits = result[violation_mask]
        viol_exp = np.sum(viol_splits == EXPLORATORY)
        viol_conf = np.sum(viol_splits == CONFIRMATORY)
        assert viol_exp == 20, f"Expected 20 exploratory violations, got {viol_exp}"
        assert viol_conf == 20, f"Expected 20 confirmatory violations, got {viol_conf}"

        # Count within non-violation stratum
        nonviol_splits = result[~violation_mask]
        nonviol_exp = np.sum(nonviol_splits == EXPLORATORY)
        nonviol_conf = np.sum(nonviol_splits == CONFIRMATORY)
        assert nonviol_exp == 30, f"Expected 30 exploratory non-violations, got {nonviol_exp}"
        assert nonviol_conf == 30, f"Expected 30 confirmatory non-violations, got {nonviol_conf}"

        # Overall
        total_exp = np.sum(result == EXPLORATORY)
        total_conf = np.sum(result == CONFIRMATORY)
        assert total_exp == 50, f"Expected 50 total exploratory, got {total_exp}"
        assert total_conf == 50, f"Expected 50 total confirmatory, got {total_conf}"


class TestStratificationIndependence:
    """Verify violation/non-violation ratio is preserved in EACH split."""

    def test_ratio_preserved_in_each_split(self) -> None:
        """Input: 40% violations (40/100). Each split should have 40% violations.
        - Exploratory: 20 violations / 50 total = 40%
        - Confirmatory: 20 violations / 50 total = 40%
        Both match the input ratio of 40/100 = 40%.
        """
        failure_index = np.concatenate([
            np.arange(40, dtype=int),
            np.full(60, -1, dtype=int),
        ])

        result = assign_split(failure_index)

        violation_mask = failure_index >= 0
        exp_mask = result == EXPLORATORY
        conf_mask = result == CONFIRMATORY

        # Violation rate in exploratory split
        exp_violations = np.sum(violation_mask & exp_mask)
        exp_total = np.sum(exp_mask)
        exp_ratio = exp_violations / exp_total
        assert abs(exp_ratio - 0.4) < 1e-10, (
            f"Exploratory violation ratio {exp_ratio} != 0.4"
        )

        # Violation rate in confirmatory split
        conf_violations = np.sum(violation_mask & conf_mask)
        conf_total = np.sum(conf_mask)
        conf_ratio = conf_violations / conf_total
        assert abs(conf_ratio - 0.4) < 1e-10, (
            f"Confirmatory violation ratio {conf_ratio} != 0.4"
        )


class TestExactDeterminism:
    """Verify exact reproducibility with same seed and sensitivity to different seeds."""

    def test_identical_results_same_seed(self) -> None:
        """Two calls with identical failure_index and split_seed=2026 must produce
        element-by-element identical results.
        """
        failure_index = np.concatenate([
            np.arange(30, dtype=int),
            np.full(70, -1, dtype=int),
        ])

        result1 = assign_split(failure_index, split_seed=2026)
        result2 = assign_split(failure_index, split_seed=2026)

        assert np.array_equal(result1, result2), (
            "Two calls with same input and seed must be element-by-element identical"
        )

    def test_different_seed_different_results(self) -> None:
        """Different seeds should produce different shuffles (and thus different assignments).
        With high probability, at least one walk will be assigned differently.
        """
        failure_index = np.concatenate([
            np.arange(50, dtype=int),
            np.full(50, -1, dtype=int),
        ])

        result_2026 = assign_split(failure_index, split_seed=2026)
        result_9999 = assign_split(failure_index, split_seed=9999)

        # With 100 elements, the probability of identical assignments with
        # different seeds is astronomically small
        assert not np.array_equal(result_2026, result_9999), (
            "Different seeds should produce different assignments"
        )

    def test_determinism_across_multiple_calls(self) -> None:
        """Call assign_split 5 times with the same seed; all must be identical.
        This verifies no external state leaks between calls.
        """
        failure_index = np.array([0, 1, -1, -1, 2, -1, 3, -1, -1, -1])
        results = [assign_split(failure_index, split_seed=42) for _ in range(5)]
        for i in range(1, 5):
            assert np.array_equal(results[0], results[i]), (
                f"Call {i} differs from call 0"
            )


class TestEdgeCases:
    """Verify correct handling of boundary and degenerate inputs."""

    def test_empty_array(self) -> None:
        """Empty failure_index should return empty array, no errors."""
        result = assign_split(np.array([], dtype=int))
        assert len(result) == 0, f"Expected empty result, got length {len(result)}"
        assert result.dtype.kind == "U", f"Expected string dtype, got {result.dtype}"

    def test_single_walk_violation(self) -> None:
        """Single violation walk: assigned to one split without error.
        With 1 element, floor(1/2) = 0 exploratory, 1 confirmatory.
        """
        result = assign_split(np.array([5]))
        assert len(result) == 1
        assert result[0] in (EXPLORATORY, CONFIRMATORY), (
            f"Expected exploratory or confirmatory, got '{result[0]}'"
        )

    def test_single_walk_non_violation(self) -> None:
        """Single non-violation walk: assigned to one split without error.
        With 1 element, floor(1/2) = 0 exploratory, 1 confirmatory.
        """
        result = assign_split(np.array([-1]))
        assert len(result) == 1
        assert result[0] in (EXPLORATORY, CONFIRMATORY), (
            f"Expected exploratory or confirmatory, got '{result[0]}'"
        )

    def test_all_violations(self) -> None:
        """All 4 walks are violations: split 2/2 within the violation stratum.
        No non-violation pool to split.
        """
        result = assign_split(np.array([0, 1, 2, 3]))
        assert len(result) == 4

        n_exp = np.sum(result == EXPLORATORY)
        n_conf = np.sum(result == CONFIRMATORY)
        # floor(4/2) = 2 exploratory, 2 confirmatory
        assert n_exp == 2, f"Expected 2 exploratory, got {n_exp}"
        assert n_conf == 2, f"Expected 2 confirmatory, got {n_conf}"

    def test_all_non_violations(self) -> None:
        """All 4 walks are non-violations: split 2/2 within the non-violation stratum.
        No violation pool to split.
        """
        result = assign_split(np.array([-1, -1, -1, -1]))
        assert len(result) == 4

        n_exp = np.sum(result == EXPLORATORY)
        n_conf = np.sum(result == CONFIRMATORY)
        assert n_exp == 2, f"Expected 2 exploratory, got {n_exp}"
        assert n_conf == 2, f"Expected 2 confirmatory, got {n_conf}"

    def test_odd_count_violations(self) -> None:
        """3 violations: floor(3/2) = 1 exploratory, 2 confirmatory.
        Verifies the floor division behavior where first half gets exploratory
        and the rest get confirmatory.
        """
        result = assign_split(np.array([0, 1, 2]))
        assert len(result) == 3

        n_exp = np.sum(result == EXPLORATORY)
        n_conf = np.sum(result == CONFIRMATORY)
        # floor(3/2) = 1 => first 1 of shuffled gets exploratory, remaining 2 get confirmatory
        assert n_exp == 1, f"Expected 1 exploratory for odd count, got {n_exp}"
        assert n_conf == 2, f"Expected 2 confirmatory for odd count, got {n_conf}"

    def test_odd_count_non_violations(self) -> None:
        """5 non-violations: floor(5/2) = 2 exploratory, 3 confirmatory."""
        result = assign_split(np.array([-1, -1, -1, -1, -1]))
        assert len(result) == 5

        n_exp = np.sum(result == EXPLORATORY)
        n_conf = np.sum(result == CONFIRMATORY)
        assert n_exp == 2, f"Expected 2 exploratory for 5 non-violations, got {n_exp}"
        assert n_conf == 3, f"Expected 3 confirmatory for 5 non-violations, got {n_conf}"


class TestValueValidation:
    """Verify all output values are valid split labels."""

    def test_all_values_are_valid_labels(self) -> None:
        """Every element in the result must be either 'exploratory' or 'confirmatory'.
        No empty strings, no other values, no NaN.
        """
        failure_index = np.concatenate([
            np.arange(25, dtype=int),
            np.full(75, -1, dtype=int),
        ])

        result = assign_split(failure_index)

        valid_labels = {EXPLORATORY, CONFIRMATORY}
        for i, val in enumerate(result):
            assert val in valid_labels, (
                f"Element {i} has invalid value '{val}', expected one of {valid_labels}"
            )

    def test_no_empty_strings(self) -> None:
        """Verify no empty string assignments (could indicate uninitialized array slots)."""
        failure_index = np.array([0, -1, 1, -1, 2, -1])
        result = assign_split(failure_index)

        for i, val in enumerate(result):
            assert len(val) > 0, f"Element {i} is an empty string"


class TestSeedDocumentation:
    """Verify SPLIT_SEED default matches pre-registration Section 5.1."""

    def test_split_seed_is_2026(self) -> None:
        """SPLIT_SEED constant must be 2026, matching docs/pre-registration.md Section 5.1:
        'Seed: Fixed seed (2026) with np.random.default_rng'.
        """
        assert SPLIT_SEED == 2026, f"Expected SPLIT_SEED = 2026, got {SPLIT_SEED}"

    def test_default_seed_used_without_explicit_argument(self) -> None:
        """Calling assign_split without explicit seed should use SPLIT_SEED=2026.
        Verify by comparing default call with explicit seed=2026 call.
        """
        failure_index = np.concatenate([
            np.arange(30, dtype=int),
            np.full(70, -1, dtype=int),
        ])

        result_default = assign_split(failure_index)
        result_explicit = assign_split(failure_index, split_seed=2026)

        assert np.array_equal(result_default, result_explicit), (
            "Default call must use SPLIT_SEED=2026"
        )
