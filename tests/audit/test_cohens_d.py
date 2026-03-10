"""Audit tests for Cohen's d effect size (STAT-04).

Verifies cohens_d in statistical_controls.py uses the correct pooled standard
deviation formula, returns correct sign, handles NaN guard edge cases, and
matches hand-calculated values exactly.
"""

import numpy as np

from src.analysis.statistical_controls import cohens_d


class TestHandCalculation:
    """Verify Cohen's d matches hand-computed values using pooled std formula."""

    def test_textbook_example(self) -> None:
        """group1 = [2, 4, 6] (mean=4, var=4, n=3)
        group2 = [1, 2, 3] (mean=2, var=1, n=3).
        Pooled std = sqrt(((3-1)*4 + (3-1)*1) / (3+3-2)) = sqrt((8+2)/4) = sqrt(2.5) ~ 1.5811.
        Cohen's d = (4 - 2) / sqrt(2.5) = 2 / 1.5811 ~ 1.2649.
        """
        group1 = np.array([2.0, 4.0, 6.0])
        group2 = np.array([1.0, 2.0, 3.0])

        result = cohens_d(group1, group2)

        # Hand calculation
        mean1, mean2 = 4.0, 2.0
        var1, var2 = 4.0, 1.0  # sample variance (ddof=1)
        n1, n2 = 3, 3
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        expected_d = (mean1 - mean2) / pooled_std

        assert abs(result - expected_d) < 1e-10, (
            f"Cohen's d = {result}, expected {expected_d}"
        )
        # Also verify against the approximate value
        assert abs(result - 1.2649110640673518) < 1e-4, (
            f"Cohen's d = {result}, expected ~1.2649"
        )

    def test_exact_case_unit_pooled_std(self) -> None:
        """group1 = [0, 2] (mean=1, var=2), group2 = [0, 0] (mean=0, var=0).
        Pooled std = sqrt(((2-1)*2 + (2-1)*0) / (2+2-2)) = sqrt(2/2) = 1.0.
        d = (1 - 0) / 1.0 = 1.0 exactly.
        """
        group1 = np.array([0.0, 2.0])
        group2 = np.array([0.0, 0.0])

        result = cohens_d(group1, group2)

        assert result == 1.0, f"Expected exactly 1.0, got {result}"

    def test_equal_groups(self) -> None:
        """group1 = [1, 2, 3, 4, 5], group2 = [1, 2, 3, 4, 5].
        Same means, so d = 0.0 regardless of pooled_std.
        """
        group1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = cohens_d(group1, group2)
        assert result == 0.0, f"Expected d=0.0 for equal groups, got {result}"


class TestSignConvention:
    """Verify sign: positive d when group1 > group2, negative when group1 < group2."""

    def test_positive_d_when_group1_higher(self) -> None:
        """group1 has higher mean than group2 -> d > 0.
        d = (mean(group1) - mean(group2)) / pooled_std.
        """
        group1 = np.array([10.0, 11.0, 12.0])
        group2 = np.array([1.0, 2.0, 3.0])

        result = cohens_d(group1, group2)
        assert result > 0, f"Expected d > 0 when group1 > group2, got {result}"

    def test_negative_d_when_group1_lower(self) -> None:
        """group1 has lower mean than group2 -> d < 0."""
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([10.0, 11.0, 12.0])

        result = cohens_d(group1, group2)
        assert result < 0, f"Expected d < 0 when group1 < group2, got {result}"

    def test_sign_reversal_symmetry(self) -> None:
        """Swapping groups should negate d: d(g1, g2) = -d(g2, g1)."""
        group1 = np.array([2.0, 4.0, 6.0, 8.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0])

        d_forward = cohens_d(group1, group2)
        d_reverse = cohens_d(group2, group1)

        assert abs(d_forward + d_reverse) < 1e-10, (
            f"Sign reversal symmetry violated: d(g1,g2)={d_forward}, d(g2,g1)={d_reverse}"
        )


class TestNaNGuardEdgeCases:
    """Verify NaN is returned for degenerate inputs per the guard clauses."""

    def test_single_element_group1_returns_nan(self) -> None:
        """n1 < 2: pooled_std undefined (division by n1+n2-2 = 1 is ok, but
        variance of single element is undefined with ddof=1). Returns NaN.
        """
        result = cohens_d(np.array([1.0]), np.array([2.0, 3.0]))
        assert np.isnan(result), f"Expected NaN for n1=1, got {result}"

    def test_single_element_group2_returns_nan(self) -> None:
        """n2 < 2: returns NaN."""
        result = cohens_d(np.array([2.0, 3.0]), np.array([1.0]))
        assert np.isnan(result), f"Expected NaN for n2=1, got {result}"

    def test_identical_values_returns_nan(self) -> None:
        """All values identical: pooled_std = 0 < 1e-12, returns NaN.
        d would be 0/0 which is undefined.
        """
        result = cohens_d(
            np.array([5.0, 5.0, 5.0]),
            np.array([5.0, 5.0, 5.0]),
        )
        assert np.isnan(result), f"Expected NaN for zero pooled_std, got {result}"

    def test_identical_within_group_different_means_returns_nan(self) -> None:
        """group1 = [3, 3, 3], group2 = [7, 7, 7]. Both have var=0.
        pooled_std = sqrt(0) = 0 < 1e-12. Returns NaN despite different means.
        """
        result = cohens_d(
            np.array([3.0, 3.0, 3.0]),
            np.array([7.0, 7.0, 7.0]),
        )
        assert np.isnan(result), f"Expected NaN for zero pooled_std, got {result}"

    def test_normal_groups_return_finite(self) -> None:
        """Normal groups with variance should return a finite d.
        group1 = [1,2,3,4,5] (mean=3), group2 = [2,3,4,5,6] (mean=4).
        Both have nonzero variance, so d should be finite.
        """
        result = cohens_d(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
        )
        assert np.isfinite(result), f"Expected finite d, got {result}"
        # d should be negative (group1 mean < group2 mean)
        assert result < 0, f"Expected d < 0 (group1 mean < group2 mean), got {result}"


class TestIndependentPooledStd:
    """Recompute pooled std independently and compare against cohens_d output."""

    def test_independent_pooled_std_computation(self) -> None:
        """Back-calculate pooled_std from cohens_d output and verify it matches
        an independent computation using a completely separate code path.
        """
        group1 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        group2 = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        d = cohens_d(group1, group2)

        # Independent pooled std computation (manual loop, not numpy var)
        n1, n2 = len(group1), len(group2)
        mean1 = sum(group1) / n1
        mean2 = sum(group2) / n2
        ss1 = sum((x - mean1) ** 2 for x in group1)  # sum of squares
        ss2 = sum((x - mean2) ** 2 for x in group2)
        pooled_std_manual = np.sqrt((ss1 + ss2) / (n1 + n2 - 2))

        # Back-calculate pooled std from d
        expected_d = (mean1 - mean2) / pooled_std_manual

        assert abs(d - expected_d) < 1e-10, (
            f"d={d} does not match independent computation {expected_d}. "
            f"pooled_std_manual={pooled_std_manual}"
        )

    def test_pooled_std_formula_matches_numpy(self) -> None:
        """Verify the pooled_std formula used in cohens_d matches the standard
        definition: sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2)) where
        s^2 = np.var(x, ddof=1).
        """
        rng = np.random.default_rng(42)
        group1 = rng.normal(5.0, 2.0, size=20)
        group2 = rng.normal(3.0, 1.5, size=25)

        d = cohens_d(group1, group2)

        # Compute expected d from first principles
        n1, n2 = len(group1), len(group2)
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        expected_d = (np.mean(group1) - np.mean(group2)) / pooled_std

        assert abs(d - expected_d) < 1e-10, (
            f"d={d} != expected {expected_d}"
        )
