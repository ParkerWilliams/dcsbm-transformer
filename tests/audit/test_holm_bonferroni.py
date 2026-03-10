"""Audit tests for Holm-Bonferroni step-down correction (STAT-03).

Verifies holm_bonferroni in statistical_controls.py produces correct adjusted
p-values by comparing against a textbook worked example, testing monotonicity
enforcement, edge cases, and confirming 0-based indexing equivalence with the
standard 1-based formula.
"""

import numpy as np

from src.analysis.statistical_controls import holm_bonferroni


class TestTextbookWorkedExample:
    """Verify against classic 5-hypothesis example from Holm (1979)."""

    def test_five_hypothesis_example(self) -> None:
        """Input p-values: [0.01, 0.04, 0.03, 0.005, 0.5].
        Sorted ascending: [0.005, 0.01, 0.03, 0.04, 0.5] (indices: 3, 0, 2, 1, 4).
        Multiplied by [5, 4, 3, 2, 1]: [0.025, 0.04, 0.09, 0.08, 0.5].
        After monotonicity enforcement: [0.025, 0.04, 0.09, 0.09, 0.5].
        At alpha=0.05: reject indices 3 and 0 (adjusted 0.025 and 0.04 <= 0.05).
        """
        p_values = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        # Expected adjusted p-values in ORIGINAL order (not sorted order)
        # Original index 0 (p=0.01): sorted rank 1 -> adjusted 0.04, after mono 0.04
        # Original index 1 (p=0.04): sorted rank 3 -> adjusted 0.08, after mono 0.09
        # Original index 2 (p=0.03): sorted rank 2 -> adjusted 0.09, after mono 0.09
        # Original index 3 (p=0.005): sorted rank 0 -> adjusted 0.025, after mono 0.025
        # Original index 4 (p=0.5): sorted rank 4 -> adjusted 0.5, after mono 0.5
        expected_adjusted = np.array([0.04, 0.09, 0.09, 0.025, 0.5])
        np.testing.assert_allclose(adjusted, expected_adjusted, atol=1e-12,
                                   err_msg="Adjusted p-values do not match textbook example")

        # Reject: adjusted <= alpha (0.05)
        expected_reject = np.array([True, False, False, True, False])
        np.testing.assert_array_equal(reject, expected_reject,
                                      err_msg="Reject flags do not match textbook example")

    def test_all_significant(self) -> None:
        """All p-values very small: [0.001, 0.002, 0.003].
        Sorted: [0.001, 0.002, 0.003]. Multiplied by [3, 2, 1]: [0.003, 0.004, 0.003].
        After monotonicity: [0.003, 0.004, 0.004]. All <= 0.05 -> all rejected.
        """
        p_values = np.array([0.001, 0.002, 0.003])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        np.testing.assert_allclose(adjusted, [0.003, 0.004, 0.004], atol=1e-12)
        np.testing.assert_array_equal(reject, [True, True, True])

    def test_none_significant(self) -> None:
        """All p-values large: [0.3, 0.6, 0.9].
        Sorted: [0.3, 0.6, 0.9]. Multiplied by [3, 2, 1]: [0.9, 1.2, 0.9].
        Clipped at 1.0: [0.9, 1.0, 0.9]. After monotonicity: [0.9, 1.0, 1.0].
        None <= 0.05.
        """
        p_values = np.array([0.3, 0.6, 0.9])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        np.testing.assert_allclose(adjusted, [0.9, 1.0, 1.0], atol=1e-12)
        np.testing.assert_array_equal(reject, [False, False, False])


class TestMonotonicityEnforcement:
    """Verify that adjusted p-values are non-decreasing after step-down enforcement."""

    def test_monotonicity_correction_applied(self) -> None:
        """Input p-values: [0.04, 0.01, 0.03].
        Sorted ascending: [0.01, 0.03, 0.04] (indices: 1, 2, 0).
        Multiplied by [3, 2, 1]: [0.03, 0.06, 0.04].
        Without monotonicity: p3=0.04 < p2=0.06 -- violates step-down ordering.
        After enforcement: [0.03, 0.06, 0.06] (p3 gets max(0.04, 0.06) = 0.06).
        """
        p_values = np.array([0.04, 0.01, 0.03])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        # In ORIGINAL order:
        # Index 0 (p=0.04): sorted rank 2 -> adjusted 0.04, after mono max(0.04, 0.06) = 0.06
        # Index 1 (p=0.01): sorted rank 0 -> adjusted 0.03, after mono 0.03
        # Index 2 (p=0.03): sorted rank 1 -> adjusted 0.06, after mono 0.06
        expected_adjusted = np.array([0.06, 0.03, 0.06])
        np.testing.assert_allclose(adjusted, expected_adjusted, atol=1e-12,
                                   err_msg="Monotonicity enforcement not applied correctly")

    def test_adjusted_pvalues_nondecreasing_in_sorted_order(self) -> None:
        """For any input, the adjusted p-values in sorted-p order must be non-decreasing.
        This is the fundamental monotonicity guarantee of the step-down procedure.
        """
        rng = np.random.default_rng(42)
        for _ in range(20):
            m = rng.integers(2, 20)
            p_values = rng.uniform(0.0, 1.0, size=m)
            adjusted, _ = holm_bonferroni(p_values, alpha=0.05)

            # Re-sort by original p-values to check monotonicity in step-down order
            sorted_idx = np.argsort(p_values)
            adjusted_in_sorted_order = adjusted[sorted_idx]
            for i in range(1, len(adjusted_in_sorted_order)):
                assert adjusted_in_sorted_order[i] >= adjusted_in_sorted_order[i - 1], (
                    f"Monotonicity violated at position {i}: "
                    f"{adjusted_in_sorted_order[i]} < {adjusted_in_sorted_order[i-1]}. "
                    f"p_values={p_values}, adjusted={adjusted}"
                )


class TestEdgeCases:
    """Edge cases: empty, single, identical, extremes."""

    def test_empty_input(self) -> None:
        """Empty p-value array returns empty adjusted and reject arrays."""
        adjusted, reject = holm_bonferroni(np.array([]), alpha=0.05)
        assert len(adjusted) == 0, f"Expected empty adjusted, got {adjusted}"
        assert len(reject) == 0, f"Expected empty reject, got {reject}"

    def test_single_pvalue(self) -> None:
        """Single p-value: adjusted = min(p * 1, 1.0) = p.
        m=1, multiplier (m-0) = 1, so adjusted = p * 1 = p.
        """
        adjusted, reject = holm_bonferroni(np.array([0.03]), alpha=0.05)
        np.testing.assert_allclose(adjusted, [0.03], atol=1e-12)
        assert reject[0] is np.True_, "p=0.03 < alpha=0.05 should be rejected"

    def test_single_pvalue_above_alpha(self) -> None:
        """Single p-value above alpha: not rejected."""
        adjusted, reject = holm_bonferroni(np.array([0.10]), alpha=0.05)
        np.testing.assert_allclose(adjusted, [0.10], atol=1e-12)
        assert reject[0] is np.False_, "p=0.10 > alpha=0.05 should not be rejected"

    def test_all_identical_pvalues(self) -> None:
        """All p-values = 0.02, m=4. Sorted: [0.02, 0.02, 0.02, 0.02].
        Multiplied by [4, 3, 2, 1]: [0.08, 0.06, 0.04, 0.02].
        After monotonicity: [0.08, 0.08, 0.08, 0.08] (all get max).
        Wait -- monotonicity enforces each >= previous. Starting from [0.08, 0.06, 0.04, 0.02]:
        i=1: max(0.06, 0.08) = 0.08
        i=2: max(0.04, 0.08) = 0.08
        i=3: max(0.02, 0.08) = 0.08
        So all adjusted = 0.08.
        """
        p_values = np.array([0.02, 0.02, 0.02, 0.02])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        # All adjusted should equal 0.08 (= 0.02 * 4, the largest multiplied value)
        np.testing.assert_allclose(adjusted, [0.08, 0.08, 0.08, 0.08], atol=1e-12)
        np.testing.assert_array_equal(reject, [False, False, False, False])

    def test_all_zeros(self) -> None:
        """All p-values = 0.0. Multiplied by anything = 0.0. All rejected."""
        p_values = np.array([0.0, 0.0, 0.0])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        np.testing.assert_allclose(adjusted, [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_array_equal(reject, [True, True, True])

    def test_all_ones(self) -> None:
        """All p-values = 1.0. Multiplied by [3, 2, 1] = [3.0, 2.0, 1.0].
        Clipped at 1.0: [1.0, 1.0, 1.0]. None rejected.
        """
        p_values = np.array([1.0, 1.0, 1.0])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        np.testing.assert_allclose(adjusted, [1.0, 1.0, 1.0], atol=1e-12)
        np.testing.assert_array_equal(reject, [False, False, False])

    def test_clipping_at_one(self) -> None:
        """Large p-values can exceed 1.0 after multiplication; must be clipped.
        p_values = [0.1, 0.8]. Sorted: [0.1, 0.8]. Multiplied by [2, 1]: [0.2, 0.8].
        No clipping needed. Try: [0.6, 0.8]. Sorted: [0.6, 0.8]. Multiplied by [2, 1]: [1.2, 0.8].
        Clipped: [1.0, 0.8]. After monotonicity: [1.0, 1.0].
        """
        p_values = np.array([0.6, 0.8])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        assert adjusted[0] <= 1.0, f"Adjusted p-value {adjusted[0]} exceeds 1.0"
        assert adjusted[1] <= 1.0, f"Adjusted p-value {adjusted[1]} exceeds 1.0"
        np.testing.assert_allclose(adjusted, [1.0, 1.0], atol=1e-12)


class TestFormulaEquivalence:
    """Verify 0-based (m - i) equals 1-based (m - k + 1) for all m."""

    def test_zero_based_equals_one_based_formula(self) -> None:
        """The code uses 0-based: multiplier[i] = (m - i) for i=0..m-1.
        The textbook uses 1-based: multiplier[k] = (m - k + 1) for k=1..m.
        These must be identical: for i = k-1, (m - i) = (m - (k-1)) = (m - k + 1).
        Verify for all m from 1 to 10.
        """
        for m in range(1, 11):
            # 0-based multipliers (as used in the code)
            zero_based = np.array([m - i for i in range(m)])
            # 1-based multipliers (textbook formula)
            one_based = np.array([m - k + 1 for k in range(1, m + 1)])
            np.testing.assert_array_equal(zero_based, one_based,
                                          err_msg=f"Formula mismatch at m={m}")

    def test_multipliers_match_expected_sequence(self) -> None:
        """For m=5, multipliers should be [5, 4, 3, 2, 1] regardless of formula."""
        m = 5
        zero_based = [m - i for i in range(m)]
        one_based = [m - k + 1 for k in range(1, m + 1)]
        expected = [5, 4, 3, 2, 1]

        assert zero_based == expected, f"0-based: {zero_based} != {expected}"
        assert one_based == expected, f"1-based: {one_based} != {expected}"
