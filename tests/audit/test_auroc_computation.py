"""Audit tests for AUROC computation formula (AUROC-01).

Verifies auroc_from_groups computes the correct rank-based P(X_violated > X_followed)
by comparing against sklearn.metrics.roc_auc_score and scipy.stats.mannwhitneyu as
oracle references. Covers edge cases (empty, tied, perfect separation) and analytic
distributions with known theoretical AUROC.
"""

import numpy as np
from scipy.stats import mannwhitneyu, norm, rankdata
from sklearn.metrics import roc_auc_score

from src.analysis.auroc_horizon import auroc_from_groups


class TestAurocVsSklearn:
    """Verify auroc_from_groups matches sklearn.metrics.roc_auc_score on identical inputs."""

    def test_overlapping_distributions(self) -> None:
        """Overlapping violations=[3,4,5,6,7] vs controls=[1,2,2.5,3.5,4.5].
        Both implementations should compute identical rank-based AUROC.
        """
        violations = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        controls = np.array([1.0, 2.0, 2.5, 3.5, 4.5])

        our_auroc = auroc_from_groups(violations, controls)

        # sklearn expects (y_true, y_score): 1=violation, 0=control
        y_true = np.array([1] * 5 + [0] * 5)
        y_score = np.concatenate([violations, controls])
        sklearn_auroc = roc_auc_score(y_true, y_score)

        assert abs(our_auroc - sklearn_auroc) < 1e-10, (
            f"Our AUROC {our_auroc} != sklearn {sklearn_auroc}"
        )

    def test_widely_separated_distributions(self) -> None:
        """violations=[10,11,12] vs controls=[1,2,3]: no overlap => AUROC = 1.0.
        Every violation exceeds every control, so P(X_viol > X_ctrl) = 1.
        """
        violations = np.array([10.0, 11.0, 12.0])
        controls = np.array([1.0, 2.0, 3.0])

        our_auroc = auroc_from_groups(violations, controls)

        y_true = np.array([1] * 3 + [0] * 3)
        y_score = np.concatenate([violations, controls])
        sklearn_auroc = roc_auc_score(y_true, y_score)

        assert our_auroc == 1.0, f"Expected 1.0, got {our_auroc}"
        assert sklearn_auroc == 1.0, f"sklearn expected 1.0, got {sklearn_auroc}"

    def test_identical_values(self) -> None:
        """violations=[5,5,5] vs controls=[5,5,5]: all tied => AUROC = 0.5.
        With all values equal, midrank gives each element the same rank,
        so the probability of a violation exceeding a control is 0.5 (chance).
        """
        violations = np.array([5.0, 5.0, 5.0])
        controls = np.array([5.0, 5.0, 5.0])

        our_auroc = auroc_from_groups(violations, controls)

        y_true = np.array([1] * 3 + [0] * 3)
        y_score = np.concatenate([violations, controls])
        sklearn_auroc = roc_auc_score(y_true, y_score)

        assert abs(our_auroc - 0.5) < 1e-10, f"Expected 0.5, got {our_auroc}"
        assert abs(sklearn_auroc - 0.5) < 1e-10, f"sklearn expected 0.5, got {sklearn_auroc}"

    def test_reversed_distributions(self) -> None:
        """violations=[1,2,3] vs controls=[10,11,12]: all violations < all controls => AUROC = 0.0.
        P(X_viol > X_ctrl) = 0 when every violation is smaller than every control.
        """
        violations = np.array([1.0, 2.0, 3.0])
        controls = np.array([10.0, 11.0, 12.0])

        our_auroc = auroc_from_groups(violations, controls)

        y_true = np.array([1] * 3 + [0] * 3)
        y_score = np.concatenate([violations, controls])
        sklearn_auroc = roc_auc_score(y_true, y_score)

        assert our_auroc == 0.0, f"Expected 0.0, got {our_auroc}"
        assert sklearn_auroc == 0.0, f"sklearn expected 0.0, got {sklearn_auroc}"


class TestAurocVsMannWhitneyU:
    """Verify auroc_from_groups matches Mann-Whitney U / (n_v * n_c)."""

    def test_overlapping_distributions_mw(self) -> None:
        """Same overlapping distributions as sklearn test.
        Mann-Whitney U = sum of (violation rank - expected rank under H0),
        and AUROC = U / (n_v * n_c).

        Per Pitfall 4 in RESEARCH.md: use alternative='greater' so violations
        having higher ranks yields the U statistic that directly converts to AUROC.
        """
        violations = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        controls = np.array([1.0, 2.0, 2.5, 3.5, 4.5])

        our_auroc = auroc_from_groups(violations, controls)

        n_v, n_c = len(violations), len(controls)
        # alternative='greater' tests H1: violations tend to be larger
        U, _ = mannwhitneyu(violations, controls, alternative="greater")
        mw_auroc = U / (n_v * n_c)

        assert abs(our_auroc - mw_auroc) < 1e-10, (
            f"Our AUROC {our_auroc} != MWU AUROC {mw_auroc}"
        )

    def test_asymmetric_overlap_mw(self) -> None:
        """violations=[2,4,6,8] vs controls=[1,3,5,7,9]: interleaved values.
        Tests Mann-Whitney equivalence with unequal group sizes.
        """
        violations = np.array([2.0, 4.0, 6.0, 8.0])
        controls = np.array([1.0, 3.0, 5.0, 7.0, 9.0])

        our_auroc = auroc_from_groups(violations, controls)

        n_v, n_c = len(violations), len(controls)
        U, _ = mannwhitneyu(violations, controls, alternative="greater")
        mw_auroc = U / (n_v * n_c)

        assert abs(our_auroc - mw_auroc) < 1e-10, (
            f"Our AUROC {our_auroc} != MWU AUROC {mw_auroc}"
        )

    def test_three_way_comparison(self) -> None:
        """For the same inputs, verify our_auroc == sklearn_auroc == mw_auroc.
        All three methods (rank-sum, ROC curve, Mann-Whitney U) are mathematically
        equivalent and must produce identical results within floating-point tolerance.
        """
        violations = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        controls = np.array([1.0, 2.0, 2.5, 3.5, 4.5])

        # Our implementation
        our_auroc = auroc_from_groups(violations, controls)

        # sklearn reference
        y_true = np.array([1] * len(violations) + [0] * len(controls))
        y_score = np.concatenate([violations, controls])
        sklearn_auroc = roc_auc_score(y_true, y_score)

        # Mann-Whitney U reference
        n_v, n_c = len(violations), len(controls)
        U, _ = mannwhitneyu(violations, controls, alternative="greater")
        mw_auroc = U / (n_v * n_c)

        # Three-way agreement within 1e-10
        assert abs(our_auroc - sklearn_auroc) < 1e-10, (
            f"Ours {our_auroc} != sklearn {sklearn_auroc}"
        )
        assert abs(our_auroc - mw_auroc) < 1e-10, (
            f"Ours {our_auroc} != MWU {mw_auroc}"
        )
        assert abs(sklearn_auroc - mw_auroc) < 1e-10, (
            f"sklearn {sklearn_auroc} != MWU {mw_auroc}"
        )


class TestAurocAnalyticDistributions:
    """Verify AUROC on distributions with known theoretical values."""

    def test_two_gaussians_known_separation(self) -> None:
        """N(mu1=2, sigma=1) vs N(mu2=0, sigma=1), n=10000 each.
        Theoretical AUROC = Phi(delta / sqrt(2)) where delta = (mu1-mu2)/sigma.
        delta = 2, so AUROC = Phi(2/sqrt(2)) = Phi(sqrt(2)) ~ 0.9214.

        With n=10000 per group, the sampling error should be well within 0.02.
        """
        rng = np.random.default_rng(42)
        violations = rng.normal(loc=2.0, scale=1.0, size=10000)
        controls = rng.normal(loc=0.0, scale=1.0, size=10000)

        our_auroc = auroc_from_groups(violations, controls)

        # Theoretical AUROC for two equal-variance Gaussians separated by delta=2
        delta = 2.0
        theoretical_auroc = float(norm.cdf(delta / np.sqrt(2)))  # Phi(sqrt(2)) ~ 0.9214

        assert abs(our_auroc - theoretical_auroc) < 0.02, (
            f"Our AUROC {our_auroc:.4f} differs from theoretical {theoretical_auroc:.4f} "
            f"by more than 0.02"
        )

    def test_perfect_separation_with_gap(self) -> None:
        """violations in [10,20] vs controls in [0,5]: disjoint ranges => AUROC = 1.0.
        Every violation exceeds every control by a wide margin.
        """
        rng = np.random.default_rng(123)
        violations = rng.uniform(10.0, 20.0, size=100)
        controls = rng.uniform(0.0, 5.0, size=100)

        our_auroc = auroc_from_groups(violations, controls)
        assert our_auroc == 1.0, f"Expected 1.0, got {our_auroc}"


class TestAurocEdgeCases:
    """Verify edge case handling: empty groups, single elements, ties."""

    def test_empty_violations_returns_nan(self) -> None:
        """Empty violations array should return NaN (no meaningful AUROC)."""
        violations = np.array([])
        controls = np.array([1.0, 2.0, 3.0])

        result = auroc_from_groups(violations, controls)
        assert np.isnan(result), f"Expected NaN for empty violations, got {result}"

    def test_empty_controls_returns_nan(self) -> None:
        """Empty controls array should return NaN (no meaningful AUROC)."""
        violations = np.array([1.0, 2.0, 3.0])
        controls = np.array([])

        result = auroc_from_groups(violations, controls)
        assert np.isnan(result), f"Expected NaN for empty controls, got {result}"

    def test_single_element_each_group(self) -> None:
        """violations=[5] vs controls=[3]: 5 > 3, so P(X_viol > X_ctrl) = 1.0.
        Even with n=1 per group, the rank-based AUROC should give a valid answer.
        """
        violations = np.array([5.0])
        controls = np.array([3.0])

        result = auroc_from_groups(violations, controls)
        assert result == 1.0, f"Expected 1.0 (5 > 3), got {result}"

    def test_tied_values_across_groups(self) -> None:
        """violations=[3,3,3] vs controls=[3,3,3]: all tied => AUROC = 0.5.
        Midrank assigns rank (1+2+3+4+5+6)/6 = 3.5 to all six elements.
        rank_sum for violations = 3 * 3.5 = 10.5.
        AUROC = (10.5 - 3*4/2) / (3*3) = (10.5 - 6) / 9 = 4.5/9 = 0.5.
        """
        violations = np.array([3.0, 3.0, 3.0])
        controls = np.array([3.0, 3.0, 3.0])

        result = auroc_from_groups(violations, controls)
        assert abs(result - 0.5) < 1e-10, f"Expected 0.5, got {result}"

        # Verify scipy.rankdata assigns midranks as expected
        combined = np.concatenate([violations, controls])
        ranks = rankdata(combined)
        # All 6 values are equal, so midrank = (1+2+3+4+5+6)/6 = 3.5
        np.testing.assert_allclose(ranks, [3.5] * 6, atol=1e-10)

    def test_partial_ties_across_groups(self) -> None:
        """violations=[3,3,5] vs controls=[3,4,4]: partial ties.
        Combined=[3,3,5,3,4,4], sorted ranks: 3,3,3->rank 2; 4,4->rank 4.5; 5->rank 6.
        Violation ranks: r(3)=2, r(3)=2, r(5)=6. rank_sum = 10.
        AUROC = (10 - 3*4/2) / (3*3) = (10-6)/9 = 4/9 ~ 0.4444.
        Verify this matches sklearn.
        """
        violations = np.array([3.0, 3.0, 5.0])
        controls = np.array([3.0, 4.0, 4.0])

        our_auroc = auroc_from_groups(violations, controls)

        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_score = np.concatenate([violations, controls])
        sklearn_auroc = roc_auc_score(y_true, y_score)

        assert abs(our_auroc - sklearn_auroc) < 1e-10, (
            f"Our AUROC {our_auroc} != sklearn {sklearn_auroc} for partial ties"
        )
