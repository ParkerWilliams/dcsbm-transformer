"""Audit tests for BCa bootstrap confidence intervals (STAT-02).

Verifies auroc_with_bootstrap_ci in statistical_controls.py correctly delegates
to scipy.stats.bootstrap with method='BCa', passes confidence_level, produces
correct CIs for known-answer cases, and handles NaN/edge cases properly.
"""

from unittest.mock import MagicMock, patch

import numpy as np

from src.analysis.statistical_controls import auroc_with_bootstrap_ci


class TestBCaDelegation:
    """Verify auroc_with_bootstrap_ci delegates to scipy.stats.bootstrap with correct args."""

    def test_calls_bootstrap_with_bca_method(self) -> None:
        """Patch scipy.stats.bootstrap and verify it is called with method='BCa'.
        The function should attempt BCa first before any fallback.
        """
        violations = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        controls = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        # Create a mock that returns a result-like object
        mock_ci = MagicMock()
        mock_ci.confidence_interval.low = 0.9
        mock_ci.confidence_interval.high = 1.0

        with patch("src.analysis.statistical_controls.bootstrap", return_value=mock_ci) as mock_boot:
            auroc_with_bootstrap_ci(violations, controls, n_resamples=100, rng=42)

            # First call should use BCa
            assert mock_boot.call_count >= 1, "bootstrap was not called"
            first_call_kwargs = mock_boot.call_args_list[0]
            # Check keyword arguments or positional args
            if first_call_kwargs.kwargs:
                assert first_call_kwargs.kwargs.get("method") == "BCa", (
                    f"First call method={first_call_kwargs.kwargs.get('method')}, expected 'BCa'"
                )
            else:
                # method might be passed as positional -- check all args
                all_args = str(first_call_kwargs)
                assert "BCa" in all_args, f"BCa not found in call args: {all_args}"

    def test_confidence_level_propagates(self) -> None:
        """Pass confidence_level=0.99 and verify it reaches scipy.stats.bootstrap."""
        violations = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        controls = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        mock_ci = MagicMock()
        mock_ci.confidence_interval.low = 0.9
        mock_ci.confidence_interval.high = 1.0

        with patch("src.analysis.statistical_controls.bootstrap", return_value=mock_ci) as mock_boot:
            auroc_with_bootstrap_ci(
                violations, controls, n_resamples=100,
                confidence_level=0.99, rng=42,
            )

            first_call_kwargs = mock_boot.call_args_list[0]
            assert first_call_kwargs.kwargs.get("confidence_level") == 0.99, (
                f"confidence_level not propagated: {first_call_kwargs.kwargs}"
            )

    def test_vectorized_true_passed(self) -> None:
        """Verify vectorized=True is passed to scipy.stats.bootstrap for efficiency."""
        violations = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        controls = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        mock_ci = MagicMock()
        mock_ci.confidence_interval.low = 0.9
        mock_ci.confidence_interval.high = 1.0

        with patch("src.analysis.statistical_controls.bootstrap", return_value=mock_ci) as mock_boot:
            auroc_with_bootstrap_ci(violations, controls, n_resamples=100, rng=42)

            first_call_kwargs = mock_boot.call_args_list[0]
            assert first_call_kwargs.kwargs.get("vectorized") is True, (
                f"vectorized not True: {first_call_kwargs.kwargs}"
            )


class TestKnownAnswerEndToEnd:
    """End-to-end tests with analytically known or predictable AUROC values."""

    def test_perfect_separation_auroc_one(self) -> None:
        """violations=[10,11,12,13,14] vs controls=[0,1,2,3,4]: perfect separation.
        Point estimate = 1.0 (every violation > every control).
        CI should contain 1.0 (ci_high >= 1.0 or very close).
        """
        violations = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        controls = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        point, ci_low, ci_high = auroc_with_bootstrap_ci(
            violations, controls, n_resamples=5000, rng=42,
        )

        # Point estimate must be exactly 1.0 for perfect separation
        assert point == 1.0, f"Expected point=1.0, got {point}"

        # CI should bracket 1.0 (for perfect separation, ci_low should be close to 1.0)
        assert ci_low >= 0.5, f"ci_low={ci_low} is unexpectedly low for perfect separation"

    def test_overlapping_groups_reasonable_ci(self) -> None:
        """Both groups from N(0,1), n=50 each. AUROC ~ 0.5.
        CI should be a nontrivial interval (ci_low < ci_high).
        """
        rng = np.random.default_rng(42)
        violations = rng.normal(0.0, 1.0, size=50)
        controls = rng.normal(0.0, 1.0, size=50)

        point, ci_low, ci_high = auroc_with_bootstrap_ci(
            violations, controls, n_resamples=5000, rng=42,
        )

        # Point estimate should be near 0.5 for equal distributions
        assert 0.3 < point < 0.7, f"Expected AUROC ~0.5, got {point}"

        # CI width should be > 0 (not degenerate)
        assert ci_low < ci_high, (
            f"CI is degenerate: ci_low={ci_low}, ci_high={ci_high}"
        )

        # CI should contain the point estimate
        assert ci_low <= point <= ci_high, (
            f"Point {point} outside CI [{ci_low}, {ci_high}]"
        )

    def test_moderate_separation(self) -> None:
        """violations from N(2,1), controls from N(0,1), n=30 each.
        Expected AUROC ~ Phi(2/sqrt(2)) ~ 0.92. CI should not include 0.5.
        """
        rng = np.random.default_rng(42)
        violations = rng.normal(2.0, 1.0, size=30)
        controls = rng.normal(0.0, 1.0, size=30)

        point, ci_low, ci_high = auroc_with_bootstrap_ci(
            violations, controls, n_resamples=5000, rng=42,
        )

        # With d=2 separation, AUROC should be well above 0.5
        assert point > 0.7, f"Expected AUROC > 0.7 for d=2, got {point}"
        # CI lower bound should be above chance
        assert ci_low > 0.5, f"CI lower {ci_low} should be > 0.5 for real signal"


class TestNaNHandling:
    """Edge cases producing NaN or requiring special handling."""

    def test_empty_violations_returns_nan(self) -> None:
        """Empty violation array: AUROC is undefined, should return (NaN, NaN, NaN)."""
        violations = np.array([])
        controls = np.array([1.0, 2.0, 3.0])

        point, ci_low, ci_high = auroc_with_bootstrap_ci(violations, controls)

        assert np.isnan(point), f"Expected NaN point, got {point}"
        assert np.isnan(ci_low), f"Expected NaN ci_low, got {ci_low}"
        assert np.isnan(ci_high), f"Expected NaN ci_high, got {ci_high}"

    def test_empty_controls_returns_nan(self) -> None:
        """Empty control array: AUROC is undefined, should return (NaN, NaN, NaN)."""
        violations = np.array([1.0, 2.0, 3.0])
        controls = np.array([])

        point, ci_low, ci_high = auroc_with_bootstrap_ci(violations, controls)

        assert np.isnan(point), f"Expected NaN point, got {point}"
        assert np.isnan(ci_low), f"Expected NaN ci_low, got {ci_low}"
        assert np.isnan(ci_high), f"Expected NaN ci_high, got {ci_high}"

    def test_single_element_groups(self) -> None:
        """Single-element groups: point estimate may be valid, CIs may be NaN.
        violations=[5] vs controls=[3]: AUROC = 1.0. Bootstrap may struggle
        with n=1, producing NaN CIs (acceptable).
        """
        violations = np.array([5.0])
        controls = np.array([3.0])

        point, ci_low, ci_high = auroc_with_bootstrap_ci(
            violations, controls, n_resamples=1000, rng=42,
        )

        # Point estimate should be finite (1.0 for 5 > 3)
        assert np.isfinite(point), f"Expected finite point estimate, got {point}"
        assert point == 1.0, f"Expected 1.0 for single-element perfect separation, got {point}"
        # CIs may or may not be finite with single-element bootstrap -- either is acceptable
