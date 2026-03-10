"""Audit tests for Marchenko-Pastur distribution formulas and sigma^2 calibration
(NULL-02).

Verifies that the MP PDF integrates to 1.0, is non-negative on its support,
the CDF is monotonically non-decreasing with correct boundary values, scalar
and array inputs are handled correctly, sigma^2 calibration satisfies
E[lambda] = sigma^2 * (1 + gamma), and the KS test correctly accepts true
MP data and rejects non-MP data.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from src.analysis.null_model import (
    marchenko_pastur_cdf,
    marchenko_pastur_pdf,
    run_mp_ks_test,
)


def _mp_bounds(gamma: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Compute Marchenko-Pastur support [lambda_minus, lambda_plus]."""
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    return lam_minus, lam_plus


class TestMarchenkoPasturPDF:
    """Verify MP PDF formula: correct support, non-negativity, and integration to 1."""

    def test_pdf_zero_outside_support(self):
        """MP PDF must return 0 for x outside [lambda_minus, lambda_plus].
        Outside the support, no eigenvalues are expected in the MP distribution."""
        gamma = 0.5
        lam_minus, lam_plus = _mp_bounds(gamma)

        # Below support
        assert marchenko_pastur_pdf(lam_minus - 0.01, gamma) == 0.0, (
            "PDF must be zero below lambda_minus"
        )
        # Above support
        assert marchenko_pastur_pdf(lam_plus + 0.01, gamma) == 0.0, (
            "PDF must be zero above lambda_plus"
        )
        # At boundaries (open interval)
        assert marchenko_pastur_pdf(lam_minus, gamma) == 0.0, (
            "PDF must be zero at lambda_minus (boundary)"
        )
        assert marchenko_pastur_pdf(lam_plus, gamma) == 0.0, (
            "PDF must be zero at lambda_plus (boundary)"
        )

    def test_pdf_positive_inside_support(self):
        """MP PDF must be strictly positive inside (lambda_minus, lambda_plus).
        This confirms the sqrt factor is computed correctly."""
        gamma = 0.5
        lam_minus, lam_plus = _mp_bounds(gamma)

        # Midpoint of support
        mid = (lam_minus + lam_plus) / 2.0
        assert marchenko_pastur_pdf(mid, gamma) > 0, (
            "PDF must be positive inside the support"
        )

    @pytest.mark.parametrize("gamma", [0.25, 0.5, 1.0])
    def test_pdf_integrates_to_one_gamma_leq_1(self, gamma):
        """For gamma <= 1, MP PDF integrates to 1.0 (no point mass at zero).
        This is the normalization requirement when m <= n."""
        lam_minus, lam_plus = _mp_bounds(gamma)

        eps = 1e-10
        integral, _ = quad(
            marchenko_pastur_pdf, lam_minus + eps, lam_plus - eps,
            args=(gamma,),
        )

        np.testing.assert_allclose(
            integral, 1.0, atol=1e-6,
            err_msg=f"MP PDF must integrate to 1.0 for gamma={gamma}, got {integral}"
        )

    def test_pdf_integrates_to_one_over_gamma_when_gamma_gt_1(self):
        """For gamma > 1, the continuous part of MP integrates to 1/gamma.
        The remaining mass (1 - 1/gamma) is a point mass at zero (n - m zero
        eigenvalues in the sample covariance matrix)."""
        gamma = 2.0
        lam_minus, lam_plus = _mp_bounds(gamma)

        eps = 1e-10
        integral, _ = quad(
            marchenko_pastur_pdf, lam_minus + eps, lam_plus - eps,
            args=(gamma,),
        )

        # Continuous part integrates to 1/gamma for gamma > 1
        expected = 1.0 / gamma
        np.testing.assert_allclose(
            integral, expected, atol=1e-6,
            err_msg=(
                f"For gamma={gamma} > 1, continuous MP PDF must integrate to "
                f"1/gamma={expected}, got {integral}"
            )
        )

    def test_pdf_non_negative_everywhere(self):
        """MP PDF must be non-negative at all points in its support.
        Sample 100 points uniformly and verify all values >= 0."""
        gamma = 0.5
        lam_minus, lam_plus = _mp_bounds(gamma)

        # Sample 100 interior points
        points = np.linspace(lam_minus + 1e-10, lam_plus - 1e-10, 100)
        values = [marchenko_pastur_pdf(float(x), gamma) for x in points]

        assert all(v >= 0 for v in values), (
            f"All PDF values must be non-negative; min value = {min(values)}"
        )

    def test_pdf_with_sigma2(self):
        """MP PDF with sigma2 != 1.0 must still integrate to 1.0.
        The sigma^2 parameter scales the support but preserves normalization."""
        gamma = 0.5
        sigma2 = 2.5
        lam_minus, lam_plus = _mp_bounds(gamma, sigma2)

        eps = 1e-10
        integral, _ = quad(
            marchenko_pastur_pdf, lam_minus + eps, lam_plus - eps,
            args=(gamma, sigma2),
        )

        np.testing.assert_allclose(
            integral, 1.0, atol=1e-6,
            err_msg=f"MP PDF with sigma2={sigma2} must still integrate to 1.0"
        )


class TestMarchenkoPasturCDF:
    """Verify MP CDF: monotonicity, boundary values, and input type handling."""

    def test_cdf_monotonicity(self):
        """MP CDF must be monotonically non-decreasing across its support.
        Any decrease would indicate an integration error."""
        gamma = 0.5
        lam_minus, lam_plus = _mp_bounds(gamma)

        # 20 equally-spaced points in support
        points = np.linspace(lam_minus, lam_plus, 20)
        cdf_values = marchenko_pastur_cdf(points, gamma)

        # Check monotonicity: each value >= previous
        for i in range(1, len(cdf_values)):
            assert cdf_values[i] >= cdf_values[i - 1] - 1e-10, (
                f"CDF must be monotonically non-decreasing: "
                f"CDF[{i-1}]={cdf_values[i-1]}, CDF[{i}]={cdf_values[i]}"
            )

    def test_cdf_boundary_at_lambda_minus(self):
        """CDF at lambda_minus must be 0 (within tolerance).
        No probability mass exists below the lower support boundary."""
        gamma = 0.5
        lam_minus, _ = _mp_bounds(gamma)

        cdf_val = marchenko_pastur_cdf(lam_minus, gamma)
        np.testing.assert_allclose(
            cdf_val, 0.0, atol=1e-6,
            err_msg="CDF at lambda_minus must be 0"
        )

    def test_cdf_boundary_at_lambda_plus(self):
        """CDF at lambda_plus must be 1.0 (within tolerance).
        All probability mass is contained within the support."""
        gamma = 0.5
        _, lam_plus = _mp_bounds(gamma)

        cdf_val = marchenko_pastur_cdf(lam_plus, gamma)
        np.testing.assert_allclose(
            cdf_val, 1.0, atol=1e-6,
            err_msg="CDF at lambda_plus must be 1.0"
        )

    def test_cdf_handles_array_input(self):
        """CDF must accept numpy array input and return an array of same length.
        This is required for scipy.stats.kstest which passes arrays."""
        gamma = 0.5
        lam_minus, lam_plus = _mp_bounds(gamma)

        points = np.linspace(lam_minus, lam_plus, 10)
        result = marchenko_pastur_cdf(points, gamma)

        assert isinstance(result, np.ndarray), "CDF must return ndarray for array input"
        assert len(result) == 10, f"CDF output length must match input, got {len(result)}"

    def test_cdf_handles_scalar_input(self):
        """CDF must accept a scalar float and return a scalar float.
        Both input types are used in different calling contexts."""
        gamma = 0.5
        lam_minus, lam_plus = _mp_bounds(gamma)

        mid = float((lam_minus + lam_plus) / 2.0)
        result = marchenko_pastur_cdf(mid, gamma)

        assert isinstance(result, float), (
            f"CDF must return float for scalar input, got {type(result)}"
        )


class TestSigma2Calibration:
    """Verify sigma^2 calibration from random matrix theory:
    E[lambda_MP(gamma, sigma^2)] = sigma^2."""

    def test_sigma2_estimation_from_random_matrix(self):
        """For X ~ N(0, sigma^2_true), eigenvalues of (1/n) X X^T (the m x m
        sample covariance) follow MP(gamma, sigma^2_true). The mean of this
        distribution is sigma^2, so sigma^2_estimated = mean(eigenvalues)
        should approximate sigma^2_true."""
        sigma2_true = 2.0
        m, n = 200, 400
        gamma = m / n  # 0.5

        errors = []
        for seed in range(5):
            rng = np.random.default_rng(42 + seed)
            X = rng.normal(0, np.sqrt(sigma2_true), size=(m, n))
            # Eigenvalues of (1/n) X X^T follow MP(gamma=m/n, sigma^2_true)
            eigenvalues = np.linalg.eigvalsh((X @ X.T) / n)

            # Calibration: sigma^2 = mean(eigenvalues) since E[lambda_MP] = sigma^2
            sigma2_est = float(np.mean(eigenvalues))
            errors.append(abs(sigma2_est - sigma2_true) / sigma2_true)

        mean_error = np.mean(errors)
        # Mean relative error should be < 5% for m=200, n=400
        assert mean_error < 0.05, (
            f"sigma^2 calibration mean relative error {mean_error:.4f} exceeds 5%"
        )

    def test_mean_eigenvalue_formula(self):
        """E[lambda_MP(gamma, sigma^2)] = sigma^2 is the mean of the MP distribution.
        Verify empirically: mean eigenvalue of (1/n) X X^T should approximate
        sigma^2_true, NOT sigma^2 * (1 + gamma). Also verify by integrating
        x * f_MP(x) over the support."""
        sigma2_true = 1.0
        gamma = 0.5

        # Verify analytically: integrate x * PDF over support
        lam_minus = sigma2_true * (1 - np.sqrt(gamma)) ** 2
        lam_plus = sigma2_true * (1 + np.sqrt(gamma)) ** 2

        def mp_mean_integrand(x):
            return x * marchenko_pastur_pdf(x, gamma, sigma2_true)

        from scipy.integrate import quad
        analytical_mean, _ = quad(mp_mean_integrand, lam_minus + 1e-10, lam_plus - 1e-10)

        np.testing.assert_allclose(
            analytical_mean, sigma2_true, atol=1e-6,
            err_msg=f"E[lambda_MP] must equal sigma^2={sigma2_true}, got {analytical_mean}"
        )

        # Verify empirically with random matrices
        m, n = 200, 400
        rng = np.random.default_rng(42)
        errors = []
        for _ in range(5):
            X = rng.normal(0, np.sqrt(sigma2_true), size=(m, n))
            eigenvalues = np.linalg.eigvalsh((X @ X.T) / n)
            actual_mean = float(np.mean(eigenvalues))
            errors.append(abs(actual_mean - sigma2_true) / sigma2_true)

        mean_error = np.mean(errors)
        assert mean_error < 0.05, (
            f"E[lambda] = sigma^2 error {mean_error:.4f} exceeds 5%"
        )

    def test_sigma2_individual_trial(self):
        """Single trial with m=200, n=400: sigma^2_estimated = mean(eigenvalues)
        should be within 10% of sigma^2_true for a reasonably sized random matrix."""
        sigma2_true = 1.5
        m, n = 200, 400

        rng = np.random.default_rng(42)
        X = rng.normal(0, np.sqrt(sigma2_true), size=(m, n))
        eigenvalues = np.linalg.eigvalsh((X @ X.T) / n)
        sigma2_est = float(np.mean(eigenvalues))

        relative_error = abs(sigma2_est - sigma2_true) / sigma2_true
        assert relative_error < 0.10, (
            f"sigma^2 estimation relative error {relative_error:.4f} exceeds 10% "
            f"(estimated={sigma2_est:.4f}, true={sigma2_true})"
        )


class TestMPKSTest:
    """Verify KS test correctly accepts true MP data and rejects non-MP data."""

    def test_ks_accepts_mp_data(self):
        """Eigenvalues of (1/n) X X^T for X ~ N(0, 1) follow MP distribution.
        run_mp_ks_test takes singular values, squares them, and tests against MP CDF.
        KS test p-value should be > 0.05 (fail to reject MP hypothesis)."""
        m, n = 100, 200
        gamma = m / n  # 0.5

        rng = np.random.default_rng(42)
        X = rng.normal(0, 1.0, size=(m, n))
        # Eigenvalues of (1/n) X X^T (m x m sample covariance)
        eigenvalues = np.linalg.eigvalsh((X @ X.T) / n)
        # run_mp_ks_test expects singular values; sv^2 = eigenvalues
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))

        result = run_mp_ks_test(singular_values, gamma)

        assert result["ks_p_value"] > 0.05, (
            f"KS test should accept true MP data (p={result['ks_p_value']:.4f}), "
            f"but got p < 0.05"
        )

    def test_ks_rejects_non_mp_data(self):
        """Eigenvalues from a uniform distribution do NOT follow MP.
        KS test p-value should be < 0.05 (reject MP hypothesis)."""
        rng = np.random.default_rng(42)
        gamma = 0.5

        # Generate 500 uniform "singular values" -- definitely not MP
        uniform_sv = rng.uniform(0.5, 5.0, size=500)

        result = run_mp_ks_test(uniform_sv, gamma)

        assert result["ks_p_value"] < 0.05, (
            f"KS test should reject non-MP data (p={result['ks_p_value']:.4f}), "
            f"but got p >= 0.05"
        )

    def test_ks_sigma2_calibration(self):
        """run_mp_ks_test calibrates sigma^2 from the data via
        sigma^2 = mean(sv^2). Verify the returned sigma2 matches this
        formula applied to the input singular values."""
        rng = np.random.default_rng(42)
        m, n = 100, 200
        gamma = m / n

        X = rng.normal(0, 1.0, size=(m, n))
        eigenvalues = np.linalg.eigvalsh((X @ X.T) / n)
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))

        result = run_mp_ks_test(singular_values, gamma)

        # Manually compute sigma^2 from the same formula: sigma^2 = mean(sv^2)
        sv_squared = singular_values.astype(np.float64) ** 2
        expected_sigma2 = float(np.mean(sv_squared))

        np.testing.assert_allclose(
            result["sigma2"], expected_sigma2, atol=1e-10,
            err_msg="sigma2 in KS test result must match manual calibration formula"
        )

    def test_ks_returns_correct_mp_bounds(self):
        """run_mp_ks_test must return lambda_minus and lambda_plus computed
        from the calibrated sigma^2 and input gamma."""
        rng = np.random.default_rng(42)
        m, n = 100, 200
        gamma = m / n

        X = rng.normal(0, 1.0, size=(m, n))
        eigenvalues = np.linalg.eigvalsh((X @ X.T) / n)
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))

        result = run_mp_ks_test(singular_values, gamma)

        sigma2 = result["sigma2"]
        expected_lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
        expected_lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

        np.testing.assert_allclose(
            result["lambda_minus"], expected_lam_minus, atol=1e-10,
            err_msg="lambda_minus must match sigma2 * (1 - sqrt(gamma))^2"
        )
        np.testing.assert_allclose(
            result["lambda_plus"], expected_lam_plus, atol=1e-10,
            err_msg="lambda_plus must match sigma2 * (1 + sqrt(gamma))^2"
        )

    def test_ks_returns_gamma(self):
        """run_mp_ks_test must echo back the input gamma in the result dict
        for downstream consumers."""
        rng = np.random.default_rng(42)
        sv = rng.normal(1.0, 0.1, size=50)
        gamma = 0.5

        result = run_mp_ks_test(np.abs(sv), gamma)

        assert result["gamma"] == gamma, (
            f"Returned gamma {result['gamma']} must match input {gamma}"
        )
