"""Audit tests for Frenet-Serret curvature and torsion discrete formulas (SVD-06).

Verifies the discrete differential geometry formulas in spectral_curvature and
spectral_torsion against analytically known curves (circle, helix). Tests
convergence with step refinement and index mapping correctness.

Strategy: bypass Savitzky-Golay smoothing by using window_length=3, polyorder=1
(minimal configuration) to isolate discrete formula accuracy from filter effects.
"""

import numpy as np
import pytest

from src.analysis.spectrum import spectral_curvature, spectral_torsion


# Minimal Savitzky-Golay parameters to nearly bypass smoothing
MINIMAL_WINDOW = 3
MINIMAL_POLYORDER = 1


def _descending_bases(k: int, gap: float = 10.0, top: float = 100.0) -> np.ndarray:
    """Return 1-D array of strictly descending base values for k dimensions.

    Real singular values are always sorted in descending order. Using well-separated
    base values (e.g. 100, 90, 80, ...) ensures the ordering-crossing detector in
    spectral_curvature/torsion never triggers, isolating the curvature/torsion
    formula from the crossing-mask logic.
    """
    return np.array([top - i * gap for i in range(k)], dtype=np.float64)


def _make_circle(n: int, r: float = 1.0, k: int = 8) -> np.ndarray:
    """Generate a circle in R^k embedded in the first two dimensions.

    dim_0(t) = base_0 + r*cos(2*pi*t/N)
    dim_1(t) = base_1 + r*sin(2*pi*t/N)
    dim_2..dim_{k-1} = base_i (constant, strictly descending)

    The oscillation amplitude r must be << gap between bases so that adjacent
    dimensions never swap ordering.

    Analytic curvature: kappa = 1/r everywhere.
    Analytic torsion: tau = 0 (planar curve).
    """
    bases = _descending_bases(k)
    t = np.arange(n, dtype=np.float64)
    spectra = np.tile(bases, (n, 1))  # [n, k] with each row = bases
    spectra[:, 0] = bases[0] + r * np.cos(2 * np.pi * t / n)
    spectra[:, 1] = bases[1] + r * np.sin(2 * np.pi * t / n)
    return spectra


def _make_helix(
    n: int, r: float = 1.0, pitch: float = 0.5, k: int = 8
) -> np.ndarray:
    """Generate a helix in R^k embedded in the first three dimensions.

    dim_0(t) = base_0 + r*cos(2*pi*t/N)
    dim_1(t) = base_1 + r*sin(2*pi*t/N)
    dim_2(t) = base_2 + pitch*t/N
    dim_3..dim_{k-1} = base_i (constant, strictly descending)

    Analytic curvature: kappa = r / (r^2 + c^2) where c = pitch/(2*pi).
    Analytic torsion: tau = c / (r^2 + c^2).
    """
    bases = _descending_bases(k)
    t = np.arange(n, dtype=np.float64)
    spectra = np.tile(bases, (n, 1))
    spectra[:, 0] = bases[0] + r * np.cos(2 * np.pi * t / n)
    spectra[:, 1] = bases[1] + r * np.sin(2 * np.pi * t / n)
    spectra[:, 2] = bases[2] + pitch * t / n
    return spectra


class TestCircleCurvature:
    """Verify spectral_curvature on a circle converges to 1/r.

    For a circle of radius r in R^k, the Frenet-Serret curvature is kappa = 1/r
    everywhere. With 1000 sample points and minimal smoothing, the discrete
    approximation should match within 5%.
    """

    def test_circle_curvature_median_matches_analytic(self) -> None:
        """Median curvature of a circle with r=1.0 should be approximately 1.0.

        Discrete forward differences on a circle of radius r produce curvature
        kappa_discrete = (2/r)*sin(pi/N) / (2*sin(pi/N))^2 which converges to
        1/r as N -> infinity. For N=1000, the O(h^2) error is negligible.
        """
        r = 1.0
        n = 1000
        spectra = _make_circle(n, r=r)
        curvature = spectral_curvature(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)

        # Extract non-NaN interior values
        valid = curvature[~np.isnan(curvature)]
        assert len(valid) > n * 0.5, f"Expected >50% valid curvature values, got {len(valid)}/{n}"

        median_kappa = np.median(valid)
        expected_kappa = 1.0 / r
        rel_error = abs(median_kappa - expected_kappa) / expected_kappa
        assert rel_error < 0.05, (
            f"Circle curvature median {median_kappa:.4f} deviates from expected {expected_kappa:.4f} "
            f"by {rel_error:.2%} (>5%)"
        )

    def test_circle_curvature_different_radius(self) -> None:
        """Curvature scales as 1/r: a circle with r=2.0 should have kappa ~ 0.5."""
        r = 2.0
        n = 1000
        spectra = _make_circle(n, r=r)
        curvature = spectral_curvature(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)

        valid = curvature[~np.isnan(curvature)]
        median_kappa = np.median(valid)
        expected_kappa = 1.0 / r
        rel_error = abs(median_kappa - expected_kappa) / expected_kappa
        assert rel_error < 0.05, (
            f"Circle (r={r}) curvature median {median_kappa:.4f} deviates from "
            f"expected {expected_kappa:.4f} by {rel_error:.2%}"
        )


class TestCircleTorsion:
    """Verify spectral_torsion on a planar circle is approximately zero.

    A planar curve has zero torsion by definition: it does not twist out of its
    osculating plane. For a circle in the (s1, s2) plane with constant s3..sk,
    all torsion values should be near zero.
    """

    def test_circle_torsion_near_zero(self) -> None:
        """Torsion of a planar circle should be approximately 0.

        The jerk component perpendicular to both velocity and normal directions
        is zero for a planar curve, making tau = ||j_perp|| / (||v|| * ||a_perp||) = 0.
        Allow small numerical error from discrete approximation.
        """
        n = 1000
        spectra = _make_circle(n, r=1.0)
        torsion = spectral_torsion(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)

        valid = torsion[~np.isnan(torsion)]
        assert len(valid) > 0, "Expected some valid torsion values for circle"

        # All torsion values should be near zero (planar curve)
        max_abs_torsion = np.max(np.abs(valid))
        assert max_abs_torsion < 0.05, (
            f"Circle torsion should be ~0 (planar curve), got max |tau| = {max_abs_torsion:.4f}"
        )


class TestHelixCurvatureTorsion:
    """Verify curvature and torsion on a helix match analytic formulas.

    For a helix with radius r and pitch p, parameterized by arc length:
        kappa = r / (r^2 + c^2)  where c = p / (2*pi)
        tau   = c / (r^2 + c^2)

    The helix has both nonzero curvature and nonzero torsion, making it a
    stronger test than the circle.
    """

    def test_helix_curvature_matches_analytic(self) -> None:
        """Median helix curvature should match kappa = r / (r^2 + c^2).

        With r=1.0, pitch=0.5: c = 0.5/(2*pi) ~ 0.0796
        kappa = 1.0 / (1.0 + 0.00633) ~ 0.9937
        """
        r = 1.0
        pitch = 0.5
        c = pitch / (2 * np.pi)
        expected_kappa = r / (r**2 + c**2)

        n = 2000
        spectra = _make_helix(n, r=r, pitch=pitch)
        curvature = spectral_curvature(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)

        valid = curvature[~np.isnan(curvature)]
        assert len(valid) > n * 0.5, f"Expected >50% valid curvature, got {len(valid)}/{n}"

        median_kappa = np.median(valid)
        rel_error = abs(median_kappa - expected_kappa) / expected_kappa
        assert rel_error < 0.10, (
            f"Helix curvature median {median_kappa:.4f} deviates from expected "
            f"{expected_kappa:.4f} by {rel_error:.2%} (>10%)"
        )

    def test_helix_torsion_matches_analytic(self) -> None:
        """Median helix torsion should match tau = c / (r^2 + c^2).

        With r=1.0, pitch=0.5: c ~ 0.0796, tau ~ 0.0791.
        Torsion is noisier than curvature, so allow 10% tolerance.
        """
        r = 1.0
        pitch = 0.5
        c = pitch / (2 * np.pi)
        expected_tau = c / (r**2 + c**2)

        n = 2000
        spectra = _make_helix(n, r=r, pitch=pitch)
        torsion = spectral_torsion(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)

        valid = torsion[~np.isnan(torsion)]
        assert len(valid) > 0, "Expected some valid torsion values for helix"

        median_tau = np.median(valid)
        rel_error = abs(median_tau - expected_tau) / expected_tau
        assert rel_error < 0.10, (
            f"Helix torsion median {median_tau:.4f} deviates from expected "
            f"{expected_tau:.4f} by {rel_error:.2%} (>10%)"
        )


class TestCurvatureConvergence:
    """Verify curvature error decreases with step refinement (O(h) rate).

    Forward differences produce O(h) discretization error where h = 2*pi/N
    for a circle. Testing at N = 100, 1000, 10000 should show monotonically
    decreasing error, with approximately 10x error reduction per 10x N increase.
    """

    def test_convergence_monotonically_decreasing(self) -> None:
        """Error should decrease as sampling resolution increases.

        At N=100 the discrete approximation is coarser than at N=1000,
        which is coarser than N=10000. The median absolute error vs
        the analytic kappa=1.0 should strictly decrease.
        """
        r = 1.0
        expected_kappa = 1.0 / r
        ns = [100, 1000, 10000]
        errors = []

        for n in ns:
            spectra = _make_circle(n, r=r)
            curvature = spectral_curvature(
                spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER
            )
            valid = curvature[~np.isnan(curvature)]
            assert len(valid) > 0, f"No valid curvature at N={n}"
            median_error = np.median(np.abs(valid - expected_kappa))
            errors.append(median_error)

        # Monotonically decreasing error
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], (
                f"Error not decreasing: N={ns[i]} error={errors[i]:.6f} "
                f"vs N={ns[i + 1]} error={errors[i + 1]:.6f}"
            )

    def test_convergence_rate_at_least_order_h(self) -> None:
        """Error ratio between successive resolutions should be at least ~10x.

        For O(h) convergence with 10x resolution increase, error decreases >= 10x.
        In practice, the discrete curvature formula on a circle achieves O(h^2)
        convergence (ratio ~100x), because the centered-difference structure of
        diff(diff(smoothed)) cancels first-order error terms on a smooth curve.
        We verify at least O(h) rate (ratio > 5x) — observing better than O(h)
        is expected and confirms the formula's numerical quality.
        """
        r = 1.0
        expected_kappa = 1.0 / r
        ns = [100, 1000, 10000]
        errors = []

        for n in ns:
            spectra = _make_circle(n, r=r)
            curvature = spectral_curvature(
                spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER
            )
            valid = curvature[~np.isnan(curvature)]
            median_error = np.median(np.abs(valid - expected_kappa))
            errors.append(median_error)

        # Check error ratio between successive refinements
        # Minimum O(h): ratio > 5x.  Allow up to O(h^2): ratio up to 200x.
        for i in range(len(errors) - 1):
            if errors[i + 1] > 0:
                ratio = errors[i] / errors[i + 1]
                assert ratio > 5.0, (
                    f"Convergence ratio N={ns[i]}->N={ns[i + 1]}: {ratio:.1f}x "
                    f"below expected minimum 5x for at-least O(h) convergence"
                )


class TestCurvatureIndexMapping:
    """Verify curvature index mapping: a[t] -> orig_idx=t+1.

    A curve with a sharp turn at a known step should produce peak curvature
    at orig_idx near the turn location. This verifies that the +1 offset in
    the implementation correctly maps derivative indices to original indices.
    """

    def test_peak_curvature_at_turn_location(self) -> None:
        """Sharp corner at t_turn should produce max curvature near orig_idx=t_turn.

        Construct a piecewise-linear path: straight in dim_0 for t < t_turn,
        then straight in dim_1 for t > t_turn. The 90-degree corner at t_turn
        creates a curvature spike that should map to approximately t_turn
        in original index space (within 2 indices of t_turn).

        Base values are well-separated (descending) to avoid crossing-mask triggers.
        The small linear increments (scale ~n=50) are negligible vs the 10-unit gap.
        """
        n = 50
        t_turn = 25
        k = 8
        bases = _descending_bases(k)

        spectra = np.tile(bases, (n, 1))
        for t in range(n):
            if t <= t_turn:
                spectra[t, 0] = bases[0] + float(t)
                spectra[t, 1] = bases[1]
            else:
                spectra[t, 0] = bases[0] + float(t_turn)
                spectra[t, 1] = bases[1] + float(t - t_turn)

        curvature = spectral_curvature(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)

        # Find index of maximum curvature (excluding NaN)
        valid_mask = ~np.isnan(curvature)
        assert np.any(valid_mask), "No valid curvature values found"

        peak_idx = np.nanargmax(curvature)
        assert abs(peak_idx - t_turn) <= 2, (
            f"Peak curvature at index {peak_idx}, expected near {t_turn} "
            f"(off by {abs(peak_idx - t_turn)})"
        )


class TestTorsionIndexMapping:
    """Verify torsion index mapping: j[t] -> orig_idx=t+2.

    A planar curve transitioning to a helix at a known step should produce
    peak torsion near orig_idx = t_twist + 2, verifying the +2 offset.
    """

    def test_peak_torsion_at_twist_location(self) -> None:
        """Transition from planar circle to helix should produce torsion peak
        near orig_idx = t_twist + 2.

        For t < t_twist: circle in (dim_0, dim_1) plane (zero torsion).
        For t >= t_twist: helix rising in dim_2 (nonzero torsion onset).
        The torsion spike from the transition should map to approximately
        t_twist + 2 in original index space (within 5 indices).

        Base values are well-separated (descending) to avoid crossing-mask triggers.
        """
        n = 100
        t_twist = 50
        k = 8
        r = 1.0
        bases = _descending_bases(k)

        spectra = np.tile(bases, (n, 1))
        for t in range(n):
            angle = 2 * np.pi * t / n
            spectra[t, 0] = bases[0] + r * np.cos(angle)
            spectra[t, 1] = bases[1] + r * np.sin(angle)
            if t >= t_twist:
                # Add out-of-plane component: rising helix
                spectra[t, 2] = bases[2] + 0.5 * (t - t_twist) / n
            else:
                spectra[t, 2] = bases[2]

        torsion = spectral_torsion(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)

        # Find index of maximum absolute torsion (excluding NaN)
        valid_mask = ~np.isnan(torsion)
        assert np.any(valid_mask), "No valid torsion values found"

        abs_torsion = np.abs(np.where(valid_mask, torsion, 0.0))
        peak_idx = np.argmax(abs_torsion)
        expected_peak = t_twist + 2

        assert abs(peak_idx - expected_peak) <= 5, (
            f"Peak torsion at index {peak_idx}, expected near {expected_peak} "
            f"(off by {abs(peak_idx - expected_peak)})"
        )


class TestBoundaryNaN:
    """Verify boundary NaN behavior at array edges.

    Curvature requires two derivatives (velocity + acceleration), so the first
    and last values should be NaN. Torsion requires three derivatives, so the
    first two and last values should be NaN.
    """

    def test_curvature_first_value_is_nan(self) -> None:
        """curvature[0] is NaN because v[0] maps to the interval [0,1],
        and a[0] maps to orig_idx=1, leaving index 0 unassigned."""
        spectra = _make_circle(100, r=1.0)
        curvature = spectral_curvature(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)
        assert np.isnan(curvature[0]), f"curvature[0] should be NaN, got {curvature[0]}"

    def test_curvature_last_value_is_nan(self) -> None:
        """curvature[-1] is NaN because the acceleration array has length n_steps-2,
        and the last mapped index is (n_steps-2-1)+1 = n_steps-2, not n_steps-1."""
        spectra = _make_circle(100, r=1.0)
        curvature = spectral_curvature(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)
        assert np.isnan(curvature[-1]), f"curvature[-1] should be NaN, got {curvature[-1]}"

    def test_torsion_first_two_values_are_nan(self) -> None:
        """torsion[0] and torsion[1] are NaN because jerk j[t] maps to orig_idx=t+2,
        so the earliest assigned index is t=0 -> orig_idx=2."""
        spectra = _make_circle(100, r=1.0)
        torsion = spectral_torsion(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)
        assert np.isnan(torsion[0]), f"torsion[0] should be NaN, got {torsion[0]}"
        assert np.isnan(torsion[1]), f"torsion[1] should be NaN, got {torsion[1]}"

    def test_torsion_last_value_is_nan(self) -> None:
        """torsion[-1] is NaN because jerk array has length n_steps-3,
        and the last mapped index is (n_steps-3-1)+2 = n_steps-2, not n_steps-1."""
        spectra = _make_circle(100, r=1.0)
        torsion = spectral_torsion(spectra, window_length=MINIMAL_WINDOW, polyorder=MINIMAL_POLYORDER)
        assert np.isnan(torsion[-1]), f"torsion[-1] should be NaN, got {torsion[-1]}"
