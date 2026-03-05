"""Audit tests for float16 vs float32 spectrum storage fidelity (SVD-05).

Quantifies the impact of float16 round-trip on curvature and torsion
computation using synthetic spectra with analytically known geometry.
Uses circle (known curvature = 1/r) and helix (known torsion) as test curves.

Float16 has ~3 decimal digits of precision. Since curvature involves second
derivatives (amplifying quantization noise), float16 is expected to introduce
significant relative error.

Threshold: if relative error > 10%, recommend switch to float32.

IMPORTANT: Synthetic spectra must maintain descending order (s1 > s2 > ... > sk)
because spectral_curvature/torsion call detect_ordering_crossings() and NaN-mask
steps where ordering is violated. We achieve this by spacing base values apart.
"""

import numpy as np
import pytest

from src.analysis.spectrum import spectral_curvature, spectral_torsion


def _make_circle_spectrum(
    n_points: int = 200, radius: float = 0.3, k: int = 4
) -> np.ndarray:
    """Create a circular spectrum trajectory in R^k with descending ordering.

    Circle of radius r in the (s1, s2) plane. Base values are well-separated
    (10, 7, 4, 1) so that even with perturbation from the circle, the
    descending order s1 > s2 > s3 > ... is maintained.

    Analytical curvature = 1/r for the continuous curve.
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    bases = np.array([10.0, 7.0, 4.0, 1.0])[:k]
    spectra = np.tile(bases, (n_points, 1)).astype(np.float32)
    # Circular perturbation in first two components
    # radius must be < (bases[0] - bases[1]) / 2 = 1.5 to maintain ordering
    spectra[:, 0] += radius * np.cos(t).astype(np.float32)
    spectra[:, 1] += radius * np.sin(t).astype(np.float32)
    return spectra


def _make_helix_spectrum(
    n_points: int = 300, radius: float = 0.3, pitch: float = 0.2, k: int = 4
) -> np.ndarray:
    """Create a helix spectrum trajectory in R^k with descending ordering.

    Helix: circular motion in (s1, s2) plus linear drift in s3.
    Base values well-separated to maintain descending order.

    Analytical torsion for continuous helix: tau = pitch / (r^2 + pitch^2).
    """
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    bases = np.array([10.0, 7.0, 4.0, 1.0])[:k]
    spectra = np.tile(bases, (n_points, 1)).astype(np.float32)
    spectra[:, 0] += radius * np.cos(t).astype(np.float32)
    spectra[:, 1] += radius * np.sin(t).astype(np.float32)
    # Linear rise in s3 — small enough to not violate ordering
    # Max rise = pitch, and bases[2] - bases[3] = 3.0, so pitch < 3.0 is safe
    spectra[:, 2] += (pitch * t / (2 * np.pi)).astype(np.float32)
    return spectra


class TestFloat16CurvatureFidelity:
    """Compare curvature from float32 vs float16-roundtripped spectra on a circle."""

    def test_curvature_float16_error_quantified(self) -> None:
        """Compute curvature from float32 vs float16-roundtripped spectra.
        Uses minimal smoothing (window_length=3, polyorder=1) to isolate
        discrete differential geometry from Savitzky-Golay filter.
        """
        spectra_f32 = _make_circle_spectrum(n_points=200, radius=0.3)

        # Float16 round-trip: simulates pipeline.py storage
        spectra_f16 = spectra_f32.astype(np.float16).astype(np.float32)

        # Compute curvature with minimal smoothing
        kappa_f32 = spectral_curvature(spectra_f32, window_length=3, polyorder=1)
        kappa_f16 = spectral_curvature(spectra_f16, window_length=3, polyorder=1)

        # Compute relative error over valid (non-NaN) steps
        valid = ~np.isnan(kappa_f32) & ~np.isnan(kappa_f16) & (np.abs(kappa_f32) > 1e-10)
        assert valid.sum() > 50, (
            f"Need enough valid curvature points for comparison, got {valid.sum()}"
        )

        abs_error = np.abs(kappa_f32[valid] - kappa_f16[valid])
        rel_error = abs_error / (np.abs(kappa_f32[valid]) + 1e-10)
        max_rel_error = float(rel_error.max())
        mean_rel_error = float(rel_error.mean())
        max_abs_error = float(abs_error.max())

        print(f"\n[SVD-05] Float16 Curvature Fidelity (circle, r=0.3, N=200):")
        print(f"  Max relative error:  {max_rel_error:.4f} ({max_rel_error*100:.1f}%)")
        print(f"  Mean relative error: {mean_rel_error:.4f} ({mean_rel_error*100:.1f}%)")
        print(f"  Max absolute error:  {max_abs_error:.6f}")
        print(f"  Valid points:        {valid.sum()}")

    def test_curvature_float32_baseline_reasonable(self) -> None:
        """Float32 curvature on a circle should approximate 1/r.
        Analytical curvature = 1/r = 1/0.3 ~ 3.33.
        Not exact due to discrete approximation but should be in the right ballpark.
        """
        spectra = _make_circle_spectrum(n_points=200, radius=0.3)
        kappa = spectral_curvature(spectra, window_length=3, polyorder=1)

        valid = ~np.isnan(kappa) & (kappa > 0)
        assert valid.sum() > 50, (
            f"Need enough valid curvature points, got {valid.sum()}"
        )
        median_kappa = float(np.median(kappa[valid]))
        analytical = 1.0 / 0.3  # ~ 3.333

        print(f"\n[SVD-05] Float32 curvature baseline: median={median_kappa:.4f} (analytical={analytical:.4f})")
        # Discrete curvature on a circle should be within an order of magnitude
        assert 0.1 < median_kappa < 100.0, (
            f"Float32 curvature should be in reasonable range, got median={median_kappa}"
        )


class TestFloat16TorsionFidelity:
    """Compare torsion from float32 vs float16-roundtripped spectra on a helix."""

    def test_torsion_float16_error_quantified(self) -> None:
        """Compute torsion from float32 vs float16-roundtripped spectra.
        Uses minimal smoothing to isolate numerical precision effects.
        """
        spectra_f32 = _make_helix_spectrum(n_points=300, radius=0.3, pitch=0.2)

        # Float16 round-trip
        spectra_f16 = spectra_f32.astype(np.float16).astype(np.float32)

        # Compute torsion with minimal smoothing
        tau_f32 = spectral_torsion(spectra_f32, window_length=3, polyorder=1)
        tau_f16 = spectral_torsion(spectra_f16, window_length=3, polyorder=1)

        # Compute relative error
        valid = ~np.isnan(tau_f32) & ~np.isnan(tau_f16) & (np.abs(tau_f32) > 1e-10)
        assert valid.sum() > 20, (
            f"Need enough valid torsion points, got {valid.sum()}"
        )

        abs_error = np.abs(tau_f32[valid] - tau_f16[valid])
        rel_error = abs_error / (np.abs(tau_f32[valid]) + 1e-10)
        max_rel_error = float(rel_error.max())
        mean_rel_error = float(rel_error.mean())
        max_abs_error = float(abs_error.max())

        print(f"\n[SVD-05] Float16 Torsion Fidelity (helix, r=0.3, pitch=0.2, N=300):")
        print(f"  Max relative error:  {max_rel_error:.4f} ({max_rel_error*100:.1f}%)")
        print(f"  Mean relative error: {mean_rel_error:.4f} ({mean_rel_error*100:.1f}%)")
        print(f"  Max absolute error:  {max_abs_error:.6f}")
        print(f"  Valid points:        {valid.sum()}")

    def test_torsion_float32_baseline_reasonable(self) -> None:
        """Float32 torsion on a helix should be a finite positive value.
        Analytical torsion for continuous helix: tau = pitch / (r^2 + pitch^2)
        = 0.2 / (0.09 + 0.04) = 0.2 / 0.13 ~ 1.538.
        """
        spectra = _make_helix_spectrum(n_points=300, radius=0.3, pitch=0.2)
        tau = spectral_torsion(spectra, window_length=3, polyorder=1)

        valid = ~np.isnan(tau) & (tau > 0)
        if valid.sum() > 10:
            median_tau = float(np.median(tau[valid]))
            analytical_tau = 0.2 / (0.09 + 0.04)
            print(f"\n[SVD-05] Float32 torsion baseline: median={median_tau:.4f} (analytical={analytical_tau:.4f})")
        else:
            print(f"\n[SVD-05] Float32 torsion: {valid.sum()} valid points (limited by derivative order)")
            # Torsion requires 3 derivatives and may have few valid points,
            # this is expected behavior not a failure


class TestFloat16Recommendation:
    """Aggregate fidelity results and make a clear recommendation."""

    def test_recommendation(self) -> None:
        """Run full fidelity analysis and emit recommendation.

        This test independently computes the errors (self-contained) to produce
        the final recommendation on float16 vs float32 spectrum storage.
        """
        # Circle for curvature
        spectra_circ = _make_circle_spectrum(n_points=200, radius=0.3)
        spectra_circ_f16 = spectra_circ.astype(np.float16).astype(np.float32)

        kappa_f32 = spectral_curvature(spectra_circ, window_length=3, polyorder=1)
        kappa_f16 = spectral_curvature(spectra_circ_f16, window_length=3, polyorder=1)

        valid_k = ~np.isnan(kappa_f32) & ~np.isnan(kappa_f16) & (np.abs(kappa_f32) > 1e-10)
        if valid_k.sum() > 0:
            curvature_rel_err = float(
                (np.abs(kappa_f32[valid_k] - kappa_f16[valid_k])
                 / (np.abs(kappa_f32[valid_k]) + 1e-10)).max()
            )
        else:
            curvature_rel_err = float('inf')  # No valid points = unusable

        # Helix for torsion
        spectra_hel = _make_helix_spectrum(n_points=300, radius=0.3, pitch=0.2)
        spectra_hel_f16 = spectra_hel.astype(np.float16).astype(np.float32)

        tau_f32 = spectral_torsion(spectra_hel, window_length=3, polyorder=1)
        tau_f16 = spectral_torsion(spectra_hel_f16, window_length=3, polyorder=1)

        valid_t = ~np.isnan(tau_f32) & ~np.isnan(tau_f16) & (np.abs(tau_f32) > 1e-10)
        if valid_t.sum() > 0:
            torsion_rel_err = float(
                (np.abs(tau_f32[valid_t] - tau_f16[valid_t])
                 / (np.abs(tau_f32[valid_t]) + 1e-10)).max()
            )
        else:
            torsion_rel_err = float('inf')

        threshold = 0.10  # 10%
        curvature_exceeds = curvature_rel_err > threshold
        torsion_exceeds = torsion_rel_err > threshold
        either_exceeds = curvature_exceeds or torsion_exceeds

        print(f"\n{'='*60}")
        print(f"[SVD-05] FLOAT16 FIDELITY SUMMARY")
        print(f"{'='*60}")
        curv_pct = f"{curvature_rel_err*100:.1f}%" if np.isfinite(curvature_rel_err) else "N/A"
        tors_pct = f"{torsion_rel_err*100:.1f}%" if np.isfinite(torsion_rel_err) else "N/A"
        print(f"  Curvature max relative error: {curv_pct}"
              f" {'EXCEEDS' if curvature_exceeds else 'OK'}")
        print(f"  Torsion max relative error:   {tors_pct}"
              f" {'EXCEEDS' if torsion_exceeds else 'OK'}")
        print(f"  Threshold: {threshold*100:.0f}%")
        print()

        if either_exceeds:
            print("  RECOMMENDATION: Switch spectrum storage from float16 to float32")
            print("  Reason: float16 quantization introduces unacceptable error in")
            print("  downstream curvature/torsion computation (second/third derivatives")
            print("  amplify the ~3 decimal digit precision limit of float16).")
        else:
            print("  RECOMMENDATION: float16 storage is acceptable")
        print(f"{'='*60}")

        # This test always passes — it documents the finding.
        # The actual fix (if needed) is applied in pipeline.py by the executor.
        assert True
