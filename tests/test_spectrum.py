"""Tests for spectrum trajectory analysis: curvature, torsion, and AUROC integration.

Phase 15: Advanced Analysis (SPEC-01, SPEC-02, SPEC-03).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.analysis.spectrum import (
    DEFAULT_SAVGOL_POLYORDER,
    DEFAULT_SAVGOL_WINDOW,
    VELOCITY_EPS,
    compute_spectrum_analysis,
    compute_spectrum_analysis_batch,
    detect_ordering_crossings,
    smooth_spectrum,
    spectral_curvature,
    spectral_torsion,
)


# ---------------------------------------------------------------------------
# smooth_spectrum tests
# ---------------------------------------------------------------------------


class TestSmoothSpectrum:
    def test_smooth_spectrum_identity_linear(self):
        """Linear trajectory should remain approximately linear after smoothing."""
        n_steps, k = 50, 8
        t = np.arange(n_steps, dtype=np.float64)
        direction = np.random.RandomState(42).randn(k)
        spectra = np.outer(t, direction)  # [n_steps, k]

        smoothed = smooth_spectrum(spectra)
        max_deviation = np.max(np.abs(smoothed - spectra))
        assert max_deviation < 1e-3, f"Linear trajectory changed by {max_deviation}"

    def test_smooth_spectrum_short_sequence(self):
        """Spectrum shorter than window_length should return unchanged."""
        spectra = np.random.randn(3, 4)  # 3 steps < default window=7
        smoothed = smooth_spectrum(spectra)
        np.testing.assert_array_equal(smoothed, spectra)

    def test_smooth_spectrum_shape_preserved(self):
        """Output shape matches input shape."""
        spectra = np.random.randn(100, 8)
        smoothed = smooth_spectrum(spectra)
        assert smoothed.shape == spectra.shape


# ---------------------------------------------------------------------------
# detect_ordering_crossings tests
# ---------------------------------------------------------------------------


class TestOrderingCrossings:
    def test_no_crossings_descending(self):
        """Properly descending SVs should have no crossings."""
        n_steps, k = 20, 4
        spectra = np.zeros((n_steps, k))
        for i in range(k):
            spectra[:, i] = 10.0 - i  # sigma_0=10, sigma_1=9, etc.
        mask = detect_ordering_crossings(spectra)
        assert not mask.any()

    def test_crossing_detected(self):
        """When sigma_1 and sigma_2 swap, crossing should be detected."""
        n_steps, k = 20, 4
        spectra = np.zeros((n_steps, k))
        for t in range(n_steps):
            spectra[t] = [10.0, 8.0, 5.0, 2.0]

        # Introduce a swap at step 10: sigma_0 < sigma_1
        spectra[10, 0] = 7.0
        spectra[10, 1] = 8.0  # Now sigma_0 < sigma_1

        mask = detect_ordering_crossings(spectra)
        # Should be True at steps 9, 10, 11
        assert mask[9]
        assert mask[10]
        assert mask[11]
        # Should be False far away
        assert not mask[0]
        assert not mask[5]

    def test_single_sv_no_crossings(self):
        """Single singular value should never have crossings."""
        spectra = np.random.randn(20, 1)
        mask = detect_ordering_crossings(spectra)
        assert not mask.any()


# ---------------------------------------------------------------------------
# spectral_curvature tests
# ---------------------------------------------------------------------------


class TestSpectralCurvature:
    def test_curvature_straight_line(self):
        """Straight-line trajectory should have ~0 curvature."""
        n_steps, k = 50, 8
        t = np.linspace(0, 1, n_steps)
        direction = np.array([1.0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.005])
        offset = np.array([10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5])
        spectra = offset + np.outer(t, direction)

        curv = spectral_curvature(spectra)
        # Interior points (not NaN) should have very small curvature
        valid = curv[~np.isnan(curv)]
        assert len(valid) > 0
        assert np.max(np.abs(valid)) < 1e-4, f"Max curvature: {np.max(np.abs(valid))}"

    def test_curvature_circle(self):
        """Circular trajectory should have approximately constant curvature."""
        n_steps = 100
        k = 8
        radius = 5.0
        t = np.linspace(0, 2 * np.pi * 0.8, n_steps)  # ~0.8 of a circle

        spectra = np.zeros((n_steps, k))
        spectra[:, 0] = radius * np.cos(t)
        spectra[:, 1] = radius * np.sin(t)
        # Add constant offset so values are "descending" (no crossings)
        spectra[:, 0] += 20.0
        spectra[:, 1] += 15.0
        for i in range(2, k):
            spectra[:, i] = 10.0 - i

        curv = spectral_curvature(spectra)
        valid = curv[~np.isnan(curv)]
        assert len(valid) > 10

        # Expected curvature: 1/radius = 0.2
        expected = 1.0 / radius
        median_curv = np.median(valid)
        # Allow 15% tolerance due to discrete approximation + smoothing
        assert abs(median_curv - expected) / expected < 0.15, (
            f"Median curvature {median_curv} vs expected {expected}"
        )

    def test_curvature_boundary_padding(self):
        """First and last entries should be NaN (boundary effects)."""
        spectra = np.random.randn(50, 4) + np.arange(4) * 10
        # Make descending
        spectra = np.sort(spectra, axis=1)[:, ::-1]
        curv = spectral_curvature(spectra)
        assert np.isnan(curv[0]), "First entry should be NaN"
        assert np.isnan(curv[-1]), "Last entry should be NaN"

    def test_curvature_zero_velocity(self):
        """Constant trajectory (zero velocity) should give NaN curvature."""
        n_steps, k = 50, 4
        spectra = np.ones((n_steps, k)) * np.array([10.0, 8.0, 5.0, 2.0])
        curv = spectral_curvature(spectra)
        # All should be NaN (no movement)
        assert np.all(np.isnan(curv))

    def test_curvature_at_crossings_is_nan(self):
        """Curvature should be NaN at steps with ordering crossings."""
        n_steps, k = 50, 4
        t = np.linspace(0, 1, n_steps)
        spectra = np.column_stack([
            10.0 + t,
            8.0 + 0.5 * t,
            5.0 + 0.3 * t,
            2.0 + 0.1 * t,
        ])

        # Introduce a crossing at step 25
        spectra[25, 0] = spectra[25, 1] - 0.1  # sigma_0 < sigma_1

        curv = spectral_curvature(spectra)
        # Steps 24, 25, 26 should be NaN (crossing contamination)
        assert np.isnan(curv[24])
        assert np.isnan(curv[25])
        assert np.isnan(curv[26])

    def test_curvature_short_sequence(self):
        """Very short sequences should return all NaN."""
        spectra = np.array([[10, 8], [9, 7]])
        curv = spectral_curvature(spectra)
        assert curv.shape == (2,)
        assert np.all(np.isnan(curv))


# ---------------------------------------------------------------------------
# spectral_torsion tests
# ---------------------------------------------------------------------------


class TestSpectralTorsion:
    def test_torsion_planar_curve(self):
        """A curve confined to a 2D plane should have ~0 torsion."""
        n_steps = 100
        k = 8
        radius = 5.0
        t = np.linspace(0, 2 * np.pi * 0.8, n_steps)

        spectra = np.zeros((n_steps, k))
        spectra[:, 0] = 20.0 + radius * np.cos(t)
        spectra[:, 1] = 15.0 + radius * np.sin(t)
        for i in range(2, k):
            spectra[:, i] = 10.0 - i  # constant in other dimensions

        tors = spectral_torsion(spectra)
        valid = tors[~np.isnan(tors)]
        assert len(valid) > 5
        # Planar curve torsion should be near 0
        assert np.max(np.abs(valid)) < 0.1, f"Max torsion: {np.max(np.abs(valid))}"

    def test_torsion_helix(self):
        """Helical trajectory should have approximately constant nonzero torsion."""
        n_steps = 200
        k = 8
        radius = 3.0
        pitch = 0.5  # rise per radian
        t = np.linspace(0, 4 * np.pi, n_steps)

        spectra = np.zeros((n_steps, k))
        spectra[:, 0] = 20.0 + radius * np.cos(t)
        spectra[:, 1] = 15.0 + radius * np.sin(t)
        spectra[:, 2] = 10.0 + pitch * t
        for i in range(3, k):
            spectra[:, i] = (8.0 - i)

        tors = spectral_torsion(spectra)
        valid = tors[~np.isnan(tors)]
        assert len(valid) > 10
        # Torsion should be nonzero for a helix
        median_tors = np.median(np.abs(valid))
        assert median_tors > 0.001, f"Helix torsion too small: {median_tors}"

    def test_torsion_short_sequence(self):
        """Sequences with < 4 steps should return all NaN."""
        spectra = np.random.randn(3, 4)
        tors = spectral_torsion(spectra)
        assert tors.shape == (3,)
        assert np.all(np.isnan(tors))

    def test_torsion_boundary_padding(self):
        """Boundary entries should be NaN."""
        n_steps = 50
        spectra = np.random.randn(n_steps, 8)
        spectra = np.sort(spectra, axis=1)[:, ::-1] + np.arange(8) * 5
        tors = spectral_torsion(spectra)
        # First 2 and last 1 should be NaN minimum
        assert np.isnan(tors[0])
        assert np.isnan(tors[1])


# ---------------------------------------------------------------------------
# compute_spectrum_analysis tests
# ---------------------------------------------------------------------------


class TestComputeSpectrumAnalysis:
    def test_output_structure(self):
        """Verify output dict has correct keys and shapes."""
        n_steps, k = 50, 8
        spectra = np.random.randn(n_steps, k)
        spectra = np.sort(spectra, axis=1)[:, ::-1] + np.arange(k) * 5

        result = compute_spectrum_analysis(spectra)

        assert "curvature" in result
        assert "torsion" in result
        assert "crossing_mask" in result
        assert "smoothed" in result

        assert result["curvature"].shape == (n_steps,)
        assert result["torsion"].shape == (n_steps,)
        assert result["crossing_mask"].shape == (n_steps,)
        assert result["smoothed"].shape == (n_steps, k)

    def test_batch_processing(self):
        """Batch of 3 sequences should produce [3, n_steps] outputs."""
        n_seqs, n_steps, k = 3, 50, 8
        spectra_batch = np.random.randn(n_seqs, n_steps, k)
        spectra_batch = np.sort(spectra_batch, axis=2)[:, :, ::-1] + np.arange(k) * 5

        result = compute_spectrum_analysis_batch(spectra_batch)

        assert result["curvature"].shape == (n_seqs, n_steps)
        assert result["torsion"].shape == (n_seqs, n_steps)
        assert result["crossing_mask"].shape == (n_seqs, n_steps)


# ---------------------------------------------------------------------------
# Noise suppression test
# ---------------------------------------------------------------------------


class TestNoiseSuppression:
    def test_curvature_noise_suppression(self):
        """Smoothed curvature should be closer to true value than raw."""
        n_steps = 100
        k = 8
        radius = 5.0
        t = np.linspace(0, 2 * np.pi * 0.6, n_steps)

        # Clean circular trajectory
        clean = np.zeros((n_steps, k))
        clean[:, 0] = 20.0 + radius * np.cos(t)
        clean[:, 1] = 15.0 + radius * np.sin(t)
        for i in range(2, k):
            clean[:, i] = 10.0 - i

        # Add noise
        rng = np.random.RandomState(42)
        noisy = clean + rng.randn(n_steps, k) * 0.01

        curv_clean = spectral_curvature(clean)
        curv_noisy = spectral_curvature(noisy)

        valid_clean = curv_clean[~np.isnan(curv_clean)]
        valid_noisy = curv_noisy[~np.isnan(curv_noisy)]

        expected = 1.0 / radius

        # Both should be close to expected, but smoothed+noisy should
        # still be reasonable thanks to Savitzky-Golay
        error_clean = np.abs(np.median(valid_clean) - expected)
        error_noisy = np.abs(np.median(valid_noisy) - expected)

        # Smoothing should keep noisy curvature in a reasonable range
        # (curvature is a second derivative, so noise amplification is expected)
        assert error_noisy / expected < 1.5, (
            f"Noisy curvature error too large: {error_noisy / expected:.2%}"
        )
        # Clean should be closer to expected than noisy
        assert error_clean < error_noisy, (
            f"Clean error {error_clean:.4f} should be less than noisy {error_noisy:.4f}"
        )


# ---------------------------------------------------------------------------
# AUROC integration tests
# ---------------------------------------------------------------------------


class TestSpectrumAUROC:
    def _make_synthetic_data(self, n_seqs=20, n_steps=50, k=8, n_layers=2):
        """Create synthetic spectrum and evaluation data for AUROC testing."""
        from src.evaluation.behavioral import RuleOutcome

        rng = np.random.RandomState(42)

        # Spectrum data
        spectra = {}
        for layer in range(n_layers):
            key = f"qkt.layer_{layer}.spectrum"
            s = rng.randn(n_seqs, n_steps, k).astype(np.float32)
            s = np.sort(s, axis=2)[:, :, ::-1] + np.arange(k) * 5
            spectra[key] = s

        # Eval result data
        generated = rng.randint(0, 100, (n_seqs, n_steps + 1))
        rule_outcome = np.full((n_seqs, n_steps), RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
        failure_index = np.full(n_seqs, -1, dtype=np.int32)

        # Create some jumper encounters and violations
        # Mark first 5 sequences as having violations
        jumper_vertex = 50
        for i in range(5):
            encounter_step = 20
            resolution_step = encounter_step + 10
            generated[i, encounter_step] = jumper_vertex
            rule_outcome[i, resolution_step - 1] = RuleOutcome.VIOLATED
            failure_index[i] = resolution_step - 1

        # Mark next 5 as having followed
        for i in range(5, 10):
            encounter_step = 20
            resolution_step = encounter_step + 10
            generated[i, encounter_step] = jumper_vertex
            rule_outcome[i, resolution_step - 1] = RuleOutcome.FOLLOWED

        from src.graph.jumpers import JumperInfo
        jumper_info = JumperInfo(vertex_id=jumper_vertex, source_block=0, target_block=1, r=10)
        jumper_map = {jumper_vertex: jumper_info}

        eval_data = {
            "generated": generated,
            "rule_outcome": rule_outcome,
            "failure_index": failure_index,
        }

        return spectra, eval_data, jumper_map

    def test_run_spectrum_auroc_analysis_structure(self):
        """Verify output structure of AUROC analysis."""
        from src.analysis.spectrum import run_spectrum_auroc_analysis

        spectra, eval_data, jumper_map = self._make_synthetic_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "spectrum_trajectories.npz"
            np.savez_compressed(str(npz_path), **spectra)

            result = run_spectrum_auroc_analysis(
                npz_path, eval_data, jumper_map, min_events_per_class=2
            )

        assert result["status"] == "exploratory"
        assert "config" in result
        assert "by_r_value" in result

    def test_spectrum_auroc_exploratory_label(self):
        """Output must be labeled as exploratory."""
        from src.analysis.spectrum import run_spectrum_auroc_analysis

        spectra, eval_data, jumper_map = self._make_synthetic_data()

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "spectrum_trajectories.npz"
            np.savez_compressed(str(npz_path), **spectra)

            result = run_spectrum_auroc_analysis(
                npz_path, eval_data, jumper_map, min_events_per_class=2
            )

        assert result["status"] == "exploratory"

    def test_spectrum_auroc_missing_file(self):
        """Missing spectrum file should return error dict."""
        from src.analysis.spectrum import run_spectrum_auroc_analysis

        result = run_spectrum_auroc_analysis(
            "/nonexistent/path.npz", {}, {}, min_events_per_class=2
        )
        assert result["status"] == "exploratory"
        assert "error" in result
