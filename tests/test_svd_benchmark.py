"""Tests for SVD computational overhead benchmarking.

Tests timing, accuracy comparison, and orchestration functions.
GPU-specific tests are skipped when CUDA is not available.
"""

import numpy as np
import pytest
import torch

from src.analysis.svd_benchmark import (
    _compare_accuracy,
    _full_svd,
    _randomized_svd,
    _time_svd_method,
    _values_only_svd,
    benchmark_svd_for_target,
    run_svd_benchmark,
)
from src.config.experiment import ExperimentConfig


class TestSvdWrappers:
    """Tests for SVD method wrapper functions."""

    def test_full_svd_shapes(self):
        """Full SVD returns (U, S, Vh) with correct shapes."""
        M = torch.randn(10, 8)
        U, S, Vh = _full_svd(M)
        assert U.shape == (10, 8)
        assert S.shape == (8,)
        assert Vh.shape == (8, 8)

    def test_randomized_svd_shapes(self):
        """Randomized SVD returns (U, S, V) with expected shapes."""
        M = torch.randn(10, 8)
        U, S, V = _randomized_svd(M)
        # svd_lowrank returns q singular values where q = min(m, n)
        q = min(10, 8)
        assert U.shape[0] == 10
        assert len(S.shape) == 1
        assert S.shape[0] <= q
        assert V.shape[0] == 8

    def test_values_only_svd_shape(self):
        """Values-only SVD returns S with correct length."""
        M = torch.randn(10, 8)
        S = _values_only_svd(M)
        assert S.shape == (8,)  # min(10, 8)

    def test_full_svd_square_matrix(self):
        """Full SVD works on square matrices."""
        M = torch.randn(16, 16)
        U, S, Vh = _full_svd(M)
        assert U.shape == (16, 16)
        assert S.shape == (16,)
        assert Vh.shape == (16, 16)


class TestTimeSvdMethod:
    """Tests for SVD timing function."""

    def test_time_svd_method_cpu(self):
        """Timing on CPU returns positive float ms."""
        M = torch.randn(20, 20)
        ms = _time_svd_method(M, _full_svd, n_warmup=2, n_timed=5)
        assert isinstance(ms, float)
        assert ms > 0

    def test_time_values_only_cpu(self):
        """Values-only timing on CPU returns positive float ms."""
        M = torch.randn(20, 20)
        ms = _time_svd_method(M, _values_only_svd, n_warmup=2, n_timed=5)
        assert isinstance(ms, float)
        assert ms > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_time_svd_method_cuda(self):
        """Timing on CUDA returns positive float ms."""
        M = torch.randn(64, 64, device="cuda")
        ms = _time_svd_method(M, _full_svd, n_warmup=3, n_timed=10)
        assert isinstance(ms, float)
        assert ms > 0


class TestCompareAccuracy:
    """Tests for SVD accuracy comparison."""

    def test_identical_singular_values(self):
        """When reference == approx, frob_error is 0 and correlation is 1."""
        S = torch.tensor([3.0, 2.0, 1.0])
        result = _compare_accuracy(S, S.clone())
        assert result["frob_error"] == pytest.approx(0.0, abs=1e-6)
        assert result["sv_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_perturbed_singular_values(self):
        """When approx is perturbed, frob_error > 0 and correlation < 1."""
        ref_S = torch.tensor([10.0, 5.0, 2.0, 1.0])
        approx_S = torch.tensor([10.0, 4.5, 2.5, 0.5])
        result = _compare_accuracy(ref_S, approx_S)
        assert result["frob_error"] > 0
        assert result["sv_correlation"] < 1.0
        assert result["sv_correlation"] > 0.5  # Should still be highly correlated

    def test_zero_reference(self):
        """All-zero reference returns NaN."""
        ref_S = torch.zeros(5)
        approx_S = torch.ones(5)
        result = _compare_accuracy(ref_S, approx_S)
        assert np.isnan(result["frob_error"])

    def test_different_lengths(self):
        """Handles different-length singular value vectors."""
        ref_S = torch.tensor([5.0, 3.0, 1.0])
        approx_S = torch.tensor([5.0, 3.0])  # Shorter
        result = _compare_accuracy(ref_S, approx_S)
        # Should compare only the first 2 values
        assert result["frob_error"] == pytest.approx(0.0, abs=1e-6)


class TestBenchmarkSvdForTarget:
    """Tests for single-target benchmarking."""

    def test_structure(self):
        """Output dict has all required keys with correct types."""
        result = benchmark_svd_for_target(
            "qkt", (16, 16), n_warmup=1, n_timed=3
        )
        assert result["target"] == "qkt"
        assert result["matrix_shape"] == [16, 16]
        assert isinstance(result["full_svd_ms"], float)
        assert isinstance(result["randomized_svd_ms"], float)
        assert isinstance(result["values_only_ms"], float)
        assert result["full_svd_ms"] > 0
        assert result["randomized_svd_ms"] > 0
        assert result["values_only_ms"] > 0

    def test_accuracy_fields(self):
        """Accuracy fields are present and reasonable."""
        result = benchmark_svd_for_target(
            "wvwo", (32, 32), n_warmup=1, n_timed=3
        )
        assert "randomized_frob_error" in result
        assert "randomized_sv_correlation" in result
        assert "values_only_sv_correlation" in result
        # Randomized should be reasonably accurate for full-rank
        assert result["randomized_sv_correlation"] > 0.9
        # Values-only should be exact (same algorithm family)
        assert result["values_only_sv_correlation"] > 0.99

    def test_reproducible_with_seed(self):
        """Same seed produces same timing results (within noise)."""
        r1 = benchmark_svd_for_target("avwo", (8, 8), seed=42, n_warmup=1, n_timed=2)
        r2 = benchmark_svd_for_target("avwo", (8, 8), seed=42, n_warmup=1, n_timed=2)
        # Accuracy should be identical (same matrix)
        assert r1["randomized_frob_error"] == pytest.approx(
            r2["randomized_frob_error"], abs=1e-6
        )


class TestRunSvdBenchmark:
    """Tests for the full benchmark orchestrator."""

    def test_structure(self):
        """Output has by_target with all three targets and summary."""
        config = ExperimentConfig()  # Default config
        result = run_svd_benchmark(config, n_warmup=1, n_timed=2)

        assert "config" in result
        assert result["config"]["n_warmup"] == 1
        assert result["config"]["n_timed"] == 2

        assert "by_target" in result
        assert set(result["by_target"].keys()) == {"qkt", "wvwo", "avwo"}

        for target_name, t_data in result["by_target"].items():
            assert "full_svd_ms" in t_data
            assert "randomized_svd_ms" in t_data
            assert "values_only_ms" in t_data
            assert "matrix_shape" in t_data

        assert "summary" in result
        assert "total_svd_ms_per_step" in result["summary"]
        assert "fastest_method" in result["summary"]
        assert result["summary"]["fastest_method"] in (
            "full_svd", "randomized_svd", "values_only"
        )

    def test_matrix_dimensions_from_config(self):
        """Matrix dimensions match config parameters."""
        config = ExperimentConfig()  # w=64, d_model=128
        result = run_svd_benchmark(config, n_warmup=1, n_timed=2)

        assert result["by_target"]["qkt"]["matrix_shape"] == [64, 64]
        assert result["by_target"]["wvwo"]["matrix_shape"] == [128, 128]
        assert result["by_target"]["avwo"]["matrix_shape"] == [64, 128]

    def test_total_includes_layers(self):
        """Total SVD ms per step accounts for n_layers."""
        config = ExperimentConfig()  # n_layers=4
        result = run_svd_benchmark(config, n_warmup=1, n_timed=2)

        # Total should be sum of per-target * n_layers
        per_target_sum = sum(
            t["full_svd_ms"] for t in result["by_target"].values()
        )
        expected_total = per_target_sum * config.model.n_layers
        assert result["summary"]["total_svd_ms_per_step"] == pytest.approx(
            expected_total, rel=1e-6
        )
