"""Unit tests for SVD metric functions against analytically known matrices.

Tests verify correctness using matrices with known SVD decompositions:
identity, rank-1, diagonal, and known-condition matrices.
"""

import math

import pytest
import torch

from src.evaluation.svd_metrics import (
    CONDITION_CAP,
    EPS,
    compute_all_metrics,
    condition_number,
    grassmannian_distance,
    guard_matrix_for_svd,
    rank1_residual_norm,
    read_write_alignment,
    spectral_entropy,
    spectral_gap_1_2,
    spectral_gap_2_3,
    spectral_gap_4_5,
    stable_rank,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_diagonal_matrix(singular_values: list[float]) -> torch.Tensor:
    """Create a matrix with known singular values via diagonal construction."""
    return torch.diag(torch.tensor(singular_values, dtype=torch.float64))


def _svd_vals(matrix: torch.Tensor) -> torch.Tensor:
    """Get singular values of a matrix."""
    return torch.linalg.svdvals(matrix)


# ---------------------------------------------------------------------------
# TestStableRank
# ---------------------------------------------------------------------------
class TestStableRank:
    """stable_rank: sum(s_i^2) / s_1^2."""

    def test_identity_matrix(self):
        """Identity (n x n): all s_i = 1, stable rank = n."""
        n = 5
        S = torch.ones(n, dtype=torch.float64)
        result = stable_rank(S)
        assert torch.isclose(result, torch.tensor(float(n), dtype=torch.float64), atol=1e-6)

    def test_rank1_matrix(self):
        """Rank-1 matrix: only s_1 > 0, stable rank = 1.0."""
        S = torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        result = stable_rank(S)
        assert torch.isclose(result, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

    def test_known_diagonal(self):
        """Diagonal with known values: stable rank = sum(s_i^2) / s_1^2."""
        S = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
        expected = (16.0 + 9.0 + 4.0 + 1.0) / 16.0  # 30/16 = 1.875
        result = stable_rank(S)
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-6)

    def test_uniform_singular_values(self):
        """Uniform s_i = c: stable rank = n."""
        n = 8
        S = torch.full((n,), 3.0, dtype=torch.float64)
        result = stable_rank(S)
        assert torch.isclose(result, torch.tensor(float(n), dtype=torch.float64), atol=1e-6)


# ---------------------------------------------------------------------------
# TestSpectralEntropy
# ---------------------------------------------------------------------------
class TestSpectralEntropy:
    """spectral_entropy: -sum(p_i * log(p_i)) where p_i = s_i / sum(s)."""

    def test_identity_matrix(self):
        """Identity: all p_i equal, entropy = log(n)."""
        n = 5
        S = torch.ones(n, dtype=torch.float64)
        result = spectral_entropy(S)
        expected = math.log(n)
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-4)

    def test_rank1_matrix(self):
        """Rank-1: one dominant singular value, entropy ~ 0."""
        S = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        result = spectral_entropy(S)
        # p = [1, 0, 0] -> entropy = 0 (with eps guard)
        assert result >= 0.0
        assert result < 0.1  # close to 0

    def test_uniform_singular_values(self):
        """Uniform [1, 1, 1]: entropy = log(3)."""
        S = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        result = spectral_entropy(S)
        expected = math.log(3)
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-4)

    def test_entropy_non_negative(self):
        """Entropy must always be >= 0."""
        S = torch.tensor([5.0, 0.001, 0.0001], dtype=torch.float64)
        result = spectral_entropy(S)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# TestSpectralGaps
# ---------------------------------------------------------------------------
class TestSpectralGaps:
    """Spectral gaps: differences between consecutive singular values."""

    def test_known_gaps(self):
        """Known singular values [5, 3, 2, 1, 0.5]: gaps = 2, 1, 0.5."""
        S = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
        assert torch.isclose(spectral_gap_1_2(S), torch.tensor(2.0, dtype=torch.float64), atol=1e-6)
        assert torch.isclose(spectral_gap_2_3(S), torch.tensor(1.0, dtype=torch.float64), atol=1e-6)
        assert torch.isclose(spectral_gap_4_5(S), torch.tensor(0.5, dtype=torch.float64), atol=1e-6)

    def test_equal_singular_values(self):
        """All equal singular values: all gaps = 0."""
        S = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
        assert torch.isclose(spectral_gap_1_2(S), torch.tensor(0.0, dtype=torch.float64), atol=1e-6)
        assert torch.isclose(spectral_gap_2_3(S), torch.tensor(0.0, dtype=torch.float64), atol=1e-6)
        assert torch.isclose(spectral_gap_4_5(S), torch.tensor(0.0, dtype=torch.float64), atol=1e-6)

    def test_decreasing_gaps(self):
        """Linearly decreasing: [10, 8, 6, 4, 2] -> gaps all = 2."""
        S = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0], dtype=torch.float64)
        assert torch.isclose(spectral_gap_1_2(S), torch.tensor(2.0, dtype=torch.float64), atol=1e-6)
        assert torch.isclose(spectral_gap_2_3(S), torch.tensor(2.0, dtype=torch.float64), atol=1e-6)
        assert torch.isclose(spectral_gap_4_5(S), torch.tensor(2.0, dtype=torch.float64), atol=1e-6)


# ---------------------------------------------------------------------------
# TestConditionNumber
# ---------------------------------------------------------------------------
class TestConditionNumber:
    """condition_number: sigma_1 / sigma_n, capped at CONDITION_CAP."""

    def test_identity(self):
        """Identity matrix: condition = 1.0."""
        S = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        result = condition_number(S)
        assert torch.isclose(result, torch.tensor(1.0, dtype=torch.float64), atol=1e-6)

    def test_known_diagonal(self):
        """Known diagonal [10, 1]: condition = 10.0."""
        S = torch.tensor([10.0, 1.0], dtype=torch.float64)
        result = condition_number(S)
        assert torch.isclose(result, torch.tensor(10.0, dtype=torch.float64), atol=1e-4)

    def test_near_singular_capped(self):
        """Near-singular (sigma_n ~ 0): condition capped at CONDITION_CAP."""
        S = torch.tensor([10.0, 1e-15], dtype=torch.float64)
        result = condition_number(S)
        assert torch.isclose(result, torch.tensor(CONDITION_CAP, dtype=torch.float64), atol=1.0)

    def test_well_conditioned(self):
        """Well-conditioned matrix: condition < CONDITION_CAP."""
        S = torch.tensor([5.0, 2.0], dtype=torch.float64)
        result = condition_number(S)
        assert torch.isclose(result, torch.tensor(2.5, dtype=torch.float64), atol=1e-4)


# ---------------------------------------------------------------------------
# TestRank1ResidualNorm
# ---------------------------------------------------------------------------
class TestRank1ResidualNorm:
    """rank1_residual_norm: ||M - sigma_1*u_1*v_1^T||_F / ||M||_F."""

    def test_rank1_matrix(self):
        """Rank-1 matrix: residual = 0.0 (or very close with eps)."""
        M = _make_diagonal_matrix([3.0, 0.0, 0.0])
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        result = rank1_residual_norm(U, S, Vh)
        assert result < 0.01  # effectively 0 with eps guard

    def test_known_svd(self):
        """Known SVD with s = [3, 2, 1]: residual = sqrt(4+1) / sqrt(9+4+1)."""
        M = _make_diagonal_matrix([3.0, 2.0, 1.0])
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        expected = math.sqrt(4.0 + 1.0) / math.sqrt(9.0 + 4.0 + 1.0)
        result = rank1_residual_norm(U, S, Vh)
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-4)

    def test_two_equal_singular_values(self):
        """s = [5, 5, 0]: residual = sqrt(25) / sqrt(50) = 1/sqrt(2)."""
        M = _make_diagonal_matrix([5.0, 5.0, 0.0])
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        expected = math.sqrt(25.0) / math.sqrt(50.0)
        result = rank1_residual_norm(U, S, Vh)
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-4)


# ---------------------------------------------------------------------------
# TestReadWriteAlignment
# ---------------------------------------------------------------------------
class TestReadWriteAlignment:
    """read_write_alignment: |cos(angle)| between top left and right singular vectors."""

    def test_symmetric_matrix_alignment_one(self):
        """Symmetric positive definite matrix: U and V are the same, alignment = 1.0."""
        # For a diagonal matrix, U = I, Vh = I -> u_1 = e_1, v_1 = e_1 -> alignment = 1
        M = _make_diagonal_matrix([5.0, 3.0, 1.0])
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        result = read_write_alignment(U, Vh)
        assert torch.isclose(result, torch.tensor(1.0, dtype=torch.float64), atol=1e-4)

    def test_orthogonal_singular_vectors(self):
        """Construct matrix where top left and right singular vectors are orthogonal."""
        # Create a matrix U @ diag(S) @ Vh where u_1 orthogonal to v_1
        # Use: u_1 = [1, 0, 0], v_1 = [0, 1, 0]
        U = torch.eye(3, dtype=torch.float64)
        S = torch.tensor([5.0, 3.0, 1.0], dtype=torch.float64)
        # Vh where first row is [0, 1, 0] (orthogonal to u_1 = [1, 0, 0])
        Vh = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float64)
        result = read_write_alignment(U, Vh)
        assert torch.isclose(result, torch.tensor(0.0, dtype=torch.float64), atol=1e-6)


# ---------------------------------------------------------------------------
# TestGrassmannianDistance
# ---------------------------------------------------------------------------
class TestGrassmannianDistance:
    """grassmannian_distance: geodesic distance on Grassmann manifold."""

    def test_same_subspace(self):
        """Same subspace: distance = 0."""
        U = torch.eye(5, dtype=torch.float64)
        result = grassmannian_distance(U, U, k=2)
        assert torch.isclose(result, torch.tensor(0.0, dtype=torch.float64), atol=1e-6)

    def test_orthogonal_subspaces(self):
        """Orthogonal subspaces: distance = pi/2 * sqrt(k)."""
        k = 2
        # U_prev spans columns 0, 1; U_curr spans columns 2, 3
        U_prev = torch.zeros(5, 4, dtype=torch.float64)
        U_prev[0, 0] = 1.0
        U_prev[1, 1] = 1.0
        U_curr = torch.zeros(5, 4, dtype=torch.float64)
        U_curr[2, 0] = 1.0
        U_curr[3, 1] = 1.0
        result = grassmannian_distance(U_prev, U_curr, k=k)
        expected = (math.pi / 2) * math.sqrt(k)
        assert torch.isclose(result, torch.tensor(expected, dtype=torch.float64), atol=1e-4)

    def test_partial_overlap(self):
        """Partially overlapping subspaces: 0 < distance < pi/2*sqrt(k)."""
        k = 2
        U_prev = torch.eye(5, 4, dtype=torch.float64)
        # Rotate first column by 45 degrees in the (0,2) plane
        angle = math.pi / 4
        U_curr = torch.eye(5, 4, dtype=torch.float64)
        U_curr[0, 0] = math.cos(angle)
        U_curr[2, 0] = math.sin(angle)
        result = grassmannian_distance(U_prev, U_curr, k=k)
        assert result > 0.0
        assert result < (math.pi / 2) * math.sqrt(k)


# ---------------------------------------------------------------------------
# TestNumericalGuards
# ---------------------------------------------------------------------------
class TestNumericalGuards:
    """guard_matrix_for_svd: NaN/Inf input clamping and guard activation reporting."""

    def test_nan_input_returns_cleaned(self):
        """NaN input: returns cleaned matrix with guard_activated=True."""
        M = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
        cleaned, activated = guard_matrix_for_svd(M)
        assert activated is True
        assert torch.isfinite(cleaned).all()

    def test_finite_input_unchanged(self):
        """Finite input: returns unchanged matrix with guard_activated=False."""
        M = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        cleaned, activated = guard_matrix_for_svd(M)
        assert activated is False
        assert torch.equal(cleaned, M)

    def test_inf_input_clamped(self):
        """Inf input: clamped to 1e6."""
        M = torch.tensor([[float("inf"), 1.0], [0.0, float("-inf")]])
        cleaned, activated = guard_matrix_for_svd(M)
        assert activated is True
        assert torch.isfinite(cleaned).all()
        assert cleaned[0, 0] == 1e6
        assert cleaned[1, 1] == -1e6

    def test_zero_singular_values_no_nan(self):
        """Zero singular values don't cause NaN in any metric."""
        S = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        # All S-only metrics should produce finite output
        assert torch.isfinite(stable_rank(S))
        assert torch.isfinite(spectral_entropy(S))
        assert torch.isfinite(spectral_gap_1_2(S))
        assert torch.isfinite(spectral_gap_2_3(S))
        assert torch.isfinite(spectral_gap_4_5(S))
        assert torch.isfinite(condition_number(S))


# ---------------------------------------------------------------------------
# TestBatchedSVD
# ---------------------------------------------------------------------------
class TestBatchedSVD:
    """Batched input produces correct per-element metrics."""

    def test_batched_singular_values(self):
        """Batched [B, k] input produces [B] output."""
        B = 4
        k = 5
        S = torch.rand(B, k, dtype=torch.float64).sort(dim=-1, descending=True).values
        result = stable_rank(S)
        assert result.shape == (B,)
        # Verify each element matches unbatched
        for i in range(B):
            single = stable_rank(S[i])
            assert torch.isclose(result[i], single, atol=1e-6)

    def test_batched_entropy(self):
        """Batched spectral entropy maintains correct shape."""
        B = 3
        S = torch.tensor([
            [5.0, 3.0, 1.0],
            [1.0, 1.0, 1.0],
            [10.0, 0.0, 0.0],
        ], dtype=torch.float64)
        result = spectral_entropy(S)
        assert result.shape == (B,)

    def test_batched_3d(self):
        """3D batched input [B, T, k] produces [B, T] output."""
        B, T, k = 2, 3, 5
        S = torch.rand(B, T, k, dtype=torch.float64).sort(dim=-1, descending=True).values
        result = stable_rank(S)
        assert result.shape == (B, T)


# ---------------------------------------------------------------------------
# TestComputeAllMetrics
# ---------------------------------------------------------------------------
class TestComputeAllMetrics:
    """compute_all_metrics returns dict with all metric keys."""

    def test_returns_correct_keys_s_only(self):
        """S-only: returns 6 singular-value metrics."""
        S = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
        result = compute_all_metrics(S)
        expected_keys = {
            "stable_rank",
            "spectral_entropy",
            "spectral_gap_1_2",
            "spectral_gap_2_3",
            "spectral_gap_4_5",
            "condition_number",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_returns_correct_keys_with_u_vh(self):
        """With U and Vh: returns additional full-SVD metrics."""
        M = _make_diagonal_matrix([5.0, 3.0, 2.0, 1.0, 0.5])
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        result = compute_all_metrics(S, U=U, Vh=Vh)
        expected_keys = {
            "stable_rank",
            "spectral_entropy",
            "spectral_gap_1_2",
            "spectral_gap_2_3",
            "spectral_gap_4_5",
            "condition_number",
            "rank1_residual_norm",
            "read_write_alignment",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_values_match_individual_functions(self):
        """Convenience function values match individual function outputs."""
        S = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
        result = compute_all_metrics(S)
        assert torch.isclose(result["stable_rank"], stable_rank(S), atol=1e-6)
        assert torch.isclose(result["spectral_entropy"], spectral_entropy(S), atol=1e-6)
        assert torch.isclose(result["spectral_gap_1_2"], spectral_gap_1_2(S), atol=1e-6)
        assert torch.isclose(result["condition_number"], condition_number(S), atol=1e-6)

    def test_with_full_svd_values_match(self):
        """Full SVD metrics match individual function outputs."""
        M = _make_diagonal_matrix([5.0, 3.0, 2.0, 1.0, 0.5])
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        result = compute_all_metrics(S, U=U, Vh=Vh)
        assert torch.isclose(
            result["rank1_residual_norm"],
            rank1_residual_norm(U, S, Vh),
            atol=1e-6,
        )
        assert torch.isclose(
            result["read_write_alignment"],
            read_write_alignment(U, Vh),
            atol=1e-6,
        )
