"""Audit tests for all singular-value-derived metrics (SVD-03).

Verifies each SVD metric function against analytically known singular values
with predetermined expected outputs. Tests cover stable_rank, spectral_entropy,
spectral_gap (1-2, 2-3, 4-5), condition_number, rank1_residual_norm, and
read_write_alignment.
"""

import math

import torch

from src.evaluation.svd_metrics import (
    EPS,
    CONDITION_CAP,
    stable_rank,
    spectral_entropy,
    spectral_gap_1_2,
    spectral_gap_2_3,
    spectral_gap_4_5,
    condition_number,
    rank1_residual_norm,
    read_write_alignment,
)


class TestStableRank:
    """Verify stable_rank = sum(s_i^2) / s_1^2 (effective dimensionality)."""

    def test_stable_rank_mixed_values(self) -> None:
        """S = [4, 2, 1]: stable_rank = (16+4+1)/16 = 21/16 = 1.3125.
        Stable rank measures how spread the energy is across singular values.
        """
        S = torch.tensor([4.0, 2.0, 1.0])
        expected = 21.0 / 16.0  # 1.3125
        result = stable_rank(S)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-5, rtol=1e-5)

    def test_stable_rank_uniform_equals_rank(self) -> None:
        """S = [5, 5, 5]: stable_rank = (25+25+25)/25 = 3.0.
        When all singular values are equal, stable rank equals the matrix rank.
        """
        S = torch.tensor([5.0, 5.0, 5.0])
        expected = 3.0
        result = stable_rank(S)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-5, rtol=1e-5)

    def test_stable_rank_rank1_matrix(self) -> None:
        """S = [10, 0, 0]: stable_rank = 100/100 = 1.0.
        A rank-1 matrix has all energy concentrated in the first singular value.
        """
        S = torch.tensor([10.0, 0.0, 0.0])
        expected = 1.0
        result = stable_rank(S)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-5, rtol=1e-5)


class TestSpectralEntropy:
    """Verify spectral_entropy = -sum(p_i * log(p_i)) where p_i = sigma_i / sum(sigma)."""

    def test_spectral_entropy_uniform(self) -> None:
        """S = [1, 1, 1, 1]: p = [0.25]*4, entropy = -4*(0.25*log(0.25)) = log(4).
        Uniform distribution gives maximum entropy = log(n).
        """
        S = torch.tensor([1.0, 1.0, 1.0, 1.0])
        # p_i = 1/4 for all i, entropy = -4 * (0.25 * log(0.25)) = log(4)
        expected = math.log(4)  # ~1.3863
        result = spectral_entropy(S)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-4, rtol=1e-4)

    def test_spectral_entropy_concentrated(self) -> None:
        """S = [10, 0.001, 0.001, 0.001]: nearly all mass on first SV.
        Entropy should be near 0 when distribution is concentrated.
        """
        S = torch.tensor([10.0, 0.001, 0.001, 0.001])
        result = spectral_entropy(S)
        # Entropy should be very small (near 0) since almost all mass is on s_1
        assert result.item() < 0.05, (
            f"Entropy {result.item()} should be near 0 for concentrated distribution"
        )
        assert result.item() >= 0.0, "Entropy must be non-negative"

    def test_spectral_entropy_two_equal(self) -> None:
        """S = [5, 5]: p = [0.5, 0.5], entropy = -2*(0.5*log(0.5)) = log(2).
        Two equal singular values give entropy of log(2).
        """
        S = torch.tensor([5.0, 5.0])
        expected = math.log(2)  # ~0.6931
        result = spectral_entropy(S)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-4, rtol=1e-4)


class TestSpectralGaps:
    """Verify spectral gap formulas: sigma_i - sigma_j for adjacent pairs."""

    def test_spectral_gap_1_2(self) -> None:
        """S = [10, 7, 3, 2, 1]: gap_1_2 = 10 - 7 = 3.
        Spectral gap measures separation between consecutive singular values.
        """
        S = torch.tensor([10.0, 7.0, 3.0, 2.0, 1.0])
        result = spectral_gap_1_2(S)
        torch.testing.assert_close(result, torch.tensor(3.0), atol=1e-5, rtol=1e-5)

    def test_spectral_gap_2_3(self) -> None:
        """S = [10, 7, 3, 2, 1]: gap_2_3 = 7 - 3 = 4."""
        S = torch.tensor([10.0, 7.0, 3.0, 2.0, 1.0])
        result = spectral_gap_2_3(S)
        torch.testing.assert_close(result, torch.tensor(4.0), atol=1e-5, rtol=1e-5)

    def test_spectral_gap_4_5(self) -> None:
        """S = [10, 7, 3, 2, 1]: gap_4_5 = 2 - 1 = 1."""
        S = torch.tensor([10.0, 7.0, 3.0, 2.0, 1.0])
        result = spectral_gap_4_5(S)
        torch.testing.assert_close(result, torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_spectral_gap_uniform_is_zero(self) -> None:
        """S = [5, 5, 5, 5, 5]: all gaps = 0 (uniform singular values).
        No separation between consecutive singular values.
        """
        S = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
        assert spectral_gap_1_2(S).item() == 0.0
        assert spectral_gap_2_3(S).item() == 0.0
        assert spectral_gap_4_5(S).item() == 0.0


class TestConditionNumber:
    """Verify condition_number = sigma_1 / sigma_n, capped at CONDITION_CAP."""

    def test_condition_number_well_conditioned(self) -> None:
        """S = [10, 5, 2]: condition = 10/2 = 5.0.
        The condition number measures how sensitive the matrix is to perturbation.
        """
        S = torch.tensor([10.0, 5.0, 2.0])
        expected = 5.0
        result = condition_number(S)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-5, rtol=1e-5)

    def test_condition_number_near_singular(self) -> None:
        """S = [10, 5, 0]: condition = capped at CONDITION_CAP = 1e6.
        Division by near-zero is guarded by EPS and clamped to CONDITION_CAP.
        """
        S = torch.tensor([10.0, 5.0, 0.0])
        result = condition_number(S)
        # sigma_n = 0, so raw = 10 / (0 + EPS) which is huge, capped at 1e6
        assert result.item() == CONDITION_CAP, (
            f"Expected condition number capped at {CONDITION_CAP}, got {result.item()}"
        )

    def test_condition_number_identity_like(self) -> None:
        """S = [1, 1, 1]: condition = 1/1 = 1.0.
        Identity-like matrices are perfectly conditioned.
        """
        S = torch.tensor([1.0, 1.0, 1.0])
        expected = 1.0
        result = condition_number(S)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-5, rtol=1e-5)


class TestRank1ResidualNorm:
    """Verify rank1_residual_norm = sqrt(sum(s_i^2 for i>0)) / sqrt(sum(s_i^2))."""

    def test_rank1_residual_mixed(self) -> None:
        """S = [5, 3, 1]:
        ||M||_F = sqrt(25+9+1) = sqrt(35)
        residual = sqrt(9+1) = sqrt(10)
        rank1_residual = sqrt(10)/sqrt(35) = sqrt(10/35) = sqrt(2/7) ~ 0.5345.
        Note: EPS guards add tiny bias; use generous atol.
        """
        S = torch.tensor([5.0, 3.0, 1.0])
        # Expected: sqrt(sum(s_i^2 for i>0) + EPS) / (sqrt(sum(s_i^2) + EPS) + EPS)
        # With EPS=1e-12, this is essentially sqrt(10)/sqrt(35) = sqrt(2/7)
        expected = math.sqrt(10.0) / math.sqrt(35.0)
        # Dummy U, Vh (not used in computation but required by signature)
        U = torch.eye(3)
        Vh = torch.eye(3)
        result = rank1_residual_norm(U, S, Vh)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-4, rtol=1e-4)

    def test_rank1_residual_perfect_rank1(self) -> None:
        """S = [10, 0, 0]: residual is near 0 (perfect rank-1 approximation).
        sqrt(0 + EPS) / (sqrt(100 + EPS) + EPS) ~ sqrt(EPS)/10 ~ 0.
        """
        S = torch.tensor([10.0, 0.0, 0.0])
        U = torch.eye(3)
        Vh = torch.eye(3)
        result = rank1_residual_norm(U, S, Vh)
        # Should be very close to 0 since all energy is in first SV
        assert result.item() < 1e-5, (
            f"rank1_residual for rank-1 matrix should be ~0, got {result.item()}"
        )

    def test_rank1_residual_equal_svs(self) -> None:
        """S = [1, 1, 1]: residual = sqrt(2)/sqrt(3) ~ 0.8165.
        When SVs are equal, the rank-1 approximation captures only 1/3 of the energy.
        """
        S = torch.tensor([1.0, 1.0, 1.0])
        expected = math.sqrt(2.0) / math.sqrt(3.0)
        U = torch.eye(3)
        Vh = torch.eye(3)
        result = rank1_residual_norm(U, S, Vh)
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-4, rtol=1e-4)


class TestReadWriteAlignment:
    """Verify read_write_alignment = |u_1 . v_1| (cosine of angle between top SVs)."""

    def test_alignment_parallel(self) -> None:
        """U[:,0] = [1,0,0,...], Vh[0,:] = [1,0,0,...]: alignment = |1| = 1.0.
        Perfectly aligned read and write directions.
        """
        n = 4
        U = torch.zeros(n, n)
        U[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        Vh = torch.zeros(n, n)
        Vh[0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        result = read_write_alignment(U, Vh)
        torch.testing.assert_close(result, torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_alignment_orthogonal(self) -> None:
        """U[:,0] = [1,0,0,...], Vh[0,:] = [0,1,0,...]: alignment = |0| = 0.0.
        Perfectly orthogonal read and write directions.
        """
        n = 4
        U = torch.zeros(n, n)
        U[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        Vh = torch.zeros(n, n)
        Vh[0, :] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        result = read_write_alignment(U, Vh)
        torch.testing.assert_close(result, torch.tensor(0.0), atol=1e-5, rtol=1e-5)

    def test_alignment_diagonal(self) -> None:
        """U[:,0] = [1/sqrt(2), 1/sqrt(2), 0, ...], Vh[0,:] same: alignment = 1.0.
        Aligned but rotated 45 degrees from axis -- dot product is still 1.0.
        """
        n = 4
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        U = torch.zeros(n, n)
        U[:, 0] = torch.tensor([inv_sqrt2, inv_sqrt2, 0.0, 0.0])
        Vh = torch.zeros(n, n)
        Vh[0, :] = torch.tensor([inv_sqrt2, inv_sqrt2, 0.0, 0.0])
        result = read_write_alignment(U, Vh)
        torch.testing.assert_close(result, torch.tensor(1.0), atol=1e-5, rtol=1e-5)

    def test_alignment_partial(self) -> None:
        """U[:,0] = [1,0,0,...], Vh[0,:] = [1/sqrt(2), 1/sqrt(2), 0,...]:
        alignment = |1 * 1/sqrt(2) + 0 + 0 + 0| = 1/sqrt(2) ~ 0.7071.
        Partial alignment between read and write directions.
        """
        n = 4
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        U = torch.zeros(n, n)
        U[:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        Vh = torch.zeros(n, n)
        Vh[0, :] = torch.tensor([inv_sqrt2, inv_sqrt2, 0.0, 0.0])
        result = read_write_alignment(U, Vh)
        expected = inv_sqrt2  # ~0.7071
        torch.testing.assert_close(result, torch.tensor(expected), atol=1e-5, rtol=1e-5)
