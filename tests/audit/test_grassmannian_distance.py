"""Audit tests for Grassmannian distance computation (SVD-04).

Verifies that grassmannian_distance implements the geodesic distance on the
Grassmann manifold: d = sqrt(sum(theta_i^2)) where theta_i = arccos(sigma_i)
are principal angles, per Edelman et al. (1998).

Tests cover: identical subspaces (d=0), orthogonal subspaces (d=pi/2*sqrt(k)),
known-angle rotations, varying k (1,2,3), numerical clipping, and batched input.
"""

import math

import torch
import pytest

from src.evaluation.svd_metrics import grassmannian_distance


class TestGrassmannianIdenticalSubspaces:
    """Identical subspaces must yield distance 0."""

    def test_identical_subspaces_eye(self) -> None:
        """U1 = U2 = I_{4x2}: cos_angles = svdvals(I_2) = [1, 1],
        angles = [0, 0], d = sqrt(0+0) = 0.
        """
        U = torch.eye(4, 2, dtype=torch.float64)
        dist = grassmannian_distance(U, U, k=2)
        assert dist.item() == pytest.approx(0.0, abs=1e-6), (
            f"Distance between identical subspaces should be 0, got {dist.item()}"
        )

    def test_identical_subspaces_random_orthonormal(self) -> None:
        """Random orthonormal U: distance to itself must be 0."""
        torch.manual_seed(42)
        Q, _ = torch.linalg.qr(torch.randn(6, 4, dtype=torch.float64))
        U = Q[:, :3]  # 6x3 orthonormal
        dist = grassmannian_distance(U, U, k=3)
        assert dist.item() == pytest.approx(0.0, abs=1e-6)


class TestGrassmannianOrthogonalSubspaces:
    """Orthogonal subspaces must yield d = pi/2 * sqrt(k)."""

    def test_orthogonal_k2(self) -> None:
        """U1 spans {e1, e2}, U2 spans {e3, e4}.
        cos_angles = svdvals(zeros(2,2)) = [0, 0],
        angles = [pi/2, pi/2],
        d = sqrt(2*(pi/2)^2) = pi/2 * sqrt(2).
        """
        U1 = torch.zeros(4, 2, dtype=torch.float64)
        U1[0, 0] = 1.0
        U1[1, 1] = 1.0

        U2 = torch.zeros(4, 2, dtype=torch.float64)
        U2[2, 0] = 1.0
        U2[3, 1] = 1.0

        dist = grassmannian_distance(U1, U2, k=2)
        expected = (math.pi / 2) * math.sqrt(2)
        assert dist.item() == pytest.approx(expected, abs=1e-5), (
            f"Expected pi/2*sqrt(2) = {expected:.6f}, got {dist.item():.6f}"
        )

    def test_orthogonal_k1(self) -> None:
        """U1 spans {e1}, U2 spans {e2}: d = pi/2."""
        U1 = torch.zeros(4, 1, dtype=torch.float64)
        U1[0, 0] = 1.0

        U2 = torch.zeros(4, 1, dtype=torch.float64)
        U2[1, 0] = 1.0

        dist = grassmannian_distance(U1, U2, k=1)
        expected = math.pi / 2
        assert dist.item() == pytest.approx(expected, abs=1e-5)

    def test_orthogonal_k3(self) -> None:
        """U1 spans {e1, e2, e3}, U2 spans {e4, e5, e6}: d = pi/2 * sqrt(3)."""
        U1 = torch.zeros(6, 3, dtype=torch.float64)
        U1[0, 0] = 1.0
        U1[1, 1] = 1.0
        U1[2, 2] = 1.0

        U2 = torch.zeros(6, 3, dtype=torch.float64)
        U2[3, 0] = 1.0
        U2[4, 1] = 1.0
        U2[5, 2] = 1.0

        dist = grassmannian_distance(U1, U2, k=3)
        expected = (math.pi / 2) * math.sqrt(3)
        assert dist.item() == pytest.approx(expected, abs=1e-5)


class TestGrassmannianKnownAngle:
    """Rotation by known angle theta in one plane gives d = theta."""

    def test_single_rotation_pi_over_6(self) -> None:
        """Rotate e1 by pi/6 in the (e1, e3) plane, keep e2 fixed.
        U1 = [[1,0],[0,1],[0,0],[0,0]]
        U2 = [[cos(pi/6),0],[0,1],[sin(pi/6),0],[0,0]]
        Principal angles: theta_1 = pi/6 (rotated axis), theta_2 = 0 (fixed axis).
        d = sqrt((pi/6)^2 + 0^2) = pi/6.
        """
        theta = math.pi / 6

        U1 = torch.zeros(4, 2, dtype=torch.float64)
        U1[0, 0] = 1.0  # e1
        U1[1, 1] = 1.0  # e2

        U2 = torch.zeros(4, 2, dtype=torch.float64)
        U2[0, 0] = math.cos(theta)
        U2[2, 0] = math.sin(theta)
        U2[1, 1] = 1.0  # e2 unchanged

        dist = grassmannian_distance(U1, U2, k=2)
        expected = theta
        assert dist.item() == pytest.approx(expected, abs=1e-5), (
            f"Expected pi/6 = {expected:.6f}, got {dist.item():.6f}"
        )

    def test_two_rotations(self) -> None:
        """Rotate e1 by pi/4 in (e1,e3) and e2 by pi/3 in (e2,e4).
        Two independent principal angles: pi/4 and pi/3.
        d = sqrt((pi/4)^2 + (pi/3)^2).
        """
        theta1 = math.pi / 4
        theta2 = math.pi / 3

        U1 = torch.zeros(4, 2, dtype=torch.float64)
        U1[0, 0] = 1.0
        U1[1, 1] = 1.0

        U2 = torch.zeros(4, 2, dtype=torch.float64)
        U2[0, 0] = math.cos(theta1)
        U2[2, 0] = math.sin(theta1)
        U2[1, 1] = math.cos(theta2)
        U2[3, 1] = math.sin(theta2)

        dist = grassmannian_distance(U1, U2, k=2)
        expected = math.sqrt(theta1**2 + theta2**2)
        assert dist.item() == pytest.approx(expected, abs=1e-5), (
            f"Expected sqrt((pi/4)^2 + (pi/3)^2) = {expected:.6f}, got {dist.item():.6f}"
        )

    def test_small_angle(self) -> None:
        """Very small rotation (0.01 radians): verifies precision for near-zero angles."""
        theta = 0.01

        U1 = torch.zeros(4, 2, dtype=torch.float64)
        U1[0, 0] = 1.0
        U1[1, 1] = 1.0

        U2 = torch.zeros(4, 2, dtype=torch.float64)
        U2[0, 0] = math.cos(theta)
        U2[2, 0] = math.sin(theta)
        U2[1, 1] = 1.0

        dist = grassmannian_distance(U1, U2, k=2)
        expected = theta
        assert dist.item() == pytest.approx(expected, abs=1e-4)


class TestGrassmannianVaryingK:
    """Verify formula generalizes correctly for k=1, k=2, k=3."""

    def test_k1_single_principal_angle(self) -> None:
        """k=1: distance = |theta| for a single principal angle theta = pi/5."""
        theta = math.pi / 5

        # 6D space, rotate e1 by theta in (e1, e4) plane
        U1 = torch.zeros(6, 1, dtype=torch.float64)
        U1[0, 0] = 1.0

        U2 = torch.zeros(6, 1, dtype=torch.float64)
        U2[0, 0] = math.cos(theta)
        U2[3, 0] = math.sin(theta)

        dist = grassmannian_distance(U1, U2, k=1)
        expected = abs(theta)
        assert dist.item() == pytest.approx(expected, abs=1e-5)

    def test_k2_two_principal_angles(self) -> None:
        """k=2: d = sqrt(theta_1^2 + theta_2^2)."""
        theta1 = math.pi / 6
        theta2 = math.pi / 4

        U1 = torch.zeros(6, 2, dtype=torch.float64)
        U1[0, 0] = 1.0
        U1[1, 1] = 1.0

        U2 = torch.zeros(6, 2, dtype=torch.float64)
        U2[0, 0] = math.cos(theta1)
        U2[3, 0] = math.sin(theta1)
        U2[1, 1] = math.cos(theta2)
        U2[4, 1] = math.sin(theta2)

        dist = grassmannian_distance(U1, U2, k=2)
        expected = math.sqrt(theta1**2 + theta2**2)
        assert dist.item() == pytest.approx(expected, abs=1e-5)

    def test_k3_three_principal_angles(self) -> None:
        """k=3: d = sqrt(theta_1^2 + theta_2^2 + theta_3^2)."""
        theta1 = math.pi / 6
        theta2 = math.pi / 4
        theta3 = math.pi / 3

        U1 = torch.zeros(6, 3, dtype=torch.float64)
        U1[0, 0] = 1.0
        U1[1, 1] = 1.0
        U1[2, 2] = 1.0

        U2 = torch.zeros(6, 3, dtype=torch.float64)
        U2[0, 0] = math.cos(theta1)
        U2[3, 0] = math.sin(theta1)
        U2[1, 1] = math.cos(theta2)
        U2[4, 1] = math.sin(theta2)
        U2[2, 2] = math.cos(theta3)
        U2[5, 2] = math.sin(theta3)

        dist = grassmannian_distance(U1, U2, k=3)
        expected = math.sqrt(theta1**2 + theta2**2 + theta3**2)
        assert dist.item() == pytest.approx(expected, abs=1e-5)


class TestGrassmannianClipping:
    """Verify numerical clipping to [-1, 1] handles edge cases without hiding bugs."""

    def test_near_identical_perturbation_not_nan(self) -> None:
        """Tiny perturbation of orthonormal U: distance should be small and non-NaN.
        Numerical noise could push cos_angles slightly above 1.0 without clipping.
        """
        torch.manual_seed(42)
        U1 = torch.eye(4, 2, dtype=torch.float64)

        # Add tiny perturbation and re-orthonormalize
        perturbation = torch.randn(4, 2, dtype=torch.float64) * 1e-7
        U2_noisy = U1 + perturbation
        U2, _ = torch.linalg.qr(U2_noisy)
        U2 = U2[:, :2]

        dist = grassmannian_distance(U1, U2, k=2)

        # Must not be NaN
        assert torch.isfinite(dist).item(), "Distance should be finite, not NaN/Inf"
        # Must be non-negative
        assert dist.item() >= 0.0, "Distance should be non-negative"
        # Must be very small (subspaces nearly identical)
        assert dist.item() < 1e-4, (
            f"Distance should be very small for near-identical subspaces, got {dist.item()}"
        )

    def test_exactly_identical_float32(self) -> None:
        """Float32 precision: identical subspaces should still give 0 distance."""
        U = torch.eye(4, 2, dtype=torch.float32)
        dist = grassmannian_distance(U, U, k=2)
        assert dist.item() == pytest.approx(0.0, abs=1e-5)


class TestGrassmannianBatched:
    """Test batched input: multiple subspace pairs in parallel."""

    def test_batched_output_shape(self) -> None:
        """U_prev shape [2, 4, 3], U_curr shape [2, 4, 3], k=2 -> output shape [2]."""
        torch.manual_seed(42)
        # Create batched orthonormal matrices via QR
        A = torch.randn(2, 4, 3, dtype=torch.float64)
        # QR doesn't support batched in all PyTorch versions, do per-element
        U_prev = torch.stack([torch.linalg.qr(A[i])[0] for i in range(2)])
        B = torch.randn(2, 4, 3, dtype=torch.float64)
        U_curr = torch.stack([torch.linalg.qr(B[i])[0] for i in range(2)])

        dist = grassmannian_distance(U_prev, U_curr, k=2)
        assert dist.shape == (2,), f"Expected shape (2,), got {dist.shape}"

    def test_batched_values_match_unbatched(self) -> None:
        """Each element of batched result should match individual computation."""
        torch.manual_seed(42)
        A = torch.randn(2, 4, 3, dtype=torch.float64)
        U_prev = torch.stack([torch.linalg.qr(A[i])[0] for i in range(2)])
        B = torch.randn(2, 4, 3, dtype=torch.float64)
        U_curr = torch.stack([torch.linalg.qr(B[i])[0] for i in range(2)])

        # Batched
        dist_batch = grassmannian_distance(U_prev, U_curr, k=2)

        # Unbatched (element-wise)
        for i in range(2):
            dist_single = grassmannian_distance(U_prev[i], U_curr[i], k=2)
            assert dist_batch[i].item() == pytest.approx(dist_single.item(), abs=1e-10), (
                f"Batch element {i}: batched={dist_batch[i].item()}, "
                f"unbatched={dist_single.item()}"
            )

    def test_batched_with_known_values(self) -> None:
        """Batch of 2: first pair identical (d=0), second pair orthogonal (d=pi/2*sqrt(2))."""
        U_prev = torch.zeros(2, 4, 2, dtype=torch.float64)
        U_curr = torch.zeros(2, 4, 2, dtype=torch.float64)

        # First pair: identical subspaces (e1, e2)
        U_prev[0, 0, 0] = 1.0
        U_prev[0, 1, 1] = 1.0
        U_curr[0, 0, 0] = 1.0
        U_curr[0, 1, 1] = 1.0

        # Second pair: orthogonal subspaces
        U_prev[1, 0, 0] = 1.0
        U_prev[1, 1, 1] = 1.0
        U_curr[1, 2, 0] = 1.0
        U_curr[1, 3, 1] = 1.0

        dist = grassmannian_distance(U_prev, U_curr, k=2)
        assert dist[0].item() == pytest.approx(0.0, abs=1e-6)
        assert dist[1].item() == pytest.approx(
            (math.pi / 2) * math.sqrt(2), abs=1e-5
        )
