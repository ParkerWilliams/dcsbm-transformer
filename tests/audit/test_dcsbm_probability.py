"""Audit tests for DCSBM edge probability matrix and degree correction (GRAPH-01).

Verifies the mathematical correctness of build_probability_matrix(),
sample_adjacency(), and degree correction normalization against the
Degree-Corrected Stochastic Block Model definition (Karrer & Newman 2011).
"""

import numpy as np
import pytest
import scipy.sparse

from src.graph.dcsbm import build_probability_matrix, sample_adjacency
from src.graph.degree_correction import sample_theta


class TestDCSBMProbabilitySymmetry:
    """Verify that the DCSBM probability matrix is symmetric."""

    def test_dcsbm_probability_symmetry(self) -> None:
        """P must be symmetric because P_ij = theta_i * theta_j * B_{z_i, z_j}
        and B is a symmetric K x K matrix (omega[a,b] == omega[b,a]).
        With shared theta for in/out degree, the outer product theta_i * theta_j
        is also symmetric, so P = diag(theta) @ B_expanded @ diag(theta) is symmetric.
        """
        # DCSBM with shared theta produces symmetric P because
        # P_ij = theta_i * theta_j * B_{z_i,z_j} and B is symmetric.
        n, K = 10, 2
        p_in, p_out = 0.5, 0.1
        rng = np.random.default_rng(42)
        theta = sample_theta(n, K, alpha=1.0, rng=rng)

        P = build_probability_matrix(n, K, p_in, p_out, theta)

        np.testing.assert_array_equal(
            P, P.T, err_msg="Probability matrix must be symmetric"
        )


class TestDCSBMProbabilityFormula:
    """Verify P_ij = theta_i * theta_j * omega_{z_i, z_j} (Karrer & Newman 2011 Eq. 1)."""

    def test_dcsbm_probability_formula(self) -> None:
        """Direct verification of Karrer & Newman (2011) Eq. 1.
        For a small graph, manually compute expected P values using the formula
        P_ij = theta_i * theta_j * omega_{z_i, z_j} and compare element-wise
        against build_probability_matrix output (before clipping, i.e. using
        moderate p_in/p_out so no clipping occurs).
        """
        n, K = 6, 2
        p_in, p_out = 0.3, 0.05
        rng = np.random.default_rng(123)
        theta = sample_theta(n, K, alpha=1.0, rng=rng)

        P = build_probability_matrix(n, K, p_in, p_out, theta)

        # Manually compute expected values
        block_size = n // K
        blocks = np.arange(n) // block_size

        omega = np.full((K, K), p_out, dtype=np.float64)
        np.fill_diagonal(omega, p_in)

        block_probs = omega[blocks][:, blocks]
        P_expected = np.outer(theta, theta) * block_probs
        np.clip(P_expected, 0.0, 1.0, out=P_expected)
        np.fill_diagonal(P_expected, 0.0)

        np.testing.assert_allclose(
            P,
            P_expected,
            atol=1e-15,
            err_msg="P must match outer(theta, theta) * block_probs",
        )


class TestDCSBMNoSelfLoops:
    """Verify P[i,i] == 0 for all i."""

    def test_dcsbm_no_self_loops(self) -> None:
        """Self-loop probability must be zero by construction.
        The DCSBM model does not allow self-loops; build_probability_matrix
        zeroes the diagonal after computing the outer product.
        """
        n, K = 12, 3
        p_in, p_out = 0.4, 0.1
        rng = np.random.default_rng(99)
        theta = sample_theta(n, K, alpha=1.0, rng=rng)

        P = build_probability_matrix(n, K, p_in, p_out, theta)

        np.testing.assert_array_equal(
            np.diag(P),
            np.zeros(n),
            err_msg="Diagonal of P must be all zeros (no self-loops)",
        )


class TestDCSBMDegreeCorrection:
    """Verify per-block theta normalization preserves expected degree."""

    def test_dcsbm_degree_correction_preserves_expected_degree(self) -> None:
        """Per-block normalization preserves E[degree] from uncorrected SBM
        (Karrer & Newman 2011, Section II.A).
        For each block b, sum(theta[start:end]) == block_size, which means
        the total expected degree contribution from the block is the same as
        in an uncorrected SBM where all theta_i = 1.
        """
        for n, K in [(10, 2), (20, 4), (30, 5), (6, 3)]:
            rng = np.random.default_rng(77)
            theta = sample_theta(n, K, alpha=1.0, rng=rng)
            block_size = n // K

            for b in range(K):
                start = b * block_size
                end = start + block_size
                block_sum = theta[start:end].sum()
                np.testing.assert_allclose(
                    block_sum,
                    block_size,
                    atol=1e-10,
                    err_msg=(
                        f"Block {b} theta sum {block_sum} != {block_size} "
                        f"for n={n}, K={K}"
                    ),
                )


class TestDCSBMBlockStructure:
    """Verify that block assignments drive the affinity matrix omega."""

    def test_dcsbm_block_structure(self) -> None:
        """Block assignment drives affinity matrix omega.
        For vertices i,j in the same block: P[i,j] = theta_i * theta_j * p_in.
        For vertices in different blocks: P[i,j] = theta_i * theta_j * p_out.
        (Off-diagonal entries only, since diagonal is forced to zero.)
        """
        n, K = 8, 2
        p_in, p_out = 0.6, 0.1
        rng = np.random.default_rng(55)
        theta = sample_theta(n, K, alpha=1.0, rng=rng)

        P = build_probability_matrix(n, K, p_in, p_out, theta)

        block_size = n // K
        blocks = np.arange(n) // block_size

        for i in range(n):
            for j in range(n):
                if i == j:
                    assert P[i, j] == 0.0, "Diagonal must be zero"
                    continue

                expected_base = theta[i] * theta[j]
                if blocks[i] == blocks[j]:
                    expected = min(1.0, expected_base * p_in)
                else:
                    expected = min(1.0, expected_base * p_out)

                np.testing.assert_allclose(
                    P[i, j],
                    expected,
                    atol=1e-15,
                    err_msg=(
                        f"P[{i},{j}] mismatch: blocks=({blocks[i]},{blocks[j]})"
                    ),
                )


class TestDCSBMProbabilityClipping:
    """Verify probabilities are clipped to [0, 1]."""

    def test_dcsbm_probability_clipping(self) -> None:
        """Degree correction can push theta_i * theta_j * omega above 1.0.
        The code must clip to [0, 1]. We construct a scenario with very high
        p_in and concentrated theta to trigger clipping.
        """
        n, K = 4, 2
        # Very high p_in to trigger clipping with concentrated theta
        p_in, p_out = 0.99, 0.01

        # Manually create theta where some products will exceed 1.0
        # block_size = 2, so theta sums to 2 per block
        # If theta = [1.8, 0.2, 1.8, 0.2], then theta[0]*theta[2]*p_in is not
        # the issue because 0 and 2 are in different blocks.
        # Same block: theta[0]*theta[1]*p_in = 1.8*0.2*0.99 = 0.356 (not enough)
        # So use a larger block and extreme theta:
        n, K = 6, 2
        block_size = 3
        # theta sums to 3 per block
        # If theta = [2.5, 0.3, 0.2, 2.5, 0.3, 0.2]
        # theta[0]*theta[3]*p_out = 2.5*2.5*0.01 = 0.0625 (not enough)
        # For same block: theta[0]*theta[1]*p_in = 2.5*0.3*0.99 = 0.7425 (still < 1)
        # Need even more extreme:
        p_in = 1.0
        theta = np.array([2.7, 0.2, 0.1, 2.7, 0.2, 0.1])
        # theta[0]*theta[1]*1.0 = 2.7*0.2 = 0.54 (still < 1)
        # We need theta_i * theta_j > 1.0 in same block.
        # With block_size=3, sum=3: e.g. [2.9, 0.05, 0.05]
        # theta[0]*theta[0] would be self-loop (zeroed), but if same block:
        # Actually the problem is theta_i and theta_j are in same block of size 3
        # so we need two large values: [2.0, 1.0, 0.0] -- sum is 3.0
        # But 0.0 is problematic. Let's just do it differently:
        # Use n=4, K=1 so block_size=4, all vertices in same block
        n, K = 4, 1
        p_in = 0.5
        p_out = 0.5
        # theta sums to 4: [3.0, 0.5, 0.3, 0.2]
        theta = np.array([3.0, 0.5, 0.3, 0.2])
        assert abs(theta.sum() - n) < 1e-10
        # theta[0]*theta[0]*0.5 = 4.5 (self-loop, zeroed)
        # theta[0]*theta[1]*0.5 = 0.75
        # We need theta_i * theta_j * p_in > 1.0
        # With p_in=1.0: theta[0]*theta[1] = 3.0*0.5 = 1.5 > 1.0!
        p_in = 1.0
        p_out = 1.0  # only 1 block so p_out doesn't matter

        P = build_probability_matrix(n, K, p_in, p_out, theta)

        # Verify all values in [0, 1]
        assert P.min() >= 0.0, f"P has values below 0: {P.min()}"
        assert P.max() <= 1.0, f"P has values above 1: {P.max()}"

        # Verify clipping actually happened: without clipping, theta[0]*theta[1]*1.0 = 1.5
        assert theta[0] * theta[1] * p_in > 1.0, "Precondition: unclipped value > 1"
        assert P[0, 1] == 1.0, "Clipped value should be exactly 1.0"


class TestAdjacencyBernoulliSampling:
    """Verify adjacency sampling follows Bernoulli(P_ij) distribution."""

    def test_adjacency_bernoulli_sampling(self) -> None:
        """Each edge is an independent Bernoulli(P_ij) draw.
        Sample many adjacency matrices from the same P and verify that
        empirical edge frequencies converge to P[i,j] within statistical
        tolerance.
        """
        n, K = 6, 2
        p_in, p_out = 0.4, 0.1
        rng_theta = np.random.default_rng(42)
        theta = sample_theta(n, K, alpha=1.0, rng=rng_theta)
        P = build_probability_matrix(n, K, p_in, p_out, theta)

        n_samples = 2000
        edge_counts = np.zeros((n, n), dtype=np.float64)

        for trial in range(n_samples):
            rng = np.random.default_rng(trial * 1000 + 7)
            adj = sample_adjacency(P, rng)
            edge_counts += adj.toarray()

        empirical_P = edge_counts / n_samples

        # For each off-diagonal entry, check that empirical frequency is
        # close to the true probability. Use a generous tolerance since
        # we're averaging over finite samples.
        for i in range(n):
            for j in range(n):
                if i == j:
                    assert empirical_P[i, j] == 0.0, "No self-loops in samples"
                    continue

                # Standard error of Bernoulli mean: sqrt(p*(1-p)/n_samples)
                p = P[i, j]
                se = np.sqrt(p * (1 - p) / n_samples) if 0 < p < 1 else 0.01
                # Use 4-sigma tolerance for high confidence
                tolerance = max(4 * se, 0.02)  # floor at 0.02

                np.testing.assert_allclose(
                    empirical_P[i, j],
                    p,
                    atol=tolerance,
                    err_msg=(
                        f"Empirical P[{i},{j}]={empirical_P[i,j]:.4f} "
                        f"differs from true P[{i},{j}]={p:.4f} "
                        f"(tolerance={tolerance:.4f})"
                    ),
                )


class TestBlockAssignmentCorrectness:
    """Verify block assignment formula np.arange(n) // block_size."""

    def test_block_assignment_correctness(self) -> None:
        """np.arange(n) // block_size correctly assigns vertices to blocks
        for various n, K values. Each block should get exactly block_size
        vertices numbered consecutively.
        """
        test_cases = [
            (10, 2),  # standard
            (12, 3),  # 3 blocks
            (20, 4),  # 4 blocks
            (4, 4),   # edge case: block_size=1
            (6, 1),   # edge case: single block
        ]

        for n, K in test_cases:
            block_size = n // K
            blocks = np.arange(n) // block_size

            # Verify each vertex is assigned to the correct block
            for v in range(n):
                expected_block = v // block_size
                assert blocks[v] == expected_block, (
                    f"Vertex {v} assigned to block {blocks[v]}, "
                    f"expected {expected_block} (n={n}, K={K})"
                )

            # Verify each block has exactly block_size vertices
            for b in range(K):
                count = np.sum(blocks == b)
                assert count == block_size, (
                    f"Block {b} has {count} vertices, "
                    f"expected {block_size} (n={n}, K={K})"
                )
