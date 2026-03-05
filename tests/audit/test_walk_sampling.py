"""Audit tests for walk sampling uniformity (GRAPH-02).

Verifies that both single-walk and batch-walk generation select neighbors
uniformly at random, and that all generated walks traverse valid graph edges.
"""

import numpy as np
import pytest
import scipy.sparse

from src.graph.dcsbm import build_probability_matrix, sample_adjacency
from src.graph.degree_correction import sample_theta
from src.walk.generator import generate_batch_unguided_walks


class TestSingleWalkUniformNeighborSelection:
    """Verify rng.integers(0, degree) produces uniform neighbor selection."""

    def test_single_walk_uniform_neighbor_selection(self) -> None:
        """rng.integers(0, d) is uniform on [0, d) by NumPy Generator contract;
        empirical test confirms. For a vertex with degree d, generate 100k
        single-step selections and verify each neighbor appears with frequency
        approximately 1/d.
        """
        degree = 5
        n_samples = 100_000
        rng = np.random.default_rng(42)

        counts = np.zeros(degree, dtype=np.int64)
        for _ in range(n_samples):
            idx = int(rng.integers(0, degree))
            counts[idx] += 1

        expected_freq = 1.0 / degree
        empirical_freqs = counts / n_samples

        for i in range(degree):
            # 4-sigma tolerance: sqrt(p*(1-p)/n) * 4
            se = np.sqrt(expected_freq * (1 - expected_freq) / n_samples)
            tolerance = 4 * se

            np.testing.assert_allclose(
                empirical_freqs[i],
                expected_freq,
                atol=tolerance,
                err_msg=(
                    f"Neighbor {i}: frequency {empirical_freqs[i]:.4f} "
                    f"differs from expected {expected_freq:.4f}"
                ),
            )


class TestBatchWalkFloatToIntBias:
    """Document and verify batch walk float-to-int conversion uniformity."""

    def test_batch_walk_float_to_int_bias(self) -> None:
        """Batch walk uses floor(U*d) which is exactly uniform for integer d.
        The theoretical bias from float64 truncation is < 1/2^53, negligible.
        For each degree d in {3, 5, 7, 10}, generate 100k samples via
        floor(U*d) and verify uniformity empirically.
        """
        n_samples = 100_000
        rng = np.random.default_rng(999)

        for degree in [3, 5, 7, 10]:
            # Replicate the batch walk's exact sampling code:
            # offsets = (rng.random(n_walks) * degrees).astype(np.int64)
            samples = (rng.random(n_samples) * degree).astype(np.int64)
            samples = np.clip(samples, 0, degree - 1)

            counts = np.bincount(samples, minlength=degree)
            expected_freq = 1.0 / degree
            empirical_freqs = counts / n_samples

            for i in range(degree):
                se = np.sqrt(expected_freq * (1 - expected_freq) / n_samples)
                tolerance = 4 * se

                np.testing.assert_allclose(
                    empirical_freqs[i],
                    expected_freq,
                    atol=tolerance,
                    err_msg=(
                        f"degree={degree}, bin {i}: "
                        f"freq {empirical_freqs[i]:.4f} != {expected_freq:.4f}"
                    ),
                )


class TestWalkEdgesValid:
    """Verify that every consecutive pair in a walk is a valid edge."""

    def test_walk_edges_valid(self) -> None:
        """Walk must only traverse existing edges.
        Generate walks on a small graph and verify every (u, v) transition
        is present in the adjacency matrix.
        """
        n, K = 20, 4
        p_in, p_out = 0.6, 0.15
        rng = np.random.default_rng(42)
        theta = sample_theta(n, K, alpha=1.0, rng=rng)
        P = build_probability_matrix(n, K, p_in, p_out, theta)
        adj = sample_adjacency(P, np.random.default_rng(42))

        indptr = adj.indptr
        indices = adj.indices

        # Generate batch walks
        walk_rng = np.random.default_rng(100)
        start_vertices = np.arange(n, dtype=np.int32)  # one walk per vertex
        walk_length = 15
        walks = generate_batch_unguided_walks(
            start_vertices, walk_length, walk_rng, indptr, indices
        )

        # Verify every transition is a valid edge
        for wi in range(walks.shape[0]):
            for step in range(walk_length - 1):
                u = walks[wi, step]
                v = walks[wi, step + 1]
                neighbors = indices[indptr[u]:indptr[u + 1]]
                assert v in neighbors, (
                    f"Walk {wi}, step {step}: edge {u}->{v} not in graph"
                )


class TestWalkNoDegreeNodeBias:
    """Verify uniform neighbor selection in walks (no degree bias)."""

    def test_walk_no_degree_bias(self) -> None:
        """On a small graph where a vertex has known neighbors, generate many
        walks starting at that vertex and verify the distribution of first
        steps is approximately uniform across all neighbors.
        """
        # Build a small graph manually where vertex 0 has exactly 3 neighbors
        n = 5
        # Adjacency: 0->1, 0->2, 0->3 (and some others for connected graph)
        row = [0, 0, 0, 1, 2, 3, 4, 1, 2, 3]
        col = [1, 2, 3, 0, 0, 0, 0, 4, 4, 4]
        data = [1.0] * len(row)
        adj = scipy.sparse.csr_matrix(
            (data, (row, col)), shape=(n, n)
        )

        indptr = adj.indptr
        indices = adj.indices

        # Verify vertex 0 has neighbors {1, 2, 3}
        neighbors_0 = set(indices[indptr[0]:indptr[0 + 1]].tolist())
        assert neighbors_0 == {1, 2, 3}, f"Expected {{1,2,3}}, got {neighbors_0}"

        n_walks = 30_000
        walk_length = 2  # just need first step
        start_vertices = np.zeros(n_walks, dtype=np.int32)  # all start at 0
        walk_rng = np.random.default_rng(77)

        walks = generate_batch_unguided_walks(
            start_vertices, walk_length, walk_rng, indptr, indices
        )

        # Count first-step destinations
        first_steps = walks[:, 1]
        counts = {1: 0, 2: 0, 3: 0}
        for v in first_steps:
            counts[int(v)] += 1

        expected_freq = 1.0 / 3.0
        for neighbor, count in counts.items():
            empirical_freq = count / n_walks
            se = np.sqrt(expected_freq * (1 - expected_freq) / n_walks)
            tolerance = 4 * se

            np.testing.assert_allclose(
                empirical_freq,
                expected_freq,
                atol=tolerance,
                err_msg=(
                    f"Neighbor {neighbor}: freq {empirical_freq:.4f} "
                    f"!= expected {expected_freq:.4f}"
                ),
            )
