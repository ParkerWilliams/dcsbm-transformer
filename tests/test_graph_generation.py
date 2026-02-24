"""Tests for DCSBM graph generation, validation, and degree correction."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse
from scipy.sparse.csgraph import connected_components

from src.config.defaults import ANCHOR_CONFIG
from src.config.experiment import ExperimentConfig, GraphConfig, TrainingConfig
from src.graph.dcsbm import (
    GraphGenerationError,
    build_probability_matrix,
    generate_dcsbm_graph,
    sample_adjacency,
    validate_graph,
)
from src.graph.degree_correction import sample_theta
from src.graph.types import GraphData


class TestDegreeCorrection:
    """Tests for theta sampling and normalization."""

    def test_theta_shape(self) -> None:
        rng = np.random.default_rng(42)
        theta = sample_theta(500, 4, 1.0, rng)
        assert theta.shape == (500,)

    def test_theta_per_block_normalization(self) -> None:
        rng = np.random.default_rng(42)
        theta = sample_theta(500, 4, 1.0, rng)
        for b in range(4):
            block_sum = theta[b * 125 : (b + 1) * 125].sum()
            assert abs(block_sum - 125) < 1e-10, (
                f"Block {b} sum: {block_sum}"
            )

    def test_theta_heterogeneity(self) -> None:
        """Degree correction should produce heterogeneous distribution (CV > 0.3)."""
        rng = np.random.default_rng(42)
        theta = sample_theta(500, 4, 1.0, rng)
        cv = theta.std() / theta.mean()
        assert cv > 0.3, f"CV too low: {cv}"

    def test_theta_all_positive(self) -> None:
        rng = np.random.default_rng(42)
        theta = sample_theta(500, 4, 1.0, rng)
        assert (theta > 0).all(), "All theta values must be positive"


class TestProbabilityMatrix:
    """Tests for the DCSBM probability matrix construction."""

    def test_probability_matrix_shape(self) -> None:
        rng = np.random.default_rng(42)
        theta = sample_theta(500, 4, 1.0, rng)
        P = build_probability_matrix(500, 4, 0.25, 0.03, theta)
        assert P.shape == (500, 500)

    def test_probability_matrix_no_self_loops(self) -> None:
        rng = np.random.default_rng(42)
        theta = sample_theta(500, 4, 1.0, rng)
        P = build_probability_matrix(500, 4, 0.25, 0.03, theta)
        assert np.all(np.diag(P) == 0.0)

    def test_probability_matrix_values_in_range(self) -> None:
        rng = np.random.default_rng(42)
        theta = sample_theta(500, 4, 1.0, rng)
        P = build_probability_matrix(500, 4, 0.25, 0.03, theta)
        assert P.min() >= 0.0
        assert P.max() <= 1.0

    def test_minimum_expected_degree(self) -> None:
        """Min expected degree should be >= 3 for anchor config."""
        rng = np.random.default_rng(42)
        theta = sample_theta(500, 4, 1.0, rng)
        P = build_probability_matrix(500, 4, 0.25, 0.03, theta)
        min_expected = P.sum(axis=1).min()
        assert min_expected >= 3.0, f"Min expected degree: {min_expected:.2f}"


class TestGraphGeneration:
    """Tests for end-to-end DCSBM graph generation."""

    def test_anchor_config_generates_valid_graph(self) -> None:
        graph = generate_dcsbm_graph(ANCHOR_CONFIG)
        assert isinstance(graph, GraphData)
        assert graph.n == 500
        assert graph.K == 4
        assert graph.block_size == 125
        assert graph.adjacency.shape == (500, 500)

    def test_graph_is_strongly_connected(self) -> None:
        graph = generate_dcsbm_graph(ANCHOR_CONFIG)
        n_components, _ = connected_components(
            graph.adjacency, directed=True, connection="strong"
        )
        assert n_components == 1

    def test_no_self_loops(self) -> None:
        graph = generate_dcsbm_graph(ANCHOR_CONFIG)
        assert graph.adjacency.diagonal().sum() == 0

    def test_degree_correction_heterogeneity(self) -> None:
        """Realized degree distribution should be heterogeneous."""
        graph = generate_dcsbm_graph(ANCHOR_CONFIG)
        # Out-degree from sparse matrix
        out_degrees = np.array(graph.adjacency.sum(axis=1)).ravel()
        cv = out_degrees.std() / out_degrees.mean()
        assert cv > 0.3, f"Realized degree CV too low: {cv}"

    def test_edge_density_within_tolerance(self) -> None:
        graph = generate_dcsbm_graph(ANCHOR_CONFIG)
        n, K = graph.n, graph.K
        block_size = graph.block_size
        blocks = graph.block_assignments

        rng = np.random.default_rng(graph.generation_seed)
        theta = sample_theta(n, K, 1.0, rng)
        P = build_probability_matrix(n, K, 0.25, 0.03, theta)

        for a in range(K):
            mask_a = blocks == a
            for b in range(K):
                mask_b = blocks == b
                sub = graph.adjacency[np.ix_(mask_a, mask_b)].toarray()
                P_sub = P[np.ix_(mask_a, mask_b)]

                if a == b:
                    n_pairs = block_size * (block_size - 1)
                    np.fill_diagonal(sub, 0)
                    P_sub_copy = P_sub.copy()
                    np.fill_diagonal(P_sub_copy, 0)
                    observed = sub.sum() / n_pairs
                    expected = P_sub_copy.sum() / n_pairs
                else:
                    n_pairs = block_size * block_size
                    observed = sub.sum() / n_pairs
                    expected = P_sub.sum() / n_pairs

                sigma = np.sqrt(expected * (1 - expected) / n_pairs)
                assert abs(observed - expected) <= 3 * sigma, (
                    f"Block ({a},{b}): observed={observed:.4f}, "
                    f"expected={expected:.4f}, 3*sigma={3*sigma:.4f}"
                )

    def test_validation_rejects_disconnected_graph(self) -> None:
        """Manually create a disconnected graph and verify validation catches it."""
        # Two isolated components: vertices 0-1 connected, vertices 2-3 connected
        adj = scipy.sparse.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=np.float64,
            )
        )
        P = np.full((4, 4), 0.5)
        np.fill_diagonal(P, 0.0)
        errors = validate_graph(adj, P, 4, 2, 0.5, 0.5)
        assert any("Not strongly connected" in e for e in errors)

    def test_equal_block_size_required(self) -> None:
        """n must be evenly divisible by K."""
        bad_config = ExperimentConfig(
            graph=GraphConfig(n=501, K=4),
            training=TrainingConfig(corpus_size=100 * 501),
        )
        with pytest.raises(ValueError, match="evenly divisible"):
            generate_dcsbm_graph(bad_config)

    def test_reproducibility_same_seed(self) -> None:
        """Same config and seed produce identical graphs."""
        g1 = generate_dcsbm_graph(ANCHOR_CONFIG)
        g2 = generate_dcsbm_graph(ANCHOR_CONFIG)
        # Adjacency matrices should be identical
        diff = g1.adjacency - g2.adjacency
        assert diff.nnz == 0, "Graphs differ despite same seed"

    def test_retry_on_failure(self) -> None:
        """Verify retry logic is invoked when validation fails."""
        call_count = 0
        original_validate = validate_graph

        def mock_validate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return ["Simulated failure"]
            return original_validate(*args, **kwargs)

        with patch("src.graph.dcsbm.validate_graph", side_effect=mock_validate):
            graph = generate_dcsbm_graph(ANCHOR_CONFIG)

        assert call_count >= 3, "Should have retried at least twice"
        assert graph.attempt >= 2, "Should have used attempt >= 2"
