"""Tests for walk generation correctness.

Covers edge validity, guided walk compliance, jumper event recording,
batch walk shapes, nested jumper compliance, path-count normalization,
reproducibility, and infeasible walk handling.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.config.experiment import (
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
)
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.walk.compliance import precompute_path_counts
from src.walk.generator import (
    generate_batch_unguided_walks,
    generate_single_guided_walk,
    generate_walks,
)


def _make_small_graph(n: int = 10, K: int = 2, seed: int = 42) -> GraphData:
    """Create a small directed graph for testing.

    Creates a graph with guaranteed edges: each vertex connects to all
    vertices in its block plus some cross-block edges.
    """
    rng = np.random.default_rng(seed)
    block_size = n // K
    block_assignments = np.array(
        [i // block_size for i in range(n)], dtype=np.int32
    )

    # Dense adjacency: in-block edges with high probability, cross-block lower
    dense = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            bi, bj = block_assignments[i], block_assignments[j]
            if bi == bj:
                if rng.random() < 0.7:
                    dense[i, j] = 1.0
            else:
                if rng.random() < 0.2:
                    dense[i, j] = 1.0

    # Ensure every vertex has at least one outgoing edge
    for i in range(n):
        if dense[i].sum() == 0:
            j = (i + 1) % n
            dense[i, j] = 1.0

    adj = csr_matrix(dense)

    return GraphData(
        adjacency=adj,
        block_assignments=block_assignments,
        theta=np.ones(n),
        n=n,
        K=K,
        block_size=block_size,
        generation_seed=seed,
        attempt=0,
    )


def _make_anchor_config() -> ExperimentConfig:
    """Create anchor config (n=500, K=4, w=64) for integration tests."""
    return ExperimentConfig(
        graph=GraphConfig(n=500, K=4, p_in=0.25, p_out=0.03),
        model=ModelConfig(),
        training=TrainingConfig(
            w=64,
            walk_length=256,
            corpus_size=200_000,
        ),
        seed=42,
    )


def _make_small_config(n: int = 10) -> ExperimentConfig:
    """Create a small config for unit tests."""
    return ExperimentConfig(
        graph=GraphConfig(n=n, K=2, p_in=0.7, p_out=0.2, n_jumpers_per_block=1),
        model=ModelConfig(),
        training=TrainingConfig(
            w=4,
            walk_length=10,
            corpus_size=100 * n,
            r=4,
        ),
        seed=42,
    )


class TestWalksFollowValidEdges:
    """Test that generated walks only follow valid directed edges."""

    def test_walks_follow_valid_edges(self) -> None:
        graph = _make_small_graph(n=10, K=2)
        config = _make_small_config(n=10)
        indptr = graph.adjacency.indptr
        indices = graph.adjacency.indices

        # Create simple jumpers
        jumpers = [
            JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3),
        ]

        result = generate_walks(
            graph, jumpers, config, seed=42,
            target_n_walks=20, min_jumper_fraction=0.3,
        )

        for wi in range(result.walks.shape[0]):
            for step in range(result.walks.shape[1] - 1):
                u = result.walks[wi, step]
                v = result.walks[wi, step + 1]
                neighbors = indices[indptr[u]:indptr[u + 1]]
                assert v in neighbors, (
                    f"Walk {wi} step {step}: edge {u}->{v} not valid"
                )


class TestGuidedWalkCompliance:
    """Test that guided walks satisfy jumper rules."""

    def test_guided_walk_compliance(self) -> None:
        graph = _make_small_graph(n=20, K=2, seed=123)
        jumper = JumperInfo(
            vertex_id=0, source_block=0, target_block=1, r=3,
        )
        jumper_map = {0: jumper}
        path_counts = precompute_path_counts(
            graph.adjacency, graph.block_assignments, graph.K, 3,
        )
        indptr = graph.adjacency.indptr
        indices = graph.adjacency.indices

        compliant_count = 0
        total = 0

        for seed in range(50):
            rng = np.random.default_rng(seed)
            result = generate_single_guided_walk(
                0, 10, rng, graph, jumper_map, path_counts, indptr, indices,
            )
            if result is None:
                continue
            walk, events = result
            total += 1
            for event in events:
                if event.expected_arrival_step < len(walk):
                    actual = graph.block_assignments[
                        walk[event.expected_arrival_step]
                    ]
                    if actual == event.target_block:
                        compliant_count += 1

        assert total > 0, "No walks generated"
        assert compliant_count > 0, "No compliant walks"


class TestJumperEventsRecorded:
    """Test that jumper events are properly recorded."""

    def test_jumper_events_recorded(self) -> None:
        graph = _make_small_graph(n=10, K=2)
        config = _make_small_config(n=10)
        jumpers = [
            JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3),
        ]

        result = generate_walks(
            graph, jumpers, config, seed=42,
            target_n_walks=20, min_jumper_fraction=0.3,
        )

        # At least some walks should have events
        has_events = [e for e in result.events if len(e) > 0]
        assert len(has_events) > 0, "No walks have jumper events"

        # Check event fields
        for walk_events in has_events:
            for event in walk_events:
                assert isinstance(event.vertex_id, int)
                assert isinstance(event.step, int)
                assert isinstance(event.target_block, int)
                assert isinstance(event.expected_arrival_step, int)
                assert event.vertex_id == 0  # our only jumper
                assert event.target_block == 1


class TestBatchUnguidedWalksShape:
    """Test that batch unguided walks have correct shape and dtype."""

    def test_batch_unguided_walks_shape(self) -> None:
        graph = _make_small_graph(n=10, K=2)
        indptr = graph.adjacency.indptr
        indices = graph.adjacency.indices
        rng = np.random.default_rng(42)

        n_walks = 15
        walk_length = 8
        start_vertices = rng.integers(0, 10, size=n_walks, dtype=np.int32)

        walks = generate_batch_unguided_walks(
            start_vertices, walk_length, rng, indptr, indices,
        )

        assert walks.shape == (n_walks, walk_length)
        assert walks.dtype == np.int32
        # Verify start vertices
        np.testing.assert_array_equal(walks[:, 0], start_vertices)


class TestNestedJumperCompliance:
    """Test compliance when two jumpers create overlapping constraints."""

    def test_nested_jumper_compliance(self) -> None:
        # Create graph with 2 blocks where cross-block paths exist
        graph = _make_small_graph(n=20, K=2, seed=999)

        # Two jumpers: vertex 0 (block 0) and vertex 12 (block 1)
        jumpers = [
            JumperInfo(vertex_id=0, source_block=0, target_block=1, r=4),
            JumperInfo(vertex_id=12, source_block=1, target_block=0, r=3),
        ]
        jumper_map = {j.vertex_id: j for j in jumpers}
        max_r = max(j.r for j in jumpers)
        path_counts = precompute_path_counts(
            graph.adjacency, graph.block_assignments, graph.K, max_r,
        )
        indptr = graph.adjacency.indptr
        indices = graph.adjacency.indices

        # Generate walks starting from jumper vertex 0
        compliant = 0
        total = 0
        for seed in range(100):
            rng = np.random.default_rng(seed)
            result = generate_single_guided_walk(
                0, 20, rng, graph, jumper_map, path_counts, indptr, indices,
            )
            if result is None:
                continue
            walk, events = result
            total += 1
            all_ok = True
            for event in events:
                if event.expected_arrival_step < len(walk):
                    actual = graph.block_assignments[
                        walk[event.expected_arrival_step]
                    ]
                    if actual != event.target_block:
                        all_ok = False
            if all_ok:
                compliant += 1

        assert total > 0, "No walks generated"
        # All non-discarded walks should be compliant
        assert compliant == total, (
            f"Only {compliant}/{total} walks fully compliant"
        )


class TestPathCountNormalization:
    """Test that path-count vectors are free of NaN and Inf."""

    def test_path_count_normalization(self) -> None:
        # Use a larger graph to stress normalization
        graph = _make_small_graph(n=20, K=2, seed=42)
        max_r = 64

        path_counts = precompute_path_counts(
            graph.adjacency, graph.block_assignments, graph.K, max_r,
        )

        for tb in range(graph.K):
            for k in range(max_r + 1):
                vec = path_counts[tb][k]
                assert not np.any(np.isnan(vec)), (
                    f"NaN in path_counts[{tb}][{k}]"
                )
                assert not np.any(np.isinf(vec)), (
                    f"Inf in path_counts[{tb}][{k}]"
                )
                assert np.all(vec >= 0), (
                    f"Negative in path_counts[{tb}][{k}]"
                )


class TestReproducibilitySameSeed:
    """Test that same seed produces identical walks."""

    def test_reproducibility_same_seed(self) -> None:
        graph = _make_small_graph(n=10, K=2)
        config = _make_small_config(n=10)
        jumpers = [
            JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3),
        ]

        result1 = generate_walks(
            graph, jumpers, config, seed=42,
            target_n_walks=20, min_jumper_fraction=0.3,
        )
        result2 = generate_walks(
            graph, jumpers, config, seed=42,
            target_n_walks=20, min_jumper_fraction=0.3,
        )

        np.testing.assert_array_equal(result1.walks, result2.walks)
        np.testing.assert_array_equal(result1.walk_seeds, result2.walk_seeds)
        assert len(result1.events) == len(result2.events)
        for e1, e2 in zip(result1.events, result2.events):
            assert len(e1) == len(e2)
            for ev1, ev2 in zip(e1, e2):
                assert ev1 == ev2


class TestInfeasibleWalksDiscarded:
    """Test that infeasible walks are discarded and replaced."""

    def test_infeasible_walks_discarded(self) -> None:
        # Use a graph with moderate cross-block connectivity so some walks
        # are feasible and some may be infeasible (or at least discardable).
        # The key behavior: generate_walks handles discards and still produces
        # the requested number of walks.
        graph = _make_small_graph(n=20, K=2, seed=42)

        # Jumper at vertex 0 in block 0, must reach block 1 in 3 steps
        jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3)

        config = _make_small_config(n=20)

        # generate_walks should handle any infeasible walks gracefully
        result = generate_walks(
            graph,
            [jumper],
            config,
            seed=42,
            target_n_walks=15,
            min_jumper_fraction=0.3,
        )

        # Must produce exactly the target number of walks
        assert result.walks.shape[0] == 15, (
            f"Expected 15 walks, got {result.walks.shape[0]}"
        )
        # All walks must have valid edges
        indptr = graph.adjacency.indptr
        indices = graph.adjacency.indices
        for wi in range(result.walks.shape[0]):
            for step in range(result.walks.shape[1] - 1):
                u = result.walks[wi, step]
                v = result.walks[wi, step + 1]
                neighbors = indices[indptr[u]:indptr[u + 1]]
                assert v in neighbors, (
                    f"Walk {wi} step {step}: edge {u}->{v} not valid"
                )
