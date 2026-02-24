"""Tests for block jumper designation, non-triviality verification, and variable r."""

import numpy as np
import pytest
import scipy.sparse

from src.config.defaults import ANCHOR_CONFIG
from src.graph.dcsbm import generate_dcsbm_graph
from src.graph.jumpers import (
    R_SCALES,
    JumperInfo,
    compute_r_values,
    designate_jumpers,
)
from src.graph.validation import (
    check_non_trivial,
    reachable_blocks_at_distance,
    verify_all_jumpers,
)


class TestComputeRValues:
    """Tests for r-value computation from context window size."""

    def test_anchor_w64(self) -> None:
        """w=64 should produce the expected set of r values."""
        r_values = compute_r_values(64)
        # 0.5*64=32, 0.7*64=44.8->45, 0.9*64=57.6->58, 1.0*64=64,
        # 1.1*64=70.4->70, 1.3*64=83.2->83, 1.5*64=96, 2.0*64=128
        expected = [32, 45, 58, 64, 70, 83, 96, 128]
        assert r_values == expected

    def test_deduplication(self) -> None:
        """Very small w might cause collisions that need deduplication."""
        r_values = compute_r_values(1)
        # All scales * 1 rounded: 1,1,1,1,1,1,2,2 -> deduplicated to [1,2]
        assert len(r_values) == len(set(r_values))

    def test_minimum_r_is_one(self) -> None:
        """All r values should be at least 1."""
        r_values = compute_r_values(1)
        assert all(r >= 1 for r in r_values)

    def test_sorted_output(self) -> None:
        r_values = compute_r_values(64)
        assert r_values == sorted(r_values)

    def test_all_r_scales_represented_at_w64(self) -> None:
        """With w=64, all 8 scales should produce distinct r values."""
        r_values = compute_r_values(64)
        assert len(r_values) == 8


class TestReachability:
    """Tests for sparse reachability computation."""

    def test_simple_chain(self) -> None:
        """Linear chain: 0->1->2->3. From 0 at distance 2: reach vertex 2."""
        adj = scipy.sparse.csr_matrix(
            np.array(
                [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                dtype=np.float64,
            )
        )
        blocks = np.array([0, 0, 1, 1])
        reachable = reachable_blocks_at_distance(adj, 0, 2, blocks, 2)
        assert reachable == {1}

    def test_branching_paths(self) -> None:
        """0->1->2 and 0->3 at distance 2: 0->1->2 (block 1), 0->3->0 possible."""
        adj = scipy.sparse.csr_matrix(
            np.array(
                [[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                dtype=np.float64,
            )
        )
        blocks = np.array([0, 0, 1, 1])
        reachable = reachable_blocks_at_distance(adj, 0, 2, blocks, 2)
        assert 0 in reachable and 1 in reachable

    def test_no_paths(self) -> None:
        """Isolated vertex has no reachable blocks."""
        adj = scipy.sparse.csr_matrix(
            np.array(
                [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                dtype=np.float64,
            )
        )
        blocks = np.array([0, 1, 1])
        reachable = reachable_blocks_at_distance(adj, 0, 1, blocks, 2)
        assert reachable == set()


class TestNonTriviality:
    """Tests for non-triviality checks."""

    def test_non_trivial_assignment(self) -> None:
        """When both target and non-target blocks are reachable -> non-trivial."""
        adj = scipy.sparse.csr_matrix(
            np.array(
                [[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                dtype=np.float64,
            )
        )
        blocks = np.array([0, 0, 1, 1])
        assert check_non_trivial(adj, 0, 1, 2, blocks, 2)

    def test_trivial_only_target_reachable(self) -> None:
        """When only target block is reachable -> trivial (should fail)."""
        # 0->1->2, blocks: [0,0,1]. At distance 2 from 0: only block 1.
        adj = scipy.sparse.csr_matrix(
            np.array(
                [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
                dtype=np.float64,
            )
        )
        blocks = np.array([0, 0, 1])
        assert not check_non_trivial(adj, 0, 1, 2, blocks, 2)

    def test_unreachable_target(self) -> None:
        """When target block is not reachable -> fails."""
        adj = scipy.sparse.csr_matrix(
            np.array(
                [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                dtype=np.float64,
            )
        )
        blocks = np.array([0, 0, 1])
        # Target block 1 is not reachable at distance 2 from vertex 0
        assert not check_non_trivial(adj, 0, 1, 2, blocks, 2)


class TestDesignateJumpers:
    """Tests for the full jumper designation pipeline."""

    @pytest.fixture
    def anchor_graph(self):
        """Generate graph with anchor config for reuse."""
        return generate_dcsbm_graph(ANCHOR_CONFIG)

    def test_designate_jumpers_count(self, anchor_graph) -> None:
        """Total jumpers should be n_jumpers_per_block * K."""
        rng = np.random.default_rng(ANCHOR_CONFIG.seed)
        jumpers = designate_jumpers(anchor_graph, ANCHOR_CONFIG, rng)
        expected_count = ANCHOR_CONFIG.graph.n_jumpers_per_block * ANCHOR_CONFIG.graph.K
        assert len(jumpers) == expected_count

    def test_target_block_different_from_source(self, anchor_graph) -> None:
        """Each jumper's target block must differ from source block."""
        rng = np.random.default_rng(ANCHOR_CONFIG.seed)
        jumpers = designate_jumpers(anchor_graph, ANCHOR_CONFIG, rng)
        for j in jumpers:
            assert j.source_block != j.target_block, (
                f"Jumper v={j.vertex_id}: source==target=={j.source_block}"
            )

    def test_all_r_values_represented(self, anchor_graph) -> None:
        """All r values from compute_r_values(w) should be represented.

        With 2 jumpers/block * 4 blocks = 8 jumpers and 8 r_values,
        each r should appear at least once.
        """
        rng = np.random.default_rng(ANCHOR_CONFIG.seed)
        jumpers = designate_jumpers(anchor_graph, ANCHOR_CONFIG, rng)
        assigned_r = {j.r for j in jumpers}
        expected_r = set(compute_r_values(ANCHOR_CONFIG.training.w))
        assert assigned_r == expected_r, (
            f"Missing r values: {expected_r - assigned_r}"
        )

    def test_non_trivial_verification(self, anchor_graph) -> None:
        """All designated jumpers must pass non-triviality check."""
        rng = np.random.default_rng(ANCHOR_CONFIG.seed)
        jumpers = designate_jumpers(anchor_graph, ANCHOR_CONFIG, rng)
        failures = verify_all_jumpers(
            anchor_graph.adjacency,
            jumpers,
            anchor_graph.block_assignments,
            anchor_graph.K,
        )
        assert failures == [], f"Non-trivial check failures: {failures}"

    def test_jumper_info_frozen(self) -> None:
        """JumperInfo should be immutable."""
        j = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=32)
        with pytest.raises(AttributeError):
            j.vertex_id = 1  # type: ignore[misc]

    def test_minimum_one_jumper_per_block(self, anchor_graph) -> None:
        """Even with minimum settings, each block should have >= 1 jumper."""
        rng = np.random.default_rng(ANCHOR_CONFIG.seed)
        jumpers = designate_jumpers(anchor_graph, ANCHOR_CONFIG, rng)
        blocks_with_jumpers = {j.source_block for j in jumpers}
        for b in range(ANCHOR_CONFIG.graph.K):
            assert b in blocks_with_jumpers, (
                f"Block {b} has no jumpers"
            )

    def test_jumpers_sorted_by_block_and_vertex(self, anchor_graph) -> None:
        """Jumpers should be sorted by (source_block, vertex_id)."""
        rng = np.random.default_rng(ANCHOR_CONFIG.seed)
        jumpers = designate_jumpers(anchor_graph, ANCHOR_CONFIG, rng)
        keys = [(j.source_block, j.vertex_id) for j in jumpers]
        assert keys == sorted(keys)

    def test_jumper_reassignment(self) -> None:
        """Test reassignment when initial vertex fails non-triviality.

        Create a small graph where specific vertices have limited
        reachability to force reassignment.
        """
        # 6 vertices, 2 blocks of 3
        # Block 0: vertices 0,1,2. Block 1: vertices 3,4,5
        # Make vertex 0 only reach block 0 at distance 2 (trivial if target=1)
        # But vertex 1 can reach both blocks
        from dataclasses import replace
        from src.config.experiment import GraphConfig, TrainingConfig
        from src.graph.types import GraphData

        adj = scipy.sparse.csr_matrix(
            np.array(
                [
                    # v0 only connects within block 0
                    [0, 1, 1, 0, 0, 0],
                    # v1 connects to both blocks
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 0, 0, 0],
                    # Block 1 has internal connections
                    [0, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 1, 0],
                ],
                dtype=np.float64,
            )
        )

        graph = GraphData(
            adjacency=adj,
            block_assignments=np.array([0, 0, 0, 1, 1, 1]),
            theta=np.ones(6),
            n=6,
            K=2,
            block_size=3,
            generation_seed=42,
            attempt=0,
        )

        config = replace(
            ANCHOR_CONFIG,
            graph=GraphConfig(n=6, K=2, p_in=0.5, p_out=0.1, n_jumpers_per_block=1),
            training=TrainingConfig(
                w=2, walk_length=4, corpus_size=600, r=2,
                learning_rate=3e-4, batch_size=64, max_steps=50000,
                eval_interval=1000, checkpoint_interval=5000,
            ),
        )

        rng = np.random.default_rng(42)
        jumpers = designate_jumpers(graph, config, rng)

        # Should have at least some jumpers (reassignment may find valid vertices)
        assert len(jumpers) >= 1
