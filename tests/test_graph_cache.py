"""Tests for graph caching by config hash."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.config.defaults import ANCHOR_CONFIG
from src.config.experiment import ModelConfig
from src.config.hashing import graph_config_hash
from src.graph.cache import (
    generate_or_load_graph,
    graph_cache_key,
    load_graph,
    save_graph,
)
from src.graph.dcsbm import generate_dcsbm_graph
from src.graph.jumpers import JumperInfo, designate_jumpers


class TestCacheKey:
    """Tests for cache key computation."""

    def test_cache_key_includes_seed(self) -> None:
        key = graph_cache_key(ANCHOR_CONFIG)
        assert f"_s{ANCHOR_CONFIG.seed}" in key

    def test_cache_key_includes_graph_hash(self) -> None:
        key = graph_cache_key(ANCHOR_CONFIG)
        assert graph_config_hash(ANCHOR_CONFIG) in key

    def test_cache_key_same_for_same_config(self) -> None:
        key1 = graph_cache_key(ANCHOR_CONFIG)
        key2 = graph_cache_key(ANCHOR_CONFIG)
        assert key1 == key2

    def test_cache_key_differs_for_different_seed(self) -> None:
        cfg2 = replace(ANCHOR_CONFIG, seed=99)
        assert graph_cache_key(ANCHOR_CONFIG) != graph_cache_key(cfg2)

    def test_cache_key_differs_for_different_graph_params(self) -> None:
        from src.config.experiment import GraphConfig
        cfg2 = replace(ANCHOR_CONFIG, graph=GraphConfig(n=1000, K=4))
        assert graph_cache_key(ANCHOR_CONFIG) != graph_cache_key(cfg2)

    def test_cache_key_ignores_non_graph_params(self) -> None:
        """Description, tags, and model params should not affect cache key."""
        cfg2 = replace(ANCHOR_CONFIG, description="test run")
        assert graph_cache_key(ANCHOR_CONFIG) == graph_cache_key(cfg2)

        cfg3 = replace(ANCHOR_CONFIG, tags=("foo", "bar"))
        assert graph_cache_key(ANCHOR_CONFIG) == graph_cache_key(cfg3)

        cfg4 = replace(ANCHOR_CONFIG, model=ModelConfig(d_model=256))
        assert graph_cache_key(ANCHOR_CONFIG) == graph_cache_key(cfg4)


class TestSaveLoad:
    """Tests for graph save/load round-trip."""

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        graph = generate_dcsbm_graph(ANCHOR_CONFIG)
        rng = np.random.default_rng(ANCHOR_CONFIG.seed + 1000)
        jumpers = designate_jumpers(graph, ANCHOR_CONFIG, rng)

        save_graph(graph, jumpers, ANCHOR_CONFIG, tmp_path)
        loaded = load_graph(ANCHOR_CONFIG, tmp_path)

        assert loaded is not None
        loaded_graph, loaded_jumpers = loaded

        # Adjacency should be identical
        diff = graph.adjacency - loaded_graph.adjacency
        assert diff.nnz == 0, "Adjacency matrices differ"

        # Block assignments should match
        assert np.array_equal(
            graph.block_assignments, loaded_graph.block_assignments
        )

        # Theta should match
        assert np.allclose(graph.theta, loaded_graph.theta)

        # Scalar fields
        assert loaded_graph.n == graph.n
        assert loaded_graph.K == graph.K
        assert loaded_graph.block_size == graph.block_size
        assert loaded_graph.generation_seed == graph.generation_seed
        assert loaded_graph.attempt == graph.attempt

        # Jumpers should match
        assert len(loaded_jumpers) == len(jumpers)
        for orig, loaded in zip(jumpers, loaded_jumpers):
            assert orig.vertex_id == loaded.vertex_id
            assert orig.source_block == loaded.source_block
            assert orig.target_block == loaded.target_block
            assert orig.r == loaded.r

    def test_load_returns_none_for_missing_cache(self, tmp_path: Path) -> None:
        result = load_graph(ANCHOR_CONFIG, tmp_path)
        assert result is None

    def test_cache_metadata_contains_expected_fields(
        self, tmp_path: Path
    ) -> None:
        graph = generate_dcsbm_graph(ANCHOR_CONFIG)
        rng = np.random.default_rng(ANCHOR_CONFIG.seed + 1000)
        jumpers = designate_jumpers(graph, ANCHOR_CONFIG, rng)

        cache_path = save_graph(graph, jumpers, ANCHOR_CONFIG, tmp_path)

        with open(cache_path / "metadata.json") as f:
            metadata = json.load(f)

        expected_fields = {
            "n", "K", "block_size", "generation_seed", "attempt",
            "block_assignments", "theta", "config_hash", "seed", "timestamp",
        }
        assert set(metadata.keys()) == expected_fields

    def test_cache_jumpers_roundtrip(self, tmp_path: Path) -> None:
        graph = generate_dcsbm_graph(ANCHOR_CONFIG)
        rng = np.random.default_rng(ANCHOR_CONFIG.seed + 1000)
        jumpers = designate_jumpers(graph, ANCHOR_CONFIG, rng)

        save_graph(graph, jumpers, ANCHOR_CONFIG, tmp_path)
        _, loaded_jumpers = load_graph(ANCHOR_CONFIG, tmp_path)

        for orig, loaded in zip(jumpers, loaded_jumpers):
            assert orig == loaded


class TestGenerateOrLoad:
    """Tests for the generate-or-load API."""

    def test_caches_on_first_call(self, tmp_path: Path) -> None:
        generate_or_load_graph(ANCHOR_CONFIG, tmp_path)

        # Check cache files exist
        key = graph_cache_key(ANCHOR_CONFIG)
        cache_path = tmp_path / key
        assert (cache_path / "adjacency.npz").exists()
        assert (cache_path / "metadata.json").exists()
        assert (cache_path / "jumpers.json").exists()

    def test_cache_hit_on_second_call(self, tmp_path: Path) -> None:
        g1, j1 = generate_or_load_graph(ANCHOR_CONFIG, tmp_path)
        g2, j2 = generate_or_load_graph(ANCHOR_CONFIG, tmp_path)

        # Adjacency should be identical
        diff = g1.adjacency - g2.adjacency
        assert diff.nnz == 0

        # Jumpers should match
        assert len(j1) == len(j2)
        for a, b in zip(j1, j2):
            assert a == b

    def test_different_seed_generates_different_graph(
        self, tmp_path: Path
    ) -> None:
        cfg2 = replace(ANCHOR_CONFIG, seed=99)
        g1, _ = generate_or_load_graph(ANCHOR_CONFIG, tmp_path)
        g2, _ = generate_or_load_graph(cfg2, tmp_path)

        # Different seeds should produce different graphs
        diff = g1.adjacency - g2.adjacency
        assert diff.nnz > 0, "Different seeds should produce different graphs"
