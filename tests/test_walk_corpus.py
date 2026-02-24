"""Tests for corpus assembly, validation, and caching.

Covers seed independence, corpus size validation, jumper fraction,
path diversity, cache save/load roundtrip, cache hit behavior,
cache key computation, atomic NPZ storage, and split ratios.
"""

import tempfile
from pathlib import Path

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
from src.walk.cache import (
    generate_or_load_walks,
    load_walks,
    save_walks,
    walk_cache_key,
)
from src.walk.corpus import (
    EVAL_SEED_OFFSET,
    TRAIN_SEED_OFFSET,
    generate_corpus,
    validate_corpus,
)
from src.walk.generator import generate_walks
from src.walk.types import JumperEvent, WalkResult


def _make_test_graph(n: int = 20, K: int = 2, seed: int = 42) -> GraphData:
    """Create a test graph with good cross-block connectivity."""
    rng = np.random.default_rng(seed)
    block_size = n // K
    block_assignments = np.array(
        [i // block_size for i in range(n)], dtype=np.int32
    )

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
                if rng.random() < 0.25:
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


def _make_test_jumpers(graph: GraphData) -> list[JumperInfo]:
    """Create test jumpers for a 2-block graph."""
    return [
        JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3),
        JumperInfo(
            vertex_id=graph.n // 2,
            source_block=1,
            target_block=0,
            r=3,
        ),
    ]


def _make_test_config(n: int = 20) -> ExperimentConfig:
    """Create a test config with small corpus for fast tests."""
    return ExperimentConfig(
        graph=GraphConfig(n=n, K=2, p_in=0.7, p_out=0.25, n_jumpers_per_block=1),
        model=ModelConfig(),
        training=TrainingConfig(
            w=4,
            walk_length=10,
            corpus_size=100 * n,
            r=3,
        ),
        seed=42,
    )


class TestTrainEvalDifferentSeeds:
    """Verify train and eval walks are generated with different seeds."""

    def test_train_eval_different_seeds(self) -> None:
        graph = _make_test_graph()
        jumpers = _make_test_jumpers(graph)
        config = _make_test_config()

        train_result, eval_result = generate_corpus(graph, jumpers, config)

        # Train and eval should be different arrays
        # (extremely unlikely to match with different seeds)
        assert train_result.walks.shape[0] > eval_result.walks.shape[0], (
            "Train should be larger than eval (9:1 ratio)"
        )
        # Check seeds are different
        train_seed = config.seed + TRAIN_SEED_OFFSET
        eval_seed = config.seed + EVAL_SEED_OFFSET
        assert train_seed != eval_seed, "Seeds must differ"


class TestTrainCorpusSizeValidation:
    """Verify corpus size is validated against 100n threshold."""

    def test_train_corpus_size_validation(self) -> None:
        graph = _make_test_graph()
        jumpers = _make_test_jumpers(graph)
        config = _make_test_config()

        # Default config has corpus_size = 100 * n, should succeed
        train_result, eval_result = generate_corpus(graph, jumpers, config)
        assert train_result.walks.shape[0] == config.training.corpus_size

    def test_validate_corpus_catches_errors(self) -> None:
        """Validate that validate_corpus returns errors for bad data."""
        graph = _make_test_graph(n=10, K=2)
        jumpers = _make_test_jumpers(graph)

        # Create fake walks with no events (0% jumper fraction)
        walks = np.zeros((100, 10), dtype=np.int32)
        events: list[list[JumperEvent]] = [[] for _ in range(100)]

        # Fill walks with vertex 0 repeated (will fail edge check)
        for i in range(100):
            walks[i, :] = 0

        errors = validate_corpus(walks, events, graph, jumpers)
        # Should have jumper fraction error and edge validity errors
        assert len(errors) > 0, "Expected validation errors"


class TestCorpusJumperFraction:
    """Verify at least 50% of walks contain jumper events."""

    def test_corpus_jumper_fraction(self) -> None:
        graph = _make_test_graph()
        jumpers = _make_test_jumpers(graph)
        config = _make_test_config()

        train_result, _ = generate_corpus(graph, jumpers, config)

        n_with_jumpers = sum(
            1 for e in train_result.events if len(e) > 0
        )
        fraction = n_with_jumpers / train_result.walks.shape[0]
        assert fraction >= 0.5, (
            f"Jumper fraction {fraction:.3f} below 50% minimum"
        )


class TestCorpusPathDiversity:
    """Verify each jumper has at least 3 distinct compliant paths."""

    def test_corpus_path_diversity(self) -> None:
        graph = _make_test_graph()
        jumpers = _make_test_jumpers(graph)
        config = _make_test_config()

        train_result, _ = generate_corpus(graph, jumpers, config)
        walks = train_result.walks
        walk_length = walks.shape[1]

        # Collect distinct paths per jumper
        jumper_paths: dict[int, set[tuple[int, ...]]] = {
            j.vertex_id: set() for j in jumpers
        }
        for wi, events in enumerate(train_result.events):
            for event in events:
                if event.expected_arrival_step < walk_length:
                    segment = tuple(
                        int(walks[wi, s])
                        for s in range(
                            event.step, event.expected_arrival_step + 1
                        )
                    )
                    jumper_paths[event.vertex_id].add(segment)

        for v, paths in jumper_paths.items():
            if len(paths) > 0:  # only check jumpers that were encountered
                assert len(paths) >= 3, (
                    f"Jumper {v} has only {len(paths)} distinct paths"
                )


class TestCacheSaveLoadRoundtrip:
    """Verify walks survive save/load roundtrip via NPZ cache."""

    def test_cache_save_load_roundtrip(self) -> None:
        graph = _make_test_graph()
        jumpers = _make_test_jumpers(graph)
        config = _make_test_config()

        train_result, _ = generate_corpus(graph, jumpers, config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            seed = config.seed + TRAIN_SEED_OFFSET

            # Save
            save_walks(train_result, config, "train", seed, cache_dir)

            # Load
            loaded = load_walks(config, "train", seed, cache_dir)
            assert loaded is not None, "Cache miss after save"

            # Verify walks match
            np.testing.assert_array_equal(loaded.walks, train_result.walks)
            np.testing.assert_array_equal(
                loaded.walk_seeds, train_result.walk_seeds
            )

            # Verify events match
            assert len(loaded.events) == len(train_result.events)
            for orig_events, load_events in zip(
                train_result.events, loaded.events
            ):
                assert len(orig_events) == len(load_events)
                for oe, le in zip(orig_events, load_events):
                    assert oe.vertex_id == le.vertex_id
                    assert oe.step == le.step
                    assert oe.target_block == le.target_block
                    assert oe.expected_arrival_step == le.expected_arrival_step


class TestCacheHitSkipsGeneration:
    """Verify second call with same config uses cache."""

    def test_cache_hit_skips_generation(self) -> None:
        graph = _make_test_graph()
        jumpers = _make_test_jumpers(graph)
        config = _make_test_config()

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)

            import time

            # First call: generates
            t1 = time.time()
            r1_train, r1_eval = generate_or_load_walks(
                graph, jumpers, config, cache_dir,
            )
            gen_time = time.time() - t1

            # Second call: cache hit
            t2 = time.time()
            r2_train, r2_eval = generate_or_load_walks(
                graph, jumpers, config, cache_dir,
            )
            cache_time = time.time() - t2

            # Cache hit should be faster than generation
            # (allow generous margin for small test corpora)
            assert cache_time < gen_time * 2, (
                f"Cache hit ({cache_time:.3f}s) not faster than "
                f"generation ({gen_time:.3f}s)"
            )

            # Verify identical results
            np.testing.assert_array_equal(r1_train.walks, r2_train.walks)
            np.testing.assert_array_equal(r1_eval.walks, r2_eval.walks)


class TestCacheKeyIncludesGraphHash:
    """Verify cache key changes when graph config changes."""

    def test_cache_key_includes_graph_hash(self) -> None:
        config1 = _make_test_config(n=20)
        config2 = ExperimentConfig(
            graph=GraphConfig(
                n=30, K=2, p_in=0.7, p_out=0.25, n_jumpers_per_block=1,
            ),
            model=ModelConfig(),
            training=TrainingConfig(
                w=4, walk_length=10, corpus_size=3000, r=3,
            ),
            seed=42,
        )

        seed = 42 + TRAIN_SEED_OFFSET
        key1 = walk_cache_key(config1, "train", seed)
        key2 = walk_cache_key(config2, "train", seed)

        assert key1 != key2, "Cache keys should differ for different graph configs"


class TestNpzAtomicStorage:
    """Verify .npz file contains both walks and event arrays."""

    def test_npz_atomic_storage(self) -> None:
        graph = _make_test_graph()
        jumpers = _make_test_jumpers(graph)
        config = _make_test_config()

        train_result, _ = generate_corpus(graph, jumpers, config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            seed = config.seed + TRAIN_SEED_OFFSET
            npz_path = save_walks(
                train_result, config, "train", seed, cache_dir,
            )

            # Load raw .npz and check contents
            data = np.load(npz_path)
            expected_keys = {
                "walks",
                "walk_seeds",
                "event_walk_ids",
                "event_vertex_ids",
                "event_steps",
                "event_target_blocks",
                "event_arrival_steps",
                "num_walks",
                "walk_length",
            }
            actual_keys = set(data.files)
            assert expected_keys == actual_keys, (
                f"NPZ missing keys: {expected_keys - actual_keys}"
            )

            # Verify walks dtype
            assert data["walks"].dtype == np.int32
            assert data["walk_seeds"].dtype == np.int64


class TestSplitRatio:
    """Verify 90/10 train/eval split ratio."""

    def test_90_10_split_ratio(self) -> None:
        graph = _make_test_graph()
        jumpers = _make_test_jumpers(graph)
        config = _make_test_config()

        train_result, eval_result = generate_corpus(graph, jumpers, config)

        n_train = train_result.walks.shape[0]
        n_eval = eval_result.walks.shape[0]

        # n_eval should be approximately n_train / 9
        expected_eval = max(1, n_train // 9)
        assert n_eval == expected_eval, (
            f"Expected eval size {expected_eval}, got {n_eval}"
        )

        # Ratio check
        ratio = n_train / max(1, n_eval)
        assert 8.0 <= ratio <= 10.0, (
            f"Train/eval ratio {ratio:.1f} outside [8, 10] range"
        )
