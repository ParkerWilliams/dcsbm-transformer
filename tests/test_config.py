"""Tests for the experiment configuration system."""

import json
import re

import pytest
from dataclasses import FrozenInstanceError, replace

from src.config import (
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
    SweepConfig,
    ANCHOR_CONFIG,
    config_hash,
    graph_config_hash,
    full_config_hash,
    config_to_json,
    config_from_json,
)


class TestAnchorConfigDefaults:
    """ANCHOR_CONFIG has correct locked values."""

    def test_anchor_config_defaults(self):
        assert ANCHOR_CONFIG.graph.n == 500
        assert ANCHOR_CONFIG.training.w == 64
        assert ANCHOR_CONFIG.training.corpus_size == 200_000
        assert ANCHOR_CONFIG.model.d_model == 128
        assert ANCHOR_CONFIG.model.n_layers == 4
        assert ANCHOR_CONFIG.model.n_heads == 1
        assert ANCHOR_CONFIG.training.r == 57
        assert ANCHOR_CONFIG.training.walk_length == 256
        assert ANCHOR_CONFIG.seed == 42
        assert ANCHOR_CONFIG.graph.K == 4
        assert ANCHOR_CONFIG.graph.p_in == 0.25
        assert ANCHOR_CONFIG.graph.p_out == 0.03
        assert ANCHOR_CONFIG.graph.n_jumpers_per_block == 2


class TestConfigImmutability:
    """Frozen dataclasses prevent mutation."""

    def test_config_frozen(self):
        with pytest.raises(FrozenInstanceError):
            ANCHOR_CONFIG.seed = 99  # type: ignore[misc]

    def test_graph_config_frozen(self):
        with pytest.raises(FrozenInstanceError):
            ANCHOR_CONFIG.graph.n = 1000  # type: ignore[misc]

    def test_model_config_frozen(self):
        with pytest.raises(FrozenInstanceError):
            ANCHOR_CONFIG.model.d_model = 256  # type: ignore[misc]


class TestConfigRoundTrip:
    """JSON serialization round-trip preserves identity."""

    def test_config_round_trip_hash(self):
        json_str = config_to_json(ANCHOR_CONFIG)
        restored = config_from_json(json_str)
        assert config_hash(ANCHOR_CONFIG) == config_hash(restored)

    def test_config_round_trip_values(self):
        json_str = config_to_json(ANCHOR_CONFIG)
        restored = config_from_json(json_str)
        assert restored.graph.n == ANCHOR_CONFIG.graph.n
        assert restored.training.w == ANCHOR_CONFIG.training.w
        assert restored.seed == ANCHOR_CONFIG.seed

    def test_config_with_sweep_round_trip(self):
        cfg = replace(ANCHOR_CONFIG, sweep=SweepConfig())
        json_str = config_to_json(cfg)
        restored = config_from_json(json_str)
        assert config_hash(cfg) == config_hash(restored)
        assert restored.sweep is not None
        assert restored.sweep.seeds == (42, 123, 7)


class TestConfigHashing:
    """Hashing behavior for graph cache and full identity."""

    def test_graph_hash_ignores_seed(self):
        cfg2 = replace(ANCHOR_CONFIG, seed=99)
        assert graph_config_hash(ANCHOR_CONFIG) == graph_config_hash(cfg2)

    def test_full_hash_includes_seed(self):
        cfg2 = replace(ANCHOR_CONFIG, seed=99)
        assert full_config_hash(ANCHOR_CONFIG) != full_config_hash(cfg2)

    def test_config_hash_deterministic(self):
        h1 = config_hash(ANCHOR_CONFIG)
        h2 = config_hash(ANCHOR_CONFIG)
        assert h1 == h2

    def test_config_hash_is_hex_string(self):
        h = full_config_hash(ANCHOR_CONFIG)
        assert len(h) == 16
        assert re.match(r"^[0-9a-f]{16}$", h)

    def test_different_configs_different_hash(self):
        cfg2 = replace(
            ANCHOR_CONFIG,
            graph=GraphConfig(n=1000),
            training=TrainingConfig(corpus_size=200_000),
        )
        assert full_config_hash(ANCHOR_CONFIG) != full_config_hash(cfg2)


class TestConfigValidation:
    """Cross-parameter validation catches invalid configs."""

    def test_validation_walk_length(self):
        with pytest.raises(ValueError, match="walk_length"):
            ExperimentConfig(training=TrainingConfig(walk_length=10, w=64))

    def test_validation_corpus_size(self):
        with pytest.raises(ValueError, match="corpus_size"):
            ExperimentConfig(training=TrainingConfig(corpus_size=100))

    def test_validation_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            ExperimentConfig(model=ModelConfig(n_heads=4))

    def test_validation_r_exceeds_walk_length(self):
        with pytest.raises(ValueError, match="r"):
            ExperimentConfig(training=TrainingConfig(r=300, walk_length=256))

    def test_valid_config_passes(self):
        cfg = ExperimentConfig()
        assert cfg.graph.n == 500


class TestSerializationStrict:
    """Strict mode rejects unknown keys."""

    def test_serialization_strict_rejects_extra_keys(self):
        json_str = config_to_json(ANCHOR_CONFIG)
        data = json.loads(json_str)
        data["unknown_field"] = "sneaky"
        with pytest.raises(Exception):
            config_from_json(json.dumps(data))
