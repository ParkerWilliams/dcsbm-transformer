"""Experiment configuration system with frozen, hashable, serializable dataclasses."""

from src.config.experiment import (
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
    SweepConfig,
)
from src.config.defaults import ANCHOR_CONFIG
from src.config.hashing import config_hash, graph_config_hash, full_config_hash
from src.config.serialization import config_to_json, config_from_json

__all__ = [
    "ExperimentConfig",
    "GraphConfig",
    "ModelConfig",
    "TrainingConfig",
    "SweepConfig",
    "ANCHOR_CONFIG",
    "config_hash",
    "graph_config_hash",
    "full_config_hash",
    "config_to_json",
    "config_from_json",
]
