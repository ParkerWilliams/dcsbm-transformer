"""Anchor configuration — single source of truth for default experiment parameters."""

from src.config.experiment import ExperimentConfig

# Anchor config with all locked defaults from the project specification.
# Instantiated with all-default values: n=500, w=64, t=200k, d_model=128,
# n_layers=4, n_heads=1, r=57, walk_length=256, seed=42.
ANCHOR_CONFIG = ExperimentConfig()
