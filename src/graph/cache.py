"""Graph caching by config hash with gzip-compressed sparse matrix storage.

Caches generated graphs to disk to avoid redundant O(n^2) graph generation
and O(r * nnz * n_jumpers) non-triviality verification across sweep configs
that share the same graph parameters.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse

from src.config.experiment import ExperimentConfig
from src.config.hashing import graph_config_hash
from src.graph.dcsbm import generate_dcsbm_graph
from src.graph.jumpers import JumperInfo, designate_jumpers
from src.graph.types import GraphData

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache/graphs")


def graph_cache_key(config: ExperimentConfig) -> str:
    """Compute cache key for a graph configuration.

    Key = graph_config_hash + seed. Same graph params + same seed = cache hit.
    Non-graph params (description, tags, model config) don't affect the key.

    Args:
        config: Full experiment configuration.

    Returns:
        Cache key string like "a1b2c3d4e5f6g7h8_s42".
    """
    return f"{graph_config_hash(config)}_s{config.seed}"


def _cache_path(
    config: ExperimentConfig, cache_dir: Path = DEFAULT_CACHE_DIR
) -> Path:
    """Compute the directory path for a cached graph."""
    return cache_dir / graph_cache_key(config)


def save_graph(
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Save a generated graph and its jumpers to the cache.

    Stores:
    - adjacency.npz: scipy sparse matrix
    - metadata.json: block assignments, theta, generation params
    - jumpers.json: jumper designations

    Args:
        graph_data: The generated graph.
        jumpers: List of jumper designations.
        config: Experiment configuration.
        cache_dir: Root cache directory.

    Returns:
        Path to the cache directory for this graph.
    """
    cache_path = _cache_path(config, cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save adjacency as scipy sparse npz
    scipy.sparse.save_npz(str(cache_path / "adjacency.npz"), graph_data.adjacency)

    # Save metadata as JSON
    metadata = {
        "n": graph_data.n,
        "K": graph_data.K,
        "block_size": graph_data.block_size,
        "generation_seed": graph_data.generation_seed,
        "attempt": graph_data.attempt,
        "block_assignments": graph_data.block_assignments.tolist(),
        "theta": graph_data.theta.tolist(),
        "config_hash": graph_config_hash(config),
        "seed": config.seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save jumpers as JSON (convert numpy ints to Python ints)
    jumper_data = [
        {
            "vertex_id": int(j.vertex_id),
            "source_block": int(j.source_block),
            "target_block": int(j.target_block),
            "r": int(j.r),
        }
        for j in jumpers
    ]
    with open(cache_path / "jumpers.json", "w") as f:
        json.dump(jumper_data, f, indent=2)

    log.info("Graph cached at %s", cache_path)
    return cache_path


def load_graph(
    config: ExperimentConfig, cache_dir: Path = DEFAULT_CACHE_DIR
) -> tuple[GraphData, list[JumperInfo]] | None:
    """Load a cached graph if it exists.

    Args:
        config: Experiment configuration.
        cache_dir: Root cache directory.

    Returns:
        (GraphData, list[JumperInfo]) if cache hit, None if cache miss.
    """
    cache_path = _cache_path(config, cache_dir)

    # Check all required files exist
    required_files = ["adjacency.npz", "metadata.json", "jumpers.json"]
    for fname in required_files:
        if not (cache_path / fname).exists():
            return None

    # Load adjacency
    adjacency = scipy.sparse.load_npz(str(cache_path / "adjacency.npz"))

    # Load metadata
    with open(cache_path / "metadata.json") as f:
        metadata = json.load(f)

    graph_data = GraphData(
        adjacency=adjacency,
        block_assignments=np.array(metadata["block_assignments"]),
        theta=np.array(metadata["theta"]),
        n=metadata["n"],
        K=metadata["K"],
        block_size=metadata["block_size"],
        generation_seed=metadata["generation_seed"],
        attempt=metadata["attempt"],
    )

    # Load jumpers
    with open(cache_path / "jumpers.json") as f:
        jumper_data = json.load(f)

    jumpers = [
        JumperInfo(
            vertex_id=j["vertex_id"],
            source_block=j["source_block"],
            target_block=j["target_block"],
            r=j["r"],
        )
        for j in jumper_data
    ]

    log.info("Graph loaded from cache: %s", cache_path)
    return graph_data, jumpers


def generate_or_load_graph(
    config: ExperimentConfig,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_retries: int = 10,
) -> tuple[GraphData, list[JumperInfo]]:
    """Generate a graph or load from cache if available.

    On cache miss: generates graph, designates jumpers, saves to cache.
    On cache hit: loads from disk without regeneration.

    Args:
        config: Full experiment configuration.
        cache_dir: Root cache directory.
        max_retries: Maximum generation attempts.

    Returns:
        (GraphData, list[JumperInfo]) tuple.
    """
    key = graph_cache_key(config)

    # Try cache first
    cached = load_graph(config, cache_dir)
    if cached is not None:
        log.info("Cache hit for %s", key)
        return cached

    # Cache miss: generate
    log.info("Cache miss for %s, generating...", key)
    graph_data = generate_dcsbm_graph(config, max_retries)

    # Use offset seed for jumper designation to avoid correlation
    # with graph generation seed
    jumper_rng = np.random.default_rng(config.seed + 1000)
    jumpers = designate_jumpers(graph_data, config, jumper_rng)

    # Save to cache
    save_graph(graph_data, jumpers, config, cache_dir)

    return graph_data, jumpers
