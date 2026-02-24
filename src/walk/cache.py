"""Walk caching with config-hash-based keys and atomic NPZ storage.

Follows the Phase 2 graph cache pattern: cache key computed from
relevant config parameters, stored as .npz files in a shared cache
directory. Walks and jumper event metadata are always stored atomically
in a single .npz archive to prevent desync.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from src.config.experiment import ExperimentConfig
from src.config.hashing import graph_config_hash
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.walk.corpus import EVAL_SEED_OFFSET, TRAIN_SEED_OFFSET, generate_corpus
from src.walk.types import JumperEvent, WalkResult

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path(".cache/walks")


def walk_cache_key(config: ExperimentConfig, split: str, seed: int) -> str:
    """Compute cache key for a walk corpus split.

    Includes graph config hash, walk parameters, split name, and seed
    so that any parameter change auto-invalidates the cache.

    Args:
        config: Experiment configuration.
        split: Split name ("train" or "eval").
        seed: Walk generation seed for this split.

    Returns:
        Cache key string like "a1b2c3d4e5f6g7h8_train".
    """
    key_dict = {
        "graph_hash": graph_config_hash(config),
        "walk_length": config.training.walk_length,
        "corpus_size": config.training.corpus_size,
        "split": split,
        "seed": seed,
    }
    serialized = json.dumps(
        key_dict,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        indent=None,
    )
    hash_hex = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    return f"{hash_hex}_{split}"


def save_walks(
    walk_result: WalkResult,
    config: ExperimentConfig,
    split: str,
    seed: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Save walks and event metadata atomically in a single .npz archive.

    Flattens per-walk event lists into parallel arrays for efficient
    storage and reconstruction.

    Args:
        walk_result: Walk generation results.
        config: Experiment configuration.
        split: Split name ("train" or "eval").
        seed: Walk generation seed.
        cache_dir: Root cache directory.

    Returns:
        Path to the saved .npz file.
    """
    key = walk_cache_key(config, split, seed)
    save_dir = cache_dir / key
    save_dir.mkdir(parents=True, exist_ok=True)
    npz_path = save_dir / "walks.npz"

    # Flatten events from list[list[JumperEvent]] into parallel arrays
    all_walk_ids: list[int] = []
    all_vertex_ids: list[int] = []
    all_steps: list[int] = []
    all_target_blocks: list[int] = []
    all_arrival_steps: list[int] = []

    for walk_id, events in enumerate(walk_result.events):
        for ev in events:
            all_walk_ids.append(walk_id)
            all_vertex_ids.append(ev.vertex_id)
            all_steps.append(ev.step)
            all_target_blocks.append(ev.target_block)
            all_arrival_steps.append(ev.expected_arrival_step)

    np.savez_compressed(
        npz_path,
        walks=walk_result.walks,
        walk_seeds=walk_result.walk_seeds,
        event_walk_ids=np.array(all_walk_ids, dtype=np.int32),
        event_vertex_ids=np.array(all_vertex_ids, dtype=np.int32),
        event_steps=np.array(all_steps, dtype=np.int32),
        event_target_blocks=np.array(all_target_blocks, dtype=np.int32),
        event_arrival_steps=np.array(all_arrival_steps, dtype=np.int32),
        num_walks=np.array(walk_result.walks.shape[0]),
        walk_length=np.array(walk_result.walks.shape[1]),
    )

    log.info("Walks saved to %s (%d walks)", npz_path, walk_result.walks.shape[0])
    return npz_path


def load_walks(
    config: ExperimentConfig,
    split: str,
    seed: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> WalkResult | None:
    """Load walks from cache if available.

    Reconstructs WalkResult including per-walk event lists from
    flattened parallel arrays in the .npz archive.

    Args:
        config: Experiment configuration.
        split: Split name ("train" or "eval").
        seed: Walk generation seed.
        cache_dir: Root cache directory.

    Returns:
        WalkResult if cache hit, None if cache miss.
    """
    key = walk_cache_key(config, split, seed)
    npz_path = cache_dir / key / "walks.npz"

    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    walks = data["walks"]
    walk_seeds = data["walk_seeds"]
    num_walks = int(data["num_walks"])

    # Reconstruct events from flat arrays
    event_walk_ids = data["event_walk_ids"]
    event_vertex_ids = data["event_vertex_ids"]
    event_steps = data["event_steps"]
    event_target_blocks = data["event_target_blocks"]
    event_arrival_steps = data["event_arrival_steps"]

    # Group events by walk_id
    events: list[list[JumperEvent]] = [[] for _ in range(num_walks)]
    for i in range(len(event_walk_ids)):
        wi = int(event_walk_ids[i])
        event = JumperEvent(
            vertex_id=int(event_vertex_ids[i]),
            step=int(event_steps[i]),
            target_block=int(event_target_blocks[i]),
            expected_arrival_step=int(event_arrival_steps[i]),
        )
        events[wi].append(event)

    log.info("Walks loaded from cache: %s (%d walks)", npz_path, num_walks)
    return WalkResult(walks=walks, events=events, walk_seeds=walk_seeds)


def generate_or_load_walks(
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> tuple[WalkResult, WalkResult]:
    """Generate or load cached train and eval walk corpora.

    Tries to load both splits from cache. On any miss, generates both
    splits fresh and saves to cache.

    Args:
        graph_data: DCSBM graph from Phase 2.
        jumpers: Jumper designations from Phase 2.
        config: Experiment configuration.
        cache_dir: Root cache directory.

    Returns:
        Tuple of (train_result, eval_result).
    """
    train_seed = config.seed + TRAIN_SEED_OFFSET
    eval_seed = config.seed + EVAL_SEED_OFFSET

    # Try loading both from cache
    train_cached = load_walks(config, "train", train_seed, cache_dir)
    if train_cached is not None:
        eval_cached = load_walks(config, "eval", eval_seed, cache_dir)
        if eval_cached is not None:
            log.info("Walk cache hit: both train and eval loaded from cache")
            return train_cached, eval_cached
        log.info("Walk cache partial hit: train cached, eval missing")

    # Cache miss: generate both
    log.info("Walk cache miss: generating corpus...")
    train_result, eval_result = generate_corpus(graph_data, jumpers, config)

    # Save both to cache
    save_walks(train_result, config, "train", train_seed, cache_dir)
    save_walks(eval_result, config, "eval", eval_seed, cache_dir)

    return train_result, eval_result
