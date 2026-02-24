"""Corpus assembly with train/eval splitting and validation.

Produces independent train and eval walk corpora with different seeds,
validates corpus size against the 100n threshold, and checks jumper
fraction and path diversity.
"""

import logging

import numpy as np

from src.config.experiment import ExperimentConfig
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.walk.generator import generate_walks
from src.walk.types import JumperEvent, WalkResult

log = logging.getLogger(__name__)

# Seed offsets for train/eval to avoid correlation with graph and jumper seeds
TRAIN_SEED_OFFSET = 2000
EVAL_SEED_OFFSET = 3000


def validate_corpus(
    walks: np.ndarray,
    events: list[list[JumperEvent]],
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    min_jumper_fraction: float = 0.5,
    min_paths_per_jumper: int = 3,
) -> list[str]:
    """Validate a walk corpus for correctness and quality.

    Returns a list of error strings. Empty list means the corpus is valid.

    Checks:
    - Jumper fraction: at least min_jumper_fraction of walks contain events
    - Path diversity: each jumper has at least min_paths_per_jumper distinct
      compliant paths in the corpus
    - Edge validity: sample-based check (100 random walks)
    - Rule compliance: every jumper event results in correct block at arrival

    Args:
        walks: Walk array of shape (num_walks, walk_length).
        events: Per-walk event lists.
        graph_data: Graph data for validation.
        jumpers: List of JumperInfo.
        min_jumper_fraction: Minimum fraction of walks with jumper events.
        min_paths_per_jumper: Minimum distinct paths per jumper vertex.

    Returns:
        List of error strings (empty if valid).
    """
    errors: list[str] = []
    n_walks, walk_length = walks.shape
    indptr = graph_data.adjacency.indptr
    indices = graph_data.adjacency.indices

    # 1. Jumper fraction
    n_with_jumpers = sum(1 for e in events if len(e) > 0)
    actual_fraction = n_with_jumpers / max(1, n_walks)
    if actual_fraction < min_jumper_fraction:
        errors.append(
            f"Jumper fraction {actual_fraction:.3f} below minimum "
            f"{min_jumper_fraction:.3f} ({n_with_jumpers}/{n_walks} walks)"
        )

    # 2. Path diversity per jumper
    jumper_set = {j.vertex_id for j in jumpers}
    jumper_paths: dict[int, set[tuple[int, ...]]] = {
        v: set() for v in jumper_set
    }
    for wi, walk_events in enumerate(events):
        for event in walk_events:
            if event.expected_arrival_step < walk_length:
                segment = tuple(
                    int(walks[wi, s])
                    for s in range(event.step, event.expected_arrival_step + 1)
                )
                jumper_paths[event.vertex_id].add(segment)

    for v in jumper_set:
        n_paths = len(jumper_paths.get(v, set()))
        if 0 < n_paths < min_paths_per_jumper:
            errors.append(
                f"Jumper vertex {v} has only {n_paths} distinct paths "
                f"(need >= {min_paths_per_jumper})"
            )

    # 3. Edge validity (sample-based, 100 random walks)
    n_check = min(100, n_walks)
    check_rng = np.random.default_rng(0)
    check_indices = check_rng.choice(n_walks, size=n_check, replace=False)
    for wi in check_indices:
        for step in range(walk_length - 1):
            u = walks[wi, step]
            v = walks[wi, step + 1]
            neighbors = indices[indptr[u]:indptr[u + 1]]
            if v not in neighbors:
                errors.append(
                    f"Walk {wi} step {step}: edge {u}->{v} not in graph"
                )
                break  # one error per walk is enough

    # 4. Rule compliance for all events
    for wi, walk_events in enumerate(events):
        for event in walk_events:
            if event.expected_arrival_step < walk_length:
                actual_block = int(
                    graph_data.block_assignments[
                        walks[wi, event.expected_arrival_step]
                    ]
                )
                if actual_block != event.target_block:
                    errors.append(
                        f"Walk {wi}: jumper at step {event.step} expected "
                        f"block {event.target_block} at step "
                        f"{event.expected_arrival_step}, got {actual_block}"
                    )

    return errors


def generate_corpus(
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
) -> tuple[WalkResult, WalkResult]:
    """Generate independent train and eval walk corpora.

    Train and eval sets use completely different seeds to ensure
    independence. The 100n threshold is validated for the training set.

    Args:
        graph_data: DCSBM graph from Phase 2.
        jumpers: Jumper designations from Phase 2.
        config: Experiment configuration.

    Returns:
        Tuple of (train_result, eval_result).

    Raises:
        ValueError: If corpus validation fails or size is insufficient.
    """
    walk_length = config.training.walk_length
    n_train = config.training.corpus_size
    n_eval = max(1, n_train // 9)  # 90/10 split: train:eval = 9:1

    train_seed = config.seed + TRAIN_SEED_OFFSET
    eval_seed = config.seed + EVAL_SEED_OFFSET

    log.info(
        "Generating corpus: %d train + %d eval walks, length %d",
        n_train,
        n_eval,
        walk_length,
    )

    # Validate training corpus size against 100n threshold
    min_required = 100 * config.graph.n
    if n_train < min_required:
        raise ValueError(
            f"Training corpus size {n_train} is less than 100 * n "
            f"({min_required}). Increase corpus_size."
        )

    # Generate train walks
    log.info("Generating training corpus (seed=%d)...", train_seed)
    train_result = generate_walks(
        graph_data,
        jumpers,
        config,
        seed=train_seed,
        target_n_walks=n_train,
    )

    # Generate eval walks
    log.info("Generating evaluation corpus (seed=%d)...", eval_seed)
    eval_result = generate_walks(
        graph_data,
        jumpers,
        config,
        seed=eval_seed,
        target_n_walks=n_eval,
    )

    # Validate train corpus
    train_errors = validate_corpus(
        train_result.walks, train_result.events, graph_data, jumpers,
    )
    if train_errors:
        raise ValueError(
            f"Training corpus validation failed:\n"
            + "\n".join(f"  - {e}" for e in train_errors)
        )

    # Validate eval corpus (no 100n threshold, but check quality)
    eval_errors = validate_corpus(
        eval_result.walks, eval_result.events, graph_data, jumpers,
    )
    if eval_errors:
        log.warning(
            "Evaluation corpus has %d validation warnings:\n%s",
            len(eval_errors),
            "\n".join(f"  - {e}" for e in eval_errors),
        )

    # Log corpus statistics
    train_events_total = sum(len(e) for e in train_result.events)
    eval_events_total = sum(len(e) for e in eval_result.events)
    train_jumper_walks = sum(1 for e in train_result.events if len(e) > 0)
    eval_jumper_walks = sum(1 for e in eval_result.events if len(e) > 0)

    log.info(
        "Corpus complete: train=%d walks (%d jumper, %.1f%%), "
        "eval=%d walks (%d jumper, %.1f%%), "
        "total events: train=%d eval=%d",
        n_train,
        train_jumper_walks,
        100.0 * train_jumper_walks / max(1, n_train),
        n_eval,
        eval_jumper_walks,
        100.0 * eval_jumper_walks / max(1, n_eval),
        train_events_total,
        eval_events_total,
    )

    return train_result, eval_result
