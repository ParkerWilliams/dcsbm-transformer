"""Core walk generation with guided and batch modes.

Implements two walk generation strategies:
1. Per-walk guided generation using path-count weights for jumper compliance
2. Vectorized batch generation for unguided walks (no active constraints)

The top-level generate_walks() function orchestrates both strategies to
produce a walk corpus with >= min_jumper_fraction containing jumper events.
"""

import logging
import math

import numpy as np

from src.config.experiment import ExperimentConfig
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.walk.compliance import guided_step, precompute_path_counts
from src.walk.types import JumperEvent, WalkResult

log = logging.getLogger(__name__)


def generate_single_guided_walk(
    start_vertex: int,
    walk_length: int,
    rng: np.random.Generator,
    graph_data: GraphData,
    jumper_map: dict[int, JumperInfo],
    path_counts: dict[int, list[np.ndarray]],
    indptr: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, list[JumperEvent]] | None:
    """Generate a single walk with guided segments at jumper encounters.

    When the walk visits a jumper vertex, enters a guided segment where
    neighbor selection is weighted by path-count vectors to ensure the
    walk reaches the target block at exactly the right step.

    Args:
        start_vertex: Starting vertex for the walk.
        walk_length: Total length of the walk.
        rng: Per-walk random Generator for reproducibility.
        graph_data: Graph data with adjacency matrix and block assignments.
        jumper_map: Dict mapping vertex_id -> JumperInfo.
        path_counts: Precomputed path-count vectors.
        indptr: CSR row pointer array.
        indices: CSR column indices array.

    Returns:
        Tuple of (walk array, event list) or None if walk is infeasible
        due to joint constraint conflicts.
    """
    walk = np.zeros(walk_length, dtype=np.int32)
    walk[0] = start_vertex
    events: list[JumperEvent] = []
    active_constraints: list[tuple[int, int]] = []  # (deadline_step, target_block)

    for step in range(1, walk_length):
        prev = int(walk[step - 1])

        # Check if previous vertex is a jumper
        if prev in jumper_map:
            jumper = jumper_map[prev]
            event = JumperEvent(
                vertex_id=int(prev),
                step=int(step - 1),
                target_block=int(jumper.target_block),
                expected_arrival_step=int(step - 1 + jumper.r),
            )
            events.append(event)
            active_constraints.append(
                (step - 1 + jumper.r, jumper.target_block)
            )

        # Remove expired constraints (deadline <= step means we're past it)
        active_constraints = [
            (d, tb) for d, tb in active_constraints if d > step
        ]

        if active_constraints:
            # Guided step: weight neighbors by path-count constraints
            next_v = guided_step(
                prev,
                active_constraints,
                step,
                path_counts,
                indptr,
                indices,
                rng,
            )
            if next_v is None:
                return None  # infeasible, caller discards
            walk[step] = next_v
        else:
            # Unguided step: uniform random neighbor
            start_idx = indptr[prev]
            end_idx = indptr[prev + 1]
            degree = end_idx - start_idx
            if degree == 0:
                return None  # dead-end (shouldn't happen with connected graph)
            offset = int(rng.integers(0, degree))
            walk[step] = indices[start_idx + offset]

    # Validate compliance: for each event, check block at arrival step
    for event in events:
        if event.expected_arrival_step < walk_length:
            actual_block = graph_data.block_assignments[
                walk[event.expected_arrival_step]
            ]
            if actual_block != event.target_block:
                log.warning(
                    "Compliance violation: vertex %d at step %d expected "
                    "block %d at step %d, got block %d",
                    event.vertex_id,
                    event.step,
                    event.target_block,
                    event.expected_arrival_step,
                    actual_block,
                )
                return None  # treat as infeasible

    return walk, events


def generate_batch_unguided_walks(
    start_vertices: np.ndarray,
    walk_length: int,
    rng: np.random.Generator,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Generate multiple walks simultaneously using vectorized operations.

    Processes one step at a time across all walks using NumPy array
    operations on CSR index arrays. No jumper guidance -- pure uniform
    random neighbor selection.

    Args:
        start_vertices: Array of starting vertex indices.
        walk_length: Total length of each walk.
        rng: Random Generator for reproducibility.
        indptr: CSR row pointer array.
        indices: CSR column indices array.

    Returns:
        Array of shape (n_walks, walk_length) with dtype int32.
    """
    n_walks = len(start_vertices)
    walks = np.zeros((n_walks, walk_length), dtype=np.int32)
    walks[:, 0] = start_vertices

    for step in range(1, walk_length):
        current = walks[:, step - 1]
        starts = indptr[current]
        ends = indptr[current + 1]
        degrees = ends - starts
        offsets = (rng.random(n_walks) * degrees).astype(np.int64)
        offsets = np.clip(offsets, 0, degrees - 1)
        walks[:, step] = indices[starts + offsets]

    return walks


def _walk_contains_jumper(walk: np.ndarray, jumper_set: set[int]) -> bool:
    """Check if any vertex in the walk is a jumper vertex."""
    for v in walk:
        if int(v) in jumper_set:
            return True
    return False


def generate_walks(
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    seed: int,
    target_n_walks: int,
    min_jumper_fraction: float = 0.5,
) -> WalkResult:
    """Generate a walk corpus with guided jumper compliance.

    Two-phase strategy:
    1. Jumper-seeded walks (guided): Start from jumper vertices, cycle
       through jumpers, use path-count guided walking for compliance.
    2. Random-start walks (batch): Start from random vertices, generate
       batch unguided. Re-run individual walks as guided if they
       encounter a jumper vertex.

    Args:
        graph_data: DCSBM graph from Phase 2.
        jumpers: List of JumperInfo from Phase 2 jumper designation.
        config: Experiment configuration.
        seed: Master seed for walk generation.
        target_n_walks: Target number of walks to generate.
        min_jumper_fraction: Minimum fraction of walks containing jumper events.

    Returns:
        WalkResult with walks, events, and per-walk seeds.
    """
    walk_length = config.training.walk_length

    # Build jumper lookup structures
    jumper_map = {j.vertex_id: j for j in jumpers}
    jumper_set = set(jumper_map.keys())
    jumper_vertices = list(jumper_map.keys())

    if not jumper_vertices:
        log.warning("No jumpers provided, generating all unguided walks")

    # Extract CSR arrays
    indptr = graph_data.adjacency.indptr
    indices = graph_data.adjacency.indices

    # Precompute path counts
    max_r = max((j.r for j in jumpers), default=0)
    if max_r > 0:
        path_counts = precompute_path_counts(
            graph_data.adjacency,
            graph_data.block_assignments,
            graph_data.K,
            max_r,
        )
    else:
        path_counts = {}

    # Determine walk counts
    n_jumper_walks = math.ceil(target_n_walks * min_jumper_fraction)
    n_random_walks = target_n_walks - n_jumper_walks

    # Master RNG for per-walk seed generation
    master_rng = np.random.default_rng(seed)

    # Generate per-walk seeds for ALL walks (jumper + random)
    # Over-generate by ~5% to account for discards
    overgen_factor = 1.05
    total_seeds_needed = math.ceil(
        (n_jumper_walks + n_random_walks) * overgen_factor
    )
    all_walk_seeds = master_rng.integers(0, 2**63, size=total_seeds_needed)

    seed_idx = 0
    collected_walks: list[np.ndarray] = []
    collected_events: list[list[JumperEvent]] = []
    collected_seeds: list[int] = []

    # Phase 1: Jumper-seeded guided walks
    log.info(
        "Generating %d jumper-seeded guided walks...", n_jumper_walks
    )
    jumper_walk_count = 0
    discarded_count = 0

    while jumper_walk_count < n_jumper_walks:
        if seed_idx >= len(all_walk_seeds):
            # Need more seeds
            extra = master_rng.integers(
                0, 2**63, size=max(100, n_jumper_walks - jumper_walk_count)
            )
            all_walk_seeds = np.concatenate([all_walk_seeds, extra])

        walk_seed = int(all_walk_seeds[seed_idx])
        seed_idx += 1
        walk_rng = np.random.default_rng(walk_seed)

        # Cycle through jumper vertices
        start_vertex = jumper_vertices[
            jumper_walk_count % len(jumper_vertices)
        ]

        result = generate_single_guided_walk(
            start_vertex,
            walk_length,
            walk_rng,
            graph_data,
            jumper_map,
            path_counts,
            indptr,
            indices,
        )

        if result is None:
            discarded_count += 1
            continue

        walk, events = result
        collected_walks.append(walk)
        collected_events.append(events)
        collected_seeds.append(walk_seed)
        jumper_walk_count += 1

    log.info(
        "Jumper walks: %d generated, %d discarded (%.1f%%)",
        jumper_walk_count,
        discarded_count,
        100.0 * discarded_count / max(1, jumper_walk_count + discarded_count),
    )

    # Phase 2: Random-start walks (batch unguided, then check for jumpers)
    if n_random_walks > 0:
        log.info("Generating %d random-start walks...", n_random_walks)

        # Generate seeds for random walks
        random_walk_seeds_needed = n_random_walks
        if seed_idx + random_walk_seeds_needed > len(all_walk_seeds):
            extra = master_rng.integers(
                0, 2**63, size=random_walk_seeds_needed
            )
            all_walk_seeds = np.concatenate([all_walk_seeds, extra])

        # Use a batch RNG for the batch generation
        batch_seed = int(all_walk_seeds[seed_idx])
        seed_idx += 1
        batch_rng = np.random.default_rng(batch_seed)

        # Random start vertices
        start_vertices = batch_rng.integers(
            0, graph_data.n, size=n_random_walks, dtype=np.int32
        )

        # Generate batch unguided walks
        batch_walks = generate_batch_unguided_walks(
            start_vertices, walk_length, batch_rng, indptr, indices
        )

        # Check each walk for jumper encounters and re-run as guided if needed
        random_rerun_count = 0
        random_discard_count = 0

        for i in range(n_random_walks):
            walk = batch_walks[i]

            if seed_idx >= len(all_walk_seeds):
                extra = master_rng.integers(0, 2**63, size=100)
                all_walk_seeds = np.concatenate([all_walk_seeds, extra])

            walk_seed = int(all_walk_seeds[seed_idx])
            seed_idx += 1

            if _walk_contains_jumper(walk, jumper_set):
                # Re-generate as guided walk
                walk_rng = np.random.default_rng(walk_seed)
                start_v = int(start_vertices[i])
                result = generate_single_guided_walk(
                    start_v,
                    walk_length,
                    walk_rng,
                    graph_data,
                    jumper_map,
                    path_counts,
                    indptr,
                    indices,
                )

                if result is None:
                    random_discard_count += 1
                    # Generate a replacement walk
                    while True:
                        if seed_idx >= len(all_walk_seeds):
                            extra = master_rng.integers(0, 2**63, size=100)
                            all_walk_seeds = np.concatenate(
                                [all_walk_seeds, extra]
                            )
                        retry_seed = int(all_walk_seeds[seed_idx])
                        seed_idx += 1
                        retry_rng = np.random.default_rng(retry_seed)
                        new_start = int(retry_rng.integers(0, graph_data.n))
                        retry_result = generate_single_guided_walk(
                            new_start,
                            walk_length,
                            retry_rng,
                            graph_data,
                            jumper_map,
                            path_counts,
                            indptr,
                            indices,
                        )
                        if retry_result is not None:
                            walk, events = retry_result
                            walk_seed = retry_seed
                            break
                    collected_walks.append(walk)
                    collected_events.append(events)
                    collected_seeds.append(walk_seed)
                else:
                    walk, events = result
                    collected_walks.append(walk)
                    collected_events.append(events)
                    collected_seeds.append(walk_seed)
                random_rerun_count += 1
            else:
                # Jumper-free walk: keep as-is
                collected_walks.append(walk.copy())
                collected_events.append([])
                collected_seeds.append(walk_seed)

        log.info(
            "Random walks: %d generated, %d re-run as guided, %d discarded",
            n_random_walks,
            random_rerun_count,
            random_discard_count,
        )

    # Assemble result
    walks_array = np.stack(collected_walks, axis=0)
    seeds_array = np.array(collected_seeds, dtype=np.int64)

    # Validation
    _validate_walks(
        walks_array,
        collected_events,
        graph_data,
        jumper_map,
        jumper_set,
        min_jumper_fraction,
    )

    log.info(
        "Walk generation complete: %d walks, length %d, %d total events",
        walks_array.shape[0],
        walks_array.shape[1],
        sum(len(e) for e in collected_events),
    )

    return WalkResult(
        walks=walks_array,
        events=collected_events,
        walk_seeds=seeds_array,
    )


def _validate_walks(
    walks: np.ndarray,
    events: list[list[JumperEvent]],
    graph_data: GraphData,
    jumper_map: dict[int, JumperInfo],
    jumper_set: set[int],
    min_jumper_fraction: float,
) -> None:
    """Validate walks for edge validity, compliance, and jumper fraction.

    Args:
        walks: Walk array of shape (n_walks, walk_length).
        events: Per-walk event lists.
        graph_data: Graph data for validation.
        jumper_map: Jumper vertex lookup.
        jumper_set: Set of jumper vertex IDs.
        min_jumper_fraction: Minimum fraction of walks with jumper events.

    Raises:
        ValueError: If validation fails.
    """
    n_walks, walk_length = walks.shape
    indptr = graph_data.adjacency.indptr
    indices = graph_data.adjacency.indices

    # 1. Edge validity: sample-check 100 walks
    n_check = min(100, n_walks)
    check_indices = np.random.default_rng(0).choice(
        n_walks, size=n_check, replace=False
    )
    for wi in check_indices:
        for step in range(walk_length - 1):
            u = walks[wi, step]
            v = walks[wi, step + 1]
            neighbors = indices[indptr[u]:indptr[u + 1]]
            if v not in neighbors:
                raise ValueError(
                    f"Walk {wi} step {step}: edge {u}->{v} not in graph"
                )

    # 2. Compliance: check all events
    for wi, walk_events in enumerate(events):
        for event in walk_events:
            if event.expected_arrival_step < walk_length:
                actual_block = graph_data.block_assignments[
                    walks[wi, event.expected_arrival_step]
                ]
                if actual_block != event.target_block:
                    raise ValueError(
                        f"Walk {wi}: jumper at step {event.step} expected "
                        f"block {event.target_block} at step "
                        f"{event.expected_arrival_step}, got {actual_block}"
                    )

    # 3. Jumper fraction
    n_with_jumpers = sum(1 for e in events if len(e) > 0)
    actual_fraction = n_with_jumpers / n_walks
    if actual_fraction < min_jumper_fraction:
        log.warning(
            "Jumper fraction %.1f%% below minimum %.1f%%",
            100.0 * actual_fraction,
            100.0 * min_jumper_fraction,
        )

    # 4. Path diversity: check each jumper has at least 3 distinct paths
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

    for v, paths in jumper_paths.items():
        if len(paths) < 3 and len(paths) > 0:
            log.warning(
                "Jumper vertex %d has only %d distinct paths (need >= 3)",
                v,
                len(paths),
            )
