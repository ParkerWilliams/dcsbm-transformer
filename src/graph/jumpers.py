"""Block jumper designation with variable r values and non-triviality verification.

Each graph contains jumpers at ALL r values from the discrete set
{0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w}, distributed uniformly
across jumpers. This replaces the config-level r sweep entirely.
"""

import logging
from dataclasses import dataclass

import numpy as np

from src.config.experiment import ExperimentConfig
from src.graph.types import GraphData
from src.graph.validation import check_non_trivial

log = logging.getLogger(__name__)

# Fixed set of r scale factors per CONTEXT.md locked decision
R_SCALES: tuple[float, ...] = (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0)


@dataclass(frozen=True, slots=True)
class JumperInfo:
    """Immutable descriptor for a block jumper vertex.

    A jumper in block `source_block` must have walks reaching
    `target_block` after exactly `r` steps.
    """

    vertex_id: int
    source_block: int
    target_block: int
    r: int  # jump length in steps (rounded from scale * w)


def compute_r_values(w: int) -> list[int]:
    """Compute the discrete set of r values from context window size.

    Applies each scale in R_SCALES to w, rounds to nearest integer,
    deduplicates, and returns sorted.

    Args:
        w: Context window size.

    Returns:
        Sorted list of unique r values (minimum 1).
    """
    raw = [max(1, round(scale * w)) for scale in R_SCALES]
    # Deduplicate while preserving order, then sort
    unique = list(dict.fromkeys(raw))
    return sorted(unique)


def designate_jumpers(
    graph_data: GraphData,
    config: ExperimentConfig,
    rng: np.random.Generator,
    max_reassign_per_vertex: int | None = None,
) -> list[JumperInfo]:
    """Designate block jumper vertices with variable r values.

    For each block, selects n_jumpers_per_block vertices as jumpers,
    assigns r values from compute_r_values(w) cycling through the set,
    and validates non-triviality for each assignment.

    Args:
        graph_data: Generated DCSBM graph.
        config: Experiment configuration.
        rng: numpy random Generator for reproducibility.
        max_reassign_per_vertex: Max reassignment attempts per failed vertex.
            Defaults to block_size // 2.

    Returns:
        List of JumperInfo sorted by (source_block, vertex_id).
    """
    w = config.training.w
    n_jumpers_per_block = config.graph.n_jumpers_per_block
    n = graph_data.n
    K = graph_data.K
    block_size = graph_data.block_size

    if max_reassign_per_vertex is None:
        max_reassign_per_vertex = block_size // 2

    r_values = compute_r_values(w)
    log.info(
        "Designating jumpers: %d per block, r_values=%s",
        n_jumpers_per_block,
        r_values,
    )

    jumpers: list[JumperInfo] = []
    used_vertices: set[int] = set()
    global_jumper_idx = 0  # Global counter for r-value cycling across all blocks

    for b in range(K):
        # Get all vertices in this block
        block_vertices = [
            v for v in range(n) if graph_data.block_assignments[v] == b
        ]

        # Select n_jumpers_per_block vertices randomly (without replacement)
        selected_indices = rng.choice(
            len(block_vertices), size=n_jumpers_per_block, replace=False
        )
        selected_vertices = [block_vertices[i] for i in selected_indices]

        # Other blocks for target assignment
        other_blocks = [bb for bb in range(K) if bb != b]

        for vertex in selected_vertices:
            # Assign r value by cycling through r_values (global counter)
            r = r_values[global_jumper_idx % len(r_values)]
            global_jumper_idx += 1

            # Choose random target block
            target_block = rng.choice(other_blocks)

            # Check non-triviality
            is_valid = check_non_trivial(
                graph_data.adjacency,
                vertex,
                target_block,
                r,
                graph_data.block_assignments,
                K,
            )

            if is_valid:
                jumpers.append(
                    JumperInfo(
                        vertex_id=vertex,
                        source_block=b,
                        target_block=target_block,
                        r=r,
                    )
                )
                used_vertices.add(vertex)
                continue

            # Reassignment: try alternative vertices in same block
            reassigned = False
            candidates = [
                v for v in block_vertices
                if v != vertex and v not in used_vertices
            ]
            rng.shuffle(candidates)

            for attempt, candidate in enumerate(
                candidates[:max_reassign_per_vertex]
            ):
                if check_non_trivial(
                    graph_data.adjacency,
                    candidate,
                    target_block,
                    r,
                    graph_data.block_assignments,
                    K,
                ):
                    jumpers.append(
                        JumperInfo(
                            vertex_id=candidate,
                            source_block=b,
                            target_block=target_block,
                            r=r,
                        )
                    )
                    used_vertices.add(candidate)
                    reassigned = True
                    log.info(
                        "Reassigned jumper in block %d: vertex %d -> %d "
                        "(attempt %d)",
                        b,
                        vertex,
                        candidate,
                        attempt + 1,
                    )
                    break

            if not reassigned:
                log.warning(
                    "Could not find non-trivial vertex for block %d, "
                    "r=%d, target_block=%d after %d attempts. "
                    "Jumper skipped.",
                    b,
                    r,
                    target_block,
                    min(len(candidates), max_reassign_per_vertex),
                )

    # Check r-value coverage
    assigned_r = {j.r for j in jumpers}
    missing_r = set(r_values) - assigned_r
    if missing_r:
        log.warning(
            "Not all r values represented in jumpers. Missing: %s "
            "(have %d jumpers, %d r_values)",
            sorted(missing_r),
            len(jumpers),
            len(r_values),
        )

    # Sort by (source_block, vertex_id) for deterministic ordering
    jumpers.sort(key=lambda j: (j.source_block, j.vertex_id))

    log.info(
        "Designated %d jumpers across %d blocks, r_values covered: %s",
        len(jumpers),
        K,
        sorted(assigned_r),
    )

    return jumpers
