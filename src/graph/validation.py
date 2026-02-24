"""Path-based validation for block jumper non-triviality.

Verifies that jumper vertices have valid (non-trivial) paths of length r
to their target blocks. A non-trivial assignment means:
1. Paths of length r from the jumper reach the target block (reachability)
2. Paths of length r also reach non-target blocks (non-triviality)
"""

import logging

import numpy as np
import scipy.sparse

log = logging.getLogger(__name__)


def reachable_blocks_at_distance(
    adj: scipy.sparse.csr_matrix,
    vertex: int,
    r: int,
    block_assignments: np.ndarray,
    K: int,
) -> set[int]:
    """Compute which blocks are reachable from vertex in exactly r steps.

    Uses iterative sparse vector-matrix multiplication with binary clipping
    to prevent integer overflow from path counting.

    Args:
        adj: Sparse directed adjacency matrix (n x n).
        vertex: Source vertex index.
        r: Number of steps (path length).
        block_assignments: Array mapping vertex index to block number.
        K: Total number of blocks.

    Returns:
        Set of block indices reachable in exactly r steps.
    """
    n = adj.shape[0]

    # Start with indicator vector for the source vertex
    vec = scipy.sparse.csr_matrix(
        ([1.0], ([0], [vertex])), shape=(1, n)
    )

    for _ in range(r):
        vec = vec @ adj
        # Clip to binary reachability to prevent exponential overflow
        if vec.nnz > 0:
            vec.data[:] = np.minimum(vec.data, 1.0)

    # Extract reachable vertices
    reachable_vertices = vec.nonzero()[1]

    if len(reachable_vertices) == 0:
        return set()

    # Map to blocks
    return set(block_assignments[reachable_vertices].tolist())


def check_non_trivial(
    adj: scipy.sparse.csr_matrix,
    vertex: int,
    target_block: int,
    r: int,
    block_assignments: np.ndarray,
    K: int,
) -> bool:
    """Check that a jumper assignment is non-trivial.

    Non-trivial means:
    1. Target block is reachable in r steps (paths to target exist)
    2. At least one non-target block is also reachable (paths are not unique)

    Args:
        adj: Sparse directed adjacency matrix.
        vertex: Jumper vertex index.
        target_block: Required destination block.
        r: Jump length (number of steps).
        block_assignments: Vertex-to-block mapping.
        K: Total number of blocks.

    Returns:
        True if the assignment is non-trivial.
    """
    reachable_blocks = reachable_blocks_at_distance(
        adj, vertex, r, block_assignments, K
    )

    reaches_target = target_block in reachable_blocks
    reaches_non_target = bool(reachable_blocks - {target_block})

    log.debug(
        "Vertex %d, target_block=%d, r=%d: reachable_blocks=%s, "
        "reaches_target=%s, reaches_non_target=%s",
        vertex,
        target_block,
        r,
        reachable_blocks,
        reaches_target,
        reaches_non_target,
    )

    return reaches_target and reaches_non_target


def verify_all_jumpers(
    adj: scipy.sparse.csr_matrix,
    jumpers: list,
    block_assignments: np.ndarray,
    K: int,
) -> list[tuple[int, str]]:
    """Batch-verify non-triviality for all jumper assignments.

    Args:
        adj: Sparse directed adjacency matrix.
        jumpers: List of JumperInfo objects.
        block_assignments: Vertex-to-block mapping.
        K: Total number of blocks.

    Returns:
        List of (jumper_index, failure_reason) for failed jumpers.
        Empty list means all jumpers pass.
    """
    failures: list[tuple[int, str]] = []

    for idx, jumper in enumerate(jumpers):
        reachable = reachable_blocks_at_distance(
            adj, jumper.vertex_id, jumper.r, block_assignments, K
        )

        if jumper.target_block not in reachable:
            failures.append(
                (idx, f"No paths to target block {jumper.target_block} "
                 f"from vertex {jumper.vertex_id} at distance {jumper.r}")
            )
        elif not (reachable - {jumper.target_block}):
            failures.append(
                (idx, f"Only target block {jumper.target_block} reachable "
                 f"from vertex {jumper.vertex_id} at distance {jumper.r} "
                 f"(trivial assignment)")
            )

    return failures
