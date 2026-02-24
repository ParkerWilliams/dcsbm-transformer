"""Path-count precomputation and guided step logic for jumper compliance.

Implements the path-count guided walking algorithm: precompute vectors
counting (normalized) paths from each vertex to each target block at
each step distance, then use these to weight neighbor selection during
guided walk segments.
"""

import logging

import numpy as np
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)


def precompute_path_counts(
    adj: csr_matrix,
    block_assignments: np.ndarray,
    K: int,
    max_r: int,
) -> dict[int, list[np.ndarray]]:
    """Precompute path-count vectors for all target blocks up to max_r steps.

    For each target block tb, computes a sequence of vectors N[0..max_r] where
    N[k][v] represents the (normalized) number of paths of length k from
    vertex v to any vertex in block tb. Normalization by max value at each
    step prevents overflow while preserving ratios for weighted sampling.

    Uses adj (forward edges) directly: adj @ N[k-1] propagates path counts
    forward -- for each vertex u, sums N[k-1][v] over all neighbors v,
    counting paths of length k starting from u reaching target block.

    Args:
        adj: Directed adjacency CSR matrix (n x n), adj[u][v]=1 means edge u->v.
        block_assignments: Integer array of length n, vertex -> block mapping.
        K: Number of blocks.
        max_r: Maximum jump length to precompute for.

    Returns:
        Dict mapping target_block -> list of (max_r+1) path-count vectors.
        path_counts[tb][k][v] = normalized count of paths from v to block tb
        in k steps.
    """
    n = adj.shape[0]
    path_counts: dict[int, list[np.ndarray]] = {}

    for tb in range(K):
        # N[0] = indicator vector: 1.0 for vertices in target block, 0 otherwise
        target_mask = (block_assignments == tb).astype(np.float64)
        N_all: list[np.ndarray] = [target_mask.copy()]
        N_prev = target_mask.copy()

        for k in range(1, max_r + 1):
            # Sparse matrix-vector multiply: propagate path counts forward
            N_curr = np.asarray(adj @ N_prev).ravel()
            mx = N_curr.max()
            if mx > 0:
                N_curr = N_curr / mx  # normalize to prevent overflow
            N_all.append(N_curr.copy())
            N_prev = N_curr

        path_counts[tb] = N_all

    log.debug(
        "Precomputed path counts for %d target blocks up to max_r=%d",
        K,
        max_r,
    )
    return path_counts


def guided_step(
    vertex: int,
    active_constraints: list[tuple[int, int]],
    step: int,
    path_counts: dict[int, list[np.ndarray]],
    indptr: np.ndarray,
    indices: np.ndarray,
    rng: np.random.Generator,
) -> int | None:
    """Select next vertex during a guided walk segment.

    Weights each neighbor by the product of path-count values across
    all active constraints, ensuring uniform sampling over valid
    compliant paths. Returns None if no neighbor satisfies all
    constraints simultaneously (infeasible joint constraint).

    Args:
        vertex: Current vertex in the walk.
        active_constraints: List of (deadline_step, target_block) tuples
            representing active jumper constraints.
        step: Current step index in the walk (we are choosing the vertex
            for step+1).
        path_counts: Precomputed path-count vectors from
            precompute_path_counts().
        indptr: CSR row pointer array from adjacency matrix.
        indices: CSR column indices array from adjacency matrix.
        rng: NumPy random Generator for reproducible sampling.

    Returns:
        Selected neighbor vertex index, or None if constraints are
        infeasible (all weights are zero).
    """
    # Get neighbors of vertex from CSR arrays
    start = indptr[vertex]
    end = indptr[vertex + 1]
    neighbors = indices[start:end]

    if len(neighbors) == 0:
        return None  # dead-end vertex (shouldn't happen with connected graph)

    # Initialize uniform weights
    weights = np.ones(len(neighbors), dtype=np.float64)

    for deadline, target_block in active_constraints:
        remaining = deadline - step
        if remaining <= 0:
            continue  # past deadline, skip

        # remaining-1 because choosing NEXT vertex leaves remaining-1 more
        # steps to reach the target block (per research pitfall 2)
        pc_index = remaining - 1
        if pc_index < 0 or pc_index >= len(path_counts[target_block]):
            continue  # out of precomputed range

        # Look up path counts for each neighbor
        neighbor_weights = path_counts[target_block][pc_index][neighbors]
        weights *= neighbor_weights

    total = weights.sum()
    if total == 0.0:
        return None  # infeasible joint constraint

    return int(rng.choice(neighbors, p=weights / total))
