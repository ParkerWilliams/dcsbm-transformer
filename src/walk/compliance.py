"""Path-count precomputation, viable-path precomputation, and guided step logic.

Provides two precomputation strategies for jumper compliance:
1. Path-count vectors (precompute_path_counts) for weighted neighbor selection
2. Viable-path pools (precompute_viable_paths) for guaranteed path splicing

The viable-path approach pre-computes actual r-step walks from each jumper
vertex to its target block, guaranteeing compliance by construction and
eliminating discard/retry logic.
"""

import logging
import statistics

import numpy as np
from scipy.sparse import csr_matrix

from src.graph.jumpers import JumperInfo

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


def precompute_viable_paths(
    adj: csr_matrix,
    block_assignments: np.ndarray,
    jumpers: list[JumperInfo],
    rng: np.random.Generator,
    n_paths_per_jumper: int = 200,
    max_attempts_factor: int = 20,
) -> dict[int, list[np.ndarray]]:
    """Pre-compute pools of viable r-step walks from each jumper to its target block.

    For each jumper, generates random walks of length r from the jumper vertex
    and keeps those that end in the target block. This guarantees that walk
    generation can splice in a compliant path whenever a jumper is encountered,
    eliminating probabilistic failures and discard/retry logic.

    Args:
        adj: Directed adjacency CSR matrix (n x n).
        block_assignments: Integer array of length n, vertex -> block mapping.
        jumpers: List of JumperInfo descriptors.
        rng: NumPy random Generator for reproducible path sampling.
        n_paths_per_jumper: Target number of viable paths per jumper.
        max_attempts_factor: Multiplier on n_paths_per_jumper for max attempts.

    Returns:
        Dict mapping vertex_id -> list of np.ndarray, where each array has
        shape (r+1,) dtype int32, starting at the jumper vertex and ending
        at a vertex in the target block.

    Raises:
        ValueError: If zero viable paths are found for any jumper (indicates
            incorrect jumper designation).
    """
    indptr = adj.indptr
    indices = adj.indices

    viable_paths: dict[int, list[np.ndarray]] = {}
    path_counts_per_jumper: list[int] = []

    for jumper in jumpers:
        v = jumper.vertex_id
        r = jumper.r
        tb = jumper.target_block
        max_attempts = n_paths_per_jumper * max_attempts_factor
        paths: list[np.ndarray] = []

        for _ in range(max_attempts):
            # Generate a random walk of length r from v
            walk = np.zeros(r + 1, dtype=np.int32)
            walk[0] = v
            valid = True

            for step in range(1, r + 1):
                current = walk[step - 1]
                start_idx = indptr[current]
                end_idx = indptr[current + 1]
                degree = end_idx - start_idx
                if degree == 0:
                    valid = False
                    break
                offset = int(rng.integers(0, degree))
                walk[step] = indices[start_idx + offset]

            if not valid:
                continue

            # Check if walk ends in target block
            if block_assignments[walk[-1]] == tb:
                paths.append(walk)
                if len(paths) >= n_paths_per_jumper:
                    break

        if len(paths) == 0:
            raise ValueError(
                f"Zero viable paths found for jumper vertex {v} "
                f"(target_block={tb}, r={r}) after {max_attempts} attempts. "
                f"Jumper designation may be incorrect."
            )

        if len(paths) < n_paths_per_jumper:
            log.warning(
                "Jumper vertex %d: found only %d/%d viable paths "
                "(target_block=%d, r=%d)",
                v,
                len(paths),
                n_paths_per_jumper,
                tb,
                r,
            )

        viable_paths[v] = paths
        path_counts_per_jumper.append(len(paths))

    if path_counts_per_jumper:
        log.info(
            "Precomputed viable paths for %d jumpers: "
            "min=%d, max=%d, median=%d paths per jumper",
            len(jumpers),
            min(path_counts_per_jumper),
            max(path_counts_per_jumper),
            int(statistics.median(path_counts_per_jumper)),
        )

    return viable_paths


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
