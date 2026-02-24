"""DCSBM directed graph generator with degree correction, validation, and retry.

Implements the Degree-Corrected Stochastic Block Model (Karrer & Newman 2011)
for generating directed graphs with configurable block structure and
power-law degree heterogeneity.
"""

import logging

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import connected_components

from src.config.experiment import ExperimentConfig
from src.graph.degree_correction import sample_theta
from src.graph.types import GraphData

log = logging.getLogger(__name__)


class GraphGenerationError(Exception):
    """Raised when graph generation fails after all retry attempts."""


def build_probability_matrix(
    n: int, K: int, p_in: float, p_out: float, theta: np.ndarray
) -> np.ndarray:
    """Build the DCSBM edge probability matrix.

    P[i,j] = theta[i] * theta[j] * omega[block[i], block[j]]
    where omega is p_in for same-block, p_out for different-block.

    Args:
        n: Number of vertices.
        K: Number of blocks.
        p_in: In-group edge probability.
        p_out: Out-group edge probability.
        theta: Degree correction parameters of shape (n,).

    Returns:
        Probability matrix of shape (n, n) with values in [0, 1].
    """
    block_size = n // K
    blocks = np.arange(n) // block_size

    # Build K x K omega matrix
    omega = np.full((K, K), p_out, dtype=np.float64)
    np.fill_diagonal(omega, p_in)

    # Expand to n x n block probability matrix
    block_probs = omega[blocks][:, blocks]

    # Apply degree correction: P[i,j] = theta[i] * theta[j] * omega[b_i, b_j]
    P = np.outer(theta, theta) * block_probs

    # Clip to [0, 1] (degree correction can push above 1 for high-theta vertices)
    np.clip(P, 0.0, 1.0, out=P)

    # No self-loops
    np.fill_diagonal(P, 0.0)

    return P


def sample_adjacency(
    P: np.ndarray, rng: np.random.Generator
) -> scipy.sparse.csr_matrix:
    """Sample a directed adjacency matrix from probability matrix P.

    Each potential edge (i,j) is sampled independently as Bernoulli(P[i,j]).

    Args:
        P: Edge probability matrix of shape (n, n).
        rng: numpy random Generator for reproducibility.

    Returns:
        Sparse CSR adjacency matrix.
    """
    n = P.shape[0]
    uniform = rng.random((n, n))
    edges = (uniform < P).astype(np.float64)
    np.fill_diagonal(edges, 0)  # redundant safety: no self-loops
    return scipy.sparse.csr_matrix(edges)


def validate_graph(
    adj: scipy.sparse.csr_matrix,
    P: np.ndarray,
    n: int,
    K: int,
    p_in: float,
    p_out: float,
) -> list[str]:
    """Validate a generated graph against DCSBM constraints.

    Checks (cheapest first):
    1. No self-loops
    2. Strong connectivity
    3. Minimum expected degree >= 3 (from probability matrix)
    4. Edge density within 2-sigma tolerance per block pair

    Args:
        adj: Sparse adjacency matrix.
        P: Edge probability matrix (for expected degree check).
        n: Number of vertices.
        K: Number of blocks.
        p_in: In-group edge probability.
        p_out: Out-group edge probability.

    Returns:
        List of error strings (empty = valid graph).
    """
    errors: list[str] = []
    block_size = n // K

    # 1. No self-loops
    diag_sum = adj.diagonal().sum()
    if diag_sum != 0:
        errors.append(f"Self-loops detected: diagonal sum = {diag_sum}")

    # 2. Strong connectivity
    n_components, _ = connected_components(
        adj, directed=True, connection="strong"
    )
    if n_components != 1:
        errors.append(
            f"Not strongly connected: {n_components} components found"
        )

    # 3. Minimum expected degree >= 3 (from probability matrix P)
    expected_degrees = P.sum(axis=1)
    min_expected = expected_degrees.min()
    if min_expected < 3.0:
        errors.append(
            f"Minimum expected degree {min_expected:.2f} < 3.0"
        )

    # 4. Edge density per block pair within 2-sigma tolerance
    blocks = np.arange(n) // block_size
    for a in range(K):
        mask_a = blocks == a
        for b in range(K):
            mask_b = blocks == b
            # Extract the block sub-matrix
            sub = adj[np.ix_(mask_a, mask_b)].toarray()

            if a == b:
                # Same block: exclude diagonal
                n_pairs = block_size * (block_size - 1)
                np.fill_diagonal(sub, 0)
                observed = sub.sum() / n_pairs if n_pairs > 0 else 0.0
            else:
                # Different blocks
                n_pairs = block_size * block_size
                observed = sub.sum() / n_pairs if n_pairs > 0 else 0.0

            # Expected density from probability matrix
            P_sub = P[np.ix_(mask_a, mask_b)]
            if a == b:
                # Zero diagonal for expected too
                P_sub_copy = P_sub.copy()
                np.fill_diagonal(P_sub_copy, 0)
                expected = P_sub_copy.sum() / n_pairs if n_pairs > 0 else 0.0
            else:
                expected = P_sub.sum() / n_pairs if n_pairs > 0 else 0.0

            # 2-sigma tolerance
            if n_pairs > 0 and expected > 0:
                sigma = np.sqrt(expected * (1 - expected) / n_pairs)
                if abs(observed - expected) > 2 * sigma:
                    errors.append(
                        f"Block ({a},{b}) density {observed:.4f} outside "
                        f"2-sigma of expected {expected:.4f} "
                        f"(sigma={sigma:.4f})"
                    )

    return errors


def generate_dcsbm_graph(
    config: ExperimentConfig, max_retries: int = 10
) -> GraphData:
    """Generate a valid DCSBM directed graph with degree correction.

    Implements the full generation pipeline:
    1. Validate config (equal block sizes required)
    2. Sample degree correction parameters (Zipf alpha=1.0)
    3. Build probability matrix
    4. Sample edges via Bernoulli draws
    5. Validate graph (connectivity, density, degree)
    6. Retry with incremented seed on failure

    Args:
        config: Full experiment configuration.
        max_retries: Maximum generation attempts before raising error.

    Returns:
        GraphData containing the valid graph and metadata.

    Raises:
        GraphGenerationError: If no valid graph produced after max_retries.
        ValueError: If n is not evenly divisible by K.
    """
    n = config.graph.n
    K = config.graph.K
    p_in = config.graph.p_in
    p_out = config.graph.p_out
    block_size = n // K

    if n % K != 0:
        raise ValueError(
            f"n ({n}) must be evenly divisible by K ({K}) "
            f"for equal-sized blocks"
        )

    # Degree correction exponent locked at 1.0 per CONTEXT.md
    degree_correction_alpha = 1.0

    block_assignments = np.arange(n) // block_size
    last_errors: list[str] = []

    for attempt in range(max_retries):
        rng = np.random.default_rng(config.seed + attempt)

        theta = sample_theta(n, K, degree_correction_alpha, rng)
        P = build_probability_matrix(n, K, p_in, p_out, theta)
        adj = sample_adjacency(P, rng)

        errors = validate_graph(adj, P, n, K, p_in, p_out)

        if not errors:
            log.info(
                "Graph generated successfully on attempt %d "
                "(n=%d, K=%d, edges=%d)",
                attempt,
                n,
                K,
                adj.nnz,
            )
            return GraphData(
                adjacency=adj,
                block_assignments=block_assignments,
                theta=theta,
                n=n,
                K=K,
                block_size=block_size,
                generation_seed=config.seed + attempt,
                attempt=attempt,
            )

        last_errors = errors
        log.warning(
            "Graph generation attempt %d failed: %s",
            attempt,
            "; ".join(errors),
        )

    raise GraphGenerationError(
        f"Failed to generate valid graph after {max_retries} attempts. "
        f"Last errors: {'; '.join(last_errors)}"
    )
