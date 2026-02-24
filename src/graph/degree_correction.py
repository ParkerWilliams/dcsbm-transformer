"""Degree correction parameter sampling following Zipf's law (Karrer & Newman 2011)."""

import numpy as np


def sample_theta(
    n: int, K: int, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample degree correction parameters from a Zipf (power-law) distribution.

    Per CONTEXT.md locked decisions:
    - Power-law with fixed alpha=1.0 (classic Zipf), mimicking token frequency
    - Single theta_i per vertex for both in-degree and out-degree
    - Normalize per-block so each block's theta values sum to block_size,
      preserving expected total degree from the uncorrected SBM

    Args:
        n: Total number of vertices.
        K: Number of blocks.
        alpha: Power-law exponent (locked at 1.0).
        rng: numpy random Generator for reproducibility.

    Returns:
        Array of shape (n,) with per-vertex degree correction parameters.
    """
    block_size = n // K
    theta = np.zeros(n, dtype=np.float64)

    for b in range(K):
        start = b * block_size
        end = start + block_size

        # Zipf: theta_i proportional to 1/rank^alpha
        ranks = np.arange(1, block_size + 1, dtype=np.float64)
        raw = 1.0 / (ranks**alpha)

        # Randomize which vertex gets which rank
        rng.shuffle(raw)

        # Normalize so block sum = block_size (preserves expected total degree)
        theta[start:end] = raw * (block_size / raw.sum())

    # Verify normalization
    for b in range(K):
        start = b * block_size
        end = start + block_size
        block_sum = theta[start:end].sum()
        assert abs(block_sum - block_size) < 1e-10, (
            f"Block {b} theta sum {block_sum} != {block_size}"
        )

    return theta
