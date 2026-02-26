"""Held-out evaluation split for exploratory/confirmatory walk assignment.

Implements deterministic, stratified 50/50 split of evaluation walks into
exploratory and confirmatory sets. Split is applied at evaluation time
(after behavioral labels are computed) and tagged in result.json.

Pre-registered in docs/pre-registration.md, Section 5.
"""

import numpy as np

# Fixed seed for deterministic split assignment.
# Documented in pre-registration.md Section 5.1.
SPLIT_SEED: int = 2026

# Split label constants
EXPLORATORY: str = "exploratory"
CONFIRMATORY: str = "confirmatory"


def assign_split(
    failure_index: np.ndarray,
    split_seed: int = SPLIT_SEED,
) -> np.ndarray:
    """Assign each walk to exploratory or confirmatory split.

    Stratified by event type (violation vs non-violation) to ensure
    equal proportions in each set. Deterministic via fixed seed +
    numpy default_rng.

    Args:
        failure_index: First violation step per walk, shape [n_walks].
            -1 indicates no violation.
        split_seed: Fixed seed for reproducibility.

    Returns:
        Array of strings, shape [n_walks], each 'exploratory' or 'confirmatory'.
    """
    n_walks = len(failure_index)
    if n_walks == 0:
        return np.array([], dtype="U13")

    splits = np.empty(n_walks, dtype="U13")
    rng = np.random.default_rng(split_seed)

    # Separate by violation status
    violation_mask = failure_index >= 0
    viol_indices = np.where(violation_mask)[0]
    nonviol_indices = np.where(~violation_mask)[0]

    # Shuffle and split each pool 50/50
    for indices in [viol_indices, nonviol_indices]:
        if len(indices) == 0:
            continue
        shuffled = rng.permutation(indices)
        half = len(shuffled) // 2
        splits[shuffled[:half]] = EXPLORATORY
        splits[shuffled[half:]] = CONFIRMATORY

    return splits
