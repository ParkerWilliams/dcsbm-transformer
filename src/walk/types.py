"""Walk data structures for walk generation and storage."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class JumperEvent:
    """Record of a single jumper encounter during walk generation.

    Tracks the vertex, step, target block, and expected arrival step
    for downstream evaluation of rule compliance.
    """

    vertex_id: int
    step: int
    target_block: int
    expected_arrival_step: int


@dataclass(frozen=True)
class WalkResult:
    """Immutable container for generated walks and their metadata.

    Holds the walk array, per-walk jumper events, and per-walk seeds
    for reproducibility. Uses frozen=True but omits slots=True since
    numpy arrays don't interact well with __slots__.
    """

    walks: np.ndarray  # int32 array of shape (num_walks, walk_length)
    events: list[list[JumperEvent]]  # per-walk event lists
    walk_seeds: np.ndarray  # int64 array of per-walk seeds
