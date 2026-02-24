"""Graph data structures for DCSBM generation and storage."""

from dataclasses import dataclass

import numpy as np
import scipy.sparse


@dataclass(frozen=True)
class GraphData:
    """Immutable container for a generated DCSBM graph and its metadata.

    Holds the sparse directed adjacency matrix alongside block structure
    information, degree correction parameters, and generation provenance.
    Uses frozen=True for immutability but omits slots=True since numpy/scipy
    objects don't interact well with __slots__.
    """

    adjacency: scipy.sparse.csr_matrix  # directed adjacency (n x n)
    block_assignments: np.ndarray  # int array of length n, vertex -> block
    theta: np.ndarray  # float array of length n, degree correction parameters
    n: int  # number of vertices
    K: int  # number of blocks
    block_size: int  # vertices per block (n // K)
    generation_seed: int  # seed used for this specific generation attempt
    attempt: int  # which retry attempt produced this graph (0-indexed)
