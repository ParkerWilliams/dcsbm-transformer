"""Centralized seed management for full reproducibility.

Seeds all RNG sources (Python random, NumPy, PyTorch CPU/GPU) and configures
CUDA deterministic algorithms from a single master seed.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Must be called before any random operations (model init, data loading, etc.).
    Seeds are set in this order:

    1. Python random module
    2. NumPy legacy global RNG
    3. PyTorch CPU RNG
    4. PyTorch CUDA RNG (all GPUs)
    5. cuDNN deterministic mode
    6. cuDNN benchmark disabled (prevents non-deterministic algorithm selection)
    7. PyTorch deterministic algorithms enforced
    8. cuBLAS workspace config for deterministic matrix multiplications

    Args:
        seed: Master seed value (e.g., 42).
    """
    # 1. Python stdlib random
    random.seed(seed)

    # 2. NumPy legacy global RNG (many libraries still use this)
    np.random.seed(seed)

    # 3. PyTorch CPU
    torch.manual_seed(seed)

    # 4. PyTorch CUDA (all GPUs)
    torch.cuda.manual_seed_all(seed)

    # 5. cuDNN deterministic mode
    torch.backends.cudnn.deterministic = True

    # 6. Disable cuDNN benchmark (prevents non-deterministic algo selection)
    torch.backends.cudnn.benchmark = False

    # 7. Enforce deterministic algorithms across all PyTorch operations
    torch.use_deterministic_algorithms(True)

    # 8. cuBLAS workspace config for deterministic matrix multiplications
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def verify_seed_determinism(seed: int) -> bool:
    """Verify that setting the seed produces identical sequences.

    Sets the seed, generates 10 values from each of random, numpy, and torch.
    Resets the seed, generates 10 more. Returns True if all three sequences
    are identical. This is the self-test that proves seed control works.

    Args:
        seed: Seed value to test.

    Returns:
        True if all RNG sources produce identical sequences after re-seeding.
    """
    set_seed(seed)
    r1 = [random.random() for _ in range(10)]
    n1 = np.random.rand(10).tolist()
    t1 = torch.rand(10).tolist()

    set_seed(seed)
    r2 = [random.random() for _ in range(10)]
    n2 = np.random.rand(10).tolist()
    t2 = torch.rand(10).tolist()

    return r1 == r2 and n1 == n2 and t1 == t2


def seed_worker(worker_id: int) -> None:
    """Worker init function for PyTorch DataLoader reproducibility.

    Derives a worker-specific seed from the PyTorch initial seed and uses it
    to seed numpy and random, ensuring each DataLoader worker produces
    deterministic results.

    Usage::

        g = torch.Generator()
        g.manual_seed(config.seed)
        DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            generator=g,
            num_workers=4,
        )

    Args:
        worker_id: The DataLoader worker ID (0 to num_workers-1).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
