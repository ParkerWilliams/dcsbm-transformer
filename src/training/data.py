"""Walk data loading for transformer training.

Converts numpy walk arrays into PyTorch datasets and DataLoaders
with reproducible shuffling and deterministic worker seeding.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.reproducibility.seed import seed_worker


class WalkDataset(Dataset):
    """Dataset that chunks walks into non-overlapping subsequences for training.

    Each walk of length L is split into floor(L / (w+1)) chunks of size w+1.
    For each chunk, the first w tokens are the input and the last w tokens
    (shifted by 1) are the target. This gives standard next-token prediction.

    Args:
        walks: Numpy array of shape (num_walks, walk_length) with integer vertex IDs.
        context_window: Context window size w. Chunks are w+1 tokens each.
    """

    def __init__(self, walks: np.ndarray, context_window: int):
        n_walks, walk_length = walks.shape
        chunk_size = context_window + 1  # w tokens input + 1 for shifted target
        chunks_per_walk = walk_length // chunk_size

        if chunks_per_walk == 0:
            raise ValueError(
                f"Walk length {walk_length} too short for context window {context_window}. "
                f"Need at least {chunk_size} tokens per walk."
            )

        # Pre-chunk all walks into a single tensor
        all_chunks = []
        for i in range(n_walks):
            for j in range(chunks_per_walk):
                start = j * chunk_size
                chunk = walks[i, start : start + chunk_size]
                all_chunks.append(chunk)

        self.data = torch.tensor(np.array(all_chunks), dtype=torch.long)
        self.context_window = context_window

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a chunk of shape (w+1,) — first w = input, last w = target."""
        return self.data[idx]


def create_dataloader(
    dataset: WalkDataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader with reproducible shuffling.

    Uses a torch.Generator seeded deterministically and the seed_worker
    function from reproducibility module for worker-level seeding.

    Args:
        dataset: WalkDataset instance.
        batch_size: Number of sequences per batch.
        shuffle: Whether to shuffle data each epoch.
        seed: Random seed for the DataLoader generator.

    Returns:
        Configured DataLoader.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,  # Single-process for determinism
    )
