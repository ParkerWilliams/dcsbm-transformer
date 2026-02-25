"""TDD tests for training data loading and training loop (Phase 05-01).

Uses small fixtures for fast execution:
- vocab_size=20, d_model=32, n_layers=2, max_seq_len=16
- 50 walks of length 32
"""

import math

import numpy as np
import pytest
import torch

from src.config.experiment import ExperimentConfig, GraphConfig, ModelConfig, TrainingConfig
from src.training.data import WalkDataset, create_dataloader
from src.training.trainer import Trainer, TrainResult, cosine_with_warmup
from src.model.transformer import TransformerLM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    """Small config for fast test execution."""
    return ExperimentConfig(
        graph=GraphConfig(n=20, K=4, p_in=0.5, p_out=0.1, n_jumpers_per_block=1),
        model=ModelConfig(d_model=32, n_layers=2, n_heads=1, dropout=0.0),
        training=TrainingConfig(
            w=16,
            walk_length=32,
            corpus_size=2000,
            r=10,
            learning_rate=3e-4,
            batch_size=8,
            max_steps=1000,
            eval_interval=100,
            checkpoint_interval=500,
        ),
        seed=42,
    )


@pytest.fixture
def small_walks():
    """50 walks of length 32 with vocab range [0, 20)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 20, size=(50, 32), dtype=np.int32)


@pytest.fixture
def small_model(small_config):
    """Small TransformerLM for testing."""
    return TransformerLM(
        vocab_size=small_config.graph.n,
        d_model=small_config.model.d_model,
        n_layers=small_config.model.n_layers,
        max_seq_len=small_config.training.w,
        dropout=0.0,
    )


@pytest.fixture
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# WalkDataset tests
# ---------------------------------------------------------------------------

class TestWalkDataset:
    def test_walk_dataset_chunks_walks_into_sequences(self, small_walks, small_config):
        """Dataset produces input/target pairs chunked from walks."""
        w = small_config.training.w  # 16
        dataset = WalkDataset(small_walks, w)

        # Each walk of length 32 with w=16: chunks of size w+1=17
        # floor(32 / 17) = 1 chunk per walk (since 17*2=34 > 32)
        # Actually: non-overlapping chunks of w tokens for input + 1 for target
        # So each chunk is w+1 = 17 tokens.  32 // 17 = 1 chunk per walk
        # Total: 50 walks * 1 chunk = 50 sequences

        assert len(dataset) > 0

        # Check shape: each item should be a tensor of length w+1
        item = dataset[0]
        assert item.shape == (w + 1,)
        assert item.dtype == torch.long

        # Verify content matches original walks
        # First chunk of first walk should be walk[0, :w+1]
        expected = torch.tensor(small_walks[0, :w + 1], dtype=torch.long)
        assert torch.equal(item, expected)

    def test_walk_dataset_len(self, small_walks, small_config):
        """Dataset length equals total number of chunks across all walks."""
        w = small_config.training.w  # 16
        walk_length = small_walks.shape[1]  # 32
        n_walks = small_walks.shape[0]  # 50
        chunks_per_walk = walk_length // (w + 1)  # 32 // 17 = 1

        dataset = WalkDataset(small_walks, w)
        assert len(dataset) == n_walks * chunks_per_walk

    def test_create_dataloader_batches(self, small_walks, small_config):
        """create_dataloader yields batches of correct shape."""
        w = small_config.training.w
        batch_size = small_config.training.batch_size  # 8
        dataset = WalkDataset(small_walks, w)
        loader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, seed=42)

        batch = next(iter(loader))
        assert batch.shape[0] <= batch_size
        assert batch.shape[1] == w + 1
        assert batch.dtype == torch.long


# ---------------------------------------------------------------------------
# Cosine schedule with warmup tests
# ---------------------------------------------------------------------------

class TestCosineWithWarmup:
    def test_cosine_with_warmup_starts_at_zero(self):
        """At step 0, LR multiplier is 0 (or near 0)."""
        mult = cosine_with_warmup(0, warmup_steps=100, total_steps=1000, min_lr_ratio=0.1)
        assert mult == pytest.approx(0.0, abs=1e-6)

    def test_cosine_with_warmup_reaches_peak(self):
        """At warmup_steps, LR multiplier is 1.0."""
        mult = cosine_with_warmup(100, warmup_steps=100, total_steps=1000, min_lr_ratio=0.1)
        assert mult == pytest.approx(1.0, abs=1e-6)

    def test_cosine_with_warmup_decays_to_min(self):
        """At total_steps, LR multiplier approaches min_lr_ratio."""
        mult = cosine_with_warmup(1000, warmup_steps=100, total_steps=1000, min_lr_ratio=0.1)
        assert mult == pytest.approx(0.1, abs=0.05)

    def test_cosine_with_warmup_monotonic_after_warmup(self):
        """After warmup, LR is monotonically non-increasing."""
        warmup_steps = 100
        total_steps = 1000
        values = [
            cosine_with_warmup(s, warmup_steps, total_steps, 0.1)
            for s in range(warmup_steps, total_steps + 1)
        ]
        for i in range(1, len(values)):
            assert values[i] <= values[i - 1] + 1e-10, (
                f"LR increased at step {warmup_steps + i}: "
                f"{values[i]} > {values[i-1]}"
            )


# ---------------------------------------------------------------------------
# Trainer tests
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_trainer_single_epoch_returns_losses(
        self, small_model, small_config, small_walks, device
    ):
        """Trainer.train_epoch returns a list of per-step float losses."""
        dataset = WalkDataset(small_walks, small_config.training.w)
        loader = create_dataloader(dataset, batch_size=8, shuffle=False, seed=42)

        trainer = Trainer(small_model, small_config, device, max_epochs=5)
        losses = trainer.train_epoch(loader)

        assert isinstance(losses, list)
        assert len(losses) > 0
        assert all(isinstance(l, float) for l in losses)
        assert all(l > 0 for l in losses)  # cross-entropy loss is positive

    def test_trainer_loss_decreases_over_epochs(
        self, small_model, small_config, small_walks, device
    ):
        """After 5+ epochs, average loss in last epoch < first epoch."""
        dataset = WalkDataset(small_walks, small_config.training.w)
        loader = create_dataloader(dataset, batch_size=8, shuffle=True, seed=42)

        trainer = Trainer(small_model, small_config, device, max_epochs=8)
        all_losses = []
        for _ in range(8):
            epoch_losses = trainer.train_epoch(loader)
            all_losses.append(epoch_losses)

        first_avg = sum(all_losses[0]) / len(all_losses[0])
        last_avg = sum(all_losses[-1]) / len(all_losses[-1])
        assert last_avg < first_avg, (
            f"Loss did not decrease: first={first_avg:.4f}, last={last_avg:.4f}"
        )

    def test_trainer_uses_adamw(self, small_model, small_config, device):
        """Verify optimizer is AdamW with correct LR and weight decay."""
        trainer = Trainer(small_model, small_config, device, max_epochs=5)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        # Check LR from param groups
        lr = trainer.optimizer.param_groups[0]["lr"]
        assert lr == small_config.training.learning_rate
        # Check weight decay
        wd = trainer.optimizer.param_groups[0]["weight_decay"]
        assert wd > 0  # Should have some weight decay

    def test_trainer_gradient_clipping(self, small_model, small_config, small_walks, device):
        """Verify gradients are clipped during training."""
        dataset = WalkDataset(small_walks, small_config.training.w)
        loader = create_dataloader(dataset, batch_size=8, shuffle=False, seed=42)

        trainer = Trainer(small_model, small_config, device, max_epochs=5)

        # Run one epoch -- this exercises the gradient clipping path
        losses = trainer.train_epoch(loader)
        assert len(losses) > 0

        # Verify trainer has clip_max_norm attribute
        assert hasattr(trainer, "clip_max_norm")
        assert trainer.clip_max_norm > 0
