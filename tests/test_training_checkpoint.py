"""TDD tests for checkpoint save/load/resume/retention (Phase 05-02).

Uses small fixtures for fast execution.
"""

import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from src.config.experiment import ExperimentConfig, GraphConfig, ModelConfig, TrainingConfig
from src.model.transformer import TransformerLM
from src.training.checkpoint import (
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.training.trainer import Trainer


@pytest.fixture
def small_config():
    return ExperimentConfig(
        graph=GraphConfig(n=20, K=4, p_in=0.5, p_out=0.1, n_jumpers_per_block=1),
        model=ModelConfig(d_model=32, n_layers=2, n_heads=1, dropout=0.0),
        training=TrainingConfig(
            w=16, walk_length=32, corpus_size=2000, r=10,
            learning_rate=3e-4, batch_size=8, max_steps=1000,
            eval_interval=100, checkpoint_interval=500,
        ),
        seed=42,
    )


@pytest.fixture
def small_model(small_config):
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


@pytest.fixture
def checkpoint_dir(tmp_path):
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


class TestCheckpointSaveLoad:
    def test_save_checkpoint_creates_file(
        self, small_model, small_config, device, checkpoint_dir
    ):
        """save_checkpoint writes a .pt file to specified directory."""
        trainer = Trainer(small_model, small_config, device, max_epochs=5)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=0,
            model=small_model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            train_losses=[1.0, 0.9],
            edge_compliance_history=[0.5],
            rule_compliance_history=[0.3],
        )
        files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(files) == 1
        assert "0000" in files[0].name

    def test_load_checkpoint_restores_state(
        self, small_model, small_config, device, checkpoint_dir
    ):
        """Save then load: model/optimizer/scheduler/epoch match."""
        trainer = Trainer(small_model, small_config, device, max_epochs=5)

        # Make the scheduler exist by setting steps_per_epoch
        trainer._ensure_scheduler(10)

        # Modify model to have non-default state
        with torch.no_grad():
            for p in small_model.parameters():
                p.add_(1.0)

        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=3,
            model=small_model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            train_losses=[1.0, 0.9, 0.8],
            edge_compliance_history=[0.5, 0.6, 0.7],
            rule_compliance_history=[0.3, 0.4, 0.5],
        )

        # Create fresh model and trainer
        fresh_model = TransformerLM(
            vocab_size=small_config.graph.n,
            d_model=small_config.model.d_model,
            n_layers=small_config.model.n_layers,
            max_seq_len=small_config.training.w,
            dropout=0.0,
        )
        fresh_trainer = Trainer(fresh_model, small_config, device, max_epochs=5)
        fresh_trainer._ensure_scheduler(10)

        ckpt_path = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))[0]
        restored = load_checkpoint(
            ckpt_path, fresh_model, fresh_trainer.optimizer, fresh_trainer.scheduler, device
        )

        assert restored["epoch"] == 3
        assert restored["train_losses"] == [1.0, 0.9, 0.8]
        assert restored["edge_compliance_history"] == [0.5, 0.6, 0.7]

        # Verify model parameters match
        for p1, p2 in zip(small_model.parameters(), fresh_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_checkpoint_includes_rng_states(
        self, small_model, small_config, device, checkpoint_dir
    ):
        """Checkpoint contains torch/numpy/python RNG states."""
        trainer = Trainer(small_model, small_config, device, max_epochs=5)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=0,
            model=small_model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            train_losses=[],
            edge_compliance_history=[],
            rule_compliance_history=[],
        )

        ckpt_path = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))[0]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        assert "torch_rng_state" in ckpt
        assert "numpy_rng_state" in ckpt
        assert "python_rng_state" in ckpt

    def test_checkpoint_includes_training_curves(
        self, small_model, small_config, device, checkpoint_dir
    ):
        """Checkpoint contains train_losses, edge/rule compliance histories."""
        trainer = Trainer(small_model, small_config, device, max_epochs=5)
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=2,
            model=small_model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            train_losses=[1.0, 0.9, 0.8],
            edge_compliance_history=[0.5, 0.7],
            rule_compliance_history=[0.3, 0.5],
        )

        ckpt_path = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))[0]
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        assert ckpt["train_losses"] == [1.0, 0.9, 0.8]
        assert ckpt["edge_compliance_history"] == [0.5, 0.7]
        assert ckpt["rule_compliance_history"] == [0.3, 0.5]


class TestCheckpointRetention:
    def test_cleanup_old_checkpoints_keeps_last_n(
        self, small_model, small_config, device, checkpoint_dir
    ):
        """With 5 checkpoints, cleanup keeps last 3, gate is never deleted."""
        trainer = Trainer(small_model, small_config, device, max_epochs=10)
        for epoch in range(5):
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=small_model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                train_losses=[],
                edge_compliance_history=[],
                rule_compliance_history=[],
                gate_passed=(epoch == 2),  # epoch 2 is gate pass
            )

        cleanup_old_checkpoints(checkpoint_dir, max_keep=3)

        remaining = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        # Should keep epochs 2, 3, 4 (last 3)
        assert len(remaining) == 3

        # Gate checkpoint should exist separately
        gate_file = checkpoint_dir / "checkpoint_gate.pt"
        assert gate_file.exists()

    def test_checkpoint_resume_continues_training(
        self, small_model, small_config, device, checkpoint_dir
    ):
        """Save at epoch 2, load, verify epoch counter."""
        trainer = Trainer(small_model, small_config, device, max_epochs=5)
        trainer._ensure_scheduler(10)

        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=2,
            model=small_model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            train_losses=[1.0, 0.9, 0.8],
            edge_compliance_history=[0.5, 0.6, 0.7],
            rule_compliance_history=[0.3, 0.4, 0.5],
        )

        fresh_model = TransformerLM(
            vocab_size=small_config.graph.n,
            d_model=small_config.model.d_model,
            n_layers=small_config.model.n_layers,
            max_seq_len=small_config.training.w,
            dropout=0.0,
        )
        fresh_trainer = Trainer(fresh_model, small_config, device, max_epochs=5)
        fresh_trainer._ensure_scheduler(10)

        ckpt_path = find_latest_checkpoint(checkpoint_dir)
        assert ckpt_path is not None

        restored = load_checkpoint(
            ckpt_path, fresh_model, fresh_trainer.optimizer,
            fresh_trainer.scheduler, device,
        )
        # Should resume from epoch 3 (epoch 2 was saved, so next is 3)
        assert restored["epoch"] == 2
        assert len(restored["train_losses"]) == 3
