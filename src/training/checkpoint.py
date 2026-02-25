"""Checkpoint save/load/resume with full RNG state and rolling retention.

Supports saving model, optimizer, scheduler, RNG states, and training
curves. Rolling retention keeps the last N epoch checkpoints plus any
gate-pass checkpoint.
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    train_losses: list[float],
    edge_compliance_history: list[float],
    rule_compliance_history: list[float],
    gate_passed: bool = False,
) -> Path:
    """Save a training checkpoint with full state.

    Args:
        checkpoint_dir: Directory to save checkpoint files.
        epoch: Current epoch number.
        model: Model to save.
        optimizer: Optimizer to save.
        scheduler: LR scheduler to save (None if not yet initialized).
        train_losses: Accumulated per-step training losses.
        edge_compliance_history: Per-epoch edge compliance values.
        rule_compliance_history: Per-epoch rule compliance values.
        gate_passed: Whether this checkpoint is a gate-pass checkpoint.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "train_losses": train_losses,
        "edge_compliance_history": edge_compliance_history,
        "rule_compliance_history": rule_compliance_history,
        "gate_passed": gate_passed,
    }

    path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, path)

    # Also save as gate checkpoint if gate passed
    if gate_passed:
        gate_path = checkpoint_dir / "checkpoint_gate.pt"
        torch.save(checkpoint, gate_path)

    return path


def load_checkpoint(
    checkpoint_path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
) -> dict:
    """Load a checkpoint and restore all state.

    Restores model weights, optimizer state, scheduler state, and
    all RNG states for seamless training continuation.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optimizer to load state into.
        scheduler: LR scheduler to load state into (None to skip).
        device: Device to map tensors to.

    Returns:
        Dict with: epoch, train_losses, edge_compliance_history,
        rule_compliance_history, gate_passed.
    """
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore RNG states
    if "torch_rng_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["torch_rng_state"])
    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        cuda_states = checkpoint["cuda_rng_state"]
        if cuda_states:
            for i, state in enumerate(cuda_states):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(state, device=i)
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if "python_rng_state" in checkpoint:
        random.setstate(checkpoint["python_rng_state"])

    return {
        "epoch": checkpoint["epoch"],
        "train_losses": checkpoint.get("train_losses", []),
        "edge_compliance_history": checkpoint.get("edge_compliance_history", []),
        "rule_compliance_history": checkpoint.get("rule_compliance_history", []),
        "gate_passed": checkpoint.get("gate_passed", False),
    }


def cleanup_old_checkpoints(checkpoint_dir: Path | str, max_keep: int = 3) -> None:
    """Remove old epoch checkpoints, keeping the last max_keep.

    Never removes the gate checkpoint (checkpoint_gate.pt).

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        max_keep: Maximum number of epoch checkpoints to retain.
    """
    checkpoint_dir = Path(checkpoint_dir)
    gate_path = checkpoint_dir / "checkpoint_gate.pt"

    # Find all epoch checkpoints (not gate)
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))

    if len(epoch_checkpoints) <= max_keep:
        return

    # Delete oldest, keep last max_keep
    to_delete = epoch_checkpoints[:-max_keep]
    for ckpt in to_delete:
        if ckpt != gate_path:
            ckpt.unlink()


def find_latest_checkpoint(checkpoint_dir: Path | str) -> Path | None:
    """Find the most recent epoch checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        Path to the latest checkpoint, or None if no checkpoints exist.
    """
    checkpoint_dir = Path(checkpoint_dir)
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not epoch_checkpoints:
        return None
    return epoch_checkpoints[-1]
