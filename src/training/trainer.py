"""Training loop for TransformerLM with AdamW and cosine LR schedule.

Provides the Trainer class that handles single-epoch training with
cross-entropy loss, gradient clipping, and per-step loss tracking.
The cosine_with_warmup schedule implements 10% linear warmup followed
by cosine decay to a minimum LR ratio.
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config.experiment import ExperimentConfig
from src.model.types import ExtractionMode


def cosine_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """LambdaLR multiplier: linear warmup then cosine decay.

    Args:
        step: Current optimizer step.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak (default 0.1 = 10%).

    Returns:
        LR multiplier in [0, 1] (or [min_lr_ratio, 1] after warmup).
    """
    if step < warmup_steps:
        return step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    # Clamp progress to [0, 1] for safety
    progress = min(1.0, progress)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


@dataclass
class TrainResult:
    """Result from multi-epoch training.

    Attributes:
        epoch_losses: Per-epoch list of per-step losses.
        final_lr: Learning rate at the end of training.
    """

    epoch_losses: list[list[float]] = field(default_factory=list)
    final_lr: float = 0.0


class Trainer:
    """Single-epoch trainer for TransformerLM.

    Handles AdamW optimizer, cosine LR schedule with warmup,
    gradient clipping, and per-step loss tracking. Does NOT
    handle multi-epoch loops or gate evaluation (that's pipeline.py).

    Args:
        model: TransformerLM instance.
        config: ExperimentConfig with training hyperparameters.
        device: torch device for training.
        max_epochs: Maximum number of epochs (for computing total_steps
            and warmup_steps for the LR schedule).
        steps_per_epoch: Steps per epoch (auto-computed from dataloader
            if not provided). Set this if you know it at init time.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
        max_epochs: int = 50,
        steps_per_epoch: int | None = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.max_epochs = max_epochs
        self.clip_max_norm = 1.0  # Claude's discretion per CONTEXT.md
        self._weight_decay = 0.01  # Claude's discretion per CONTEXT.md
        self._min_lr_ratio = 0.1  # Claude's discretion per CONTEXT.md
        self._epoch_count = 0

        # Create AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=self._weight_decay,
        )

        # LR schedule will be initialized on first train_epoch call
        # when we know steps_per_epoch from the dataloader
        self._steps_per_epoch = steps_per_epoch
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._total_steps: int | None = None
        self._warmup_steps: int | None = None

    def _ensure_scheduler(self, steps_per_epoch: int) -> None:
        """Initialize the LR scheduler once we know steps_per_epoch."""
        if self._scheduler is not None:
            return

        self._steps_per_epoch = steps_per_epoch
        self._total_steps = self.max_epochs * steps_per_epoch
        self._warmup_steps = int(0.1 * self._total_steps)  # 10% warmup

        warmup = self._warmup_steps
        total = self._total_steps
        min_lr = self._min_lr_ratio

        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_with_warmup(step, warmup, total, min_lr),
        )

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LambdaLR | None:
        """Current LR scheduler (None until first train_epoch call)."""
        return self._scheduler

    @property
    def current_lr(self) -> float:
        """Current learning rate from optimizer."""
        return self.optimizer.param_groups[0]["lr"]

    def train_epoch(self, dataloader: DataLoader) -> list[float]:
        """Train for one epoch, returning per-step losses.

        Args:
            dataloader: DataLoader yielding batches of shape [B, w+1].

        Returns:
            List of per-step loss values (floats).
        """
        self._ensure_scheduler(len(dataloader))
        self.model.train()

        step_losses: list[float] = []
        vocab_size = self.model.vocab_size

        for batch in dataloader:
            batch = batch.to(self.device)
            # Split into input (first w tokens) and target (last w tokens)
            inputs = batch[:, :-1]   # [B, w]
            targets = batch[:, 1:]   # [B, w]

            # Forward pass -- lean training mode, no extraction
            output = self.model(inputs, mode=ExtractionMode.NONE)

            # Cross-entropy loss: reshape to [B*w, vocab_size] and [B*w]
            loss = F.cross_entropy(
                output.logits.reshape(-1, vocab_size),
                targets.reshape(-1),
            )

            # Backward + clip + step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.clip_max_norm
            )
            self.optimizer.step()
            self._scheduler.step()

            step_losses.append(loss.item())

        self._epoch_count += 1
        return step_losses
