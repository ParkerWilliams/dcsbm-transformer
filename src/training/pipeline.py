"""Full training pipeline with sufficiency gate, checkpointing, and result writing.

Orchestrates the training loop, per-epoch compliance evaluation,
checkpoint management with rolling retention, early stopping on
gate pass, and result.json output.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config.experiment import ExperimentConfig
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.results.schema import write_result
from src.training.checkpoint import (
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.training.data import WalkDataset, create_dataloader
from src.training.evaluate import ComplianceResult, evaluate_compliance
from src.training.trainer import Trainer

log = logging.getLogger(__name__)

# Gate thresholds (from CONTEXT.md)
EDGE_COMPLIANCE_THRESHOLD = 0.95
RULE_COMPLIANCE_THRESHOLD = 0.80


@dataclass
class TrainingPipelineResult:
    """Result of the full training pipeline.

    Attributes:
        gate_passed: Whether the sufficiency gate was passed.
        final_epoch: Number of epochs actually trained.
        curves: Dict with keys train_loss, edge_compliance, rule_compliance.
        result_path: Path to the written result.json, or None if not written.
        failure_reason: Reason for gate failure, or None if passed.
    """

    gate_passed: bool
    final_epoch: int
    curves: dict[str, list[float]] = field(default_factory=dict)
    result_path: str | None = None
    failure_reason: str | None = None


def run_training_pipeline(
    model: nn.Module,
    train_walks: np.ndarray,
    eval_walks: np.ndarray,
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    device: torch.device,
    results_dir: str = "results",
    max_epochs: int = 50,
    resume_from: str | None = None,
) -> TrainingPipelineResult:
    """Run the full training pipeline with gate evaluation and checkpointing.

    1. Create dataset and trainer.
    2. Optionally resume from checkpoint.
    3. For each epoch: train, evaluate compliance, checkpoint, gate check.
    4. Write result.json with curves and gate status.

    Args:
        model: TransformerLM to train.
        train_walks: Training walk array (N_train, walk_length).
        eval_walks: Evaluation walk array (N_eval, walk_length).
        graph_data: Graph for compliance evaluation.
        jumpers: Jumper info for rule compliance.
        config: Experiment configuration.
        device: Torch device.
        results_dir: Base directory for results output.
        max_epochs: Maximum training epochs.
        resume_from: Path to checkpoint to resume from, or None.

    Returns:
        TrainingPipelineResult with gate status, curves, and result path.
    """
    # Create dataset and dataloader
    dataset = WalkDataset(train_walks, config.training.w)
    dataloader = create_dataloader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.seed,
    )

    # Create trainer
    trainer = Trainer(model, config, device, max_epochs=max_epochs)

    # Track curves
    all_step_losses: list[float] = []
    edge_compliance_history: list[float] = []
    rule_compliance_history: list[float] = []
    start_epoch = 0

    # Checkpoint directory
    from src.results.experiment_id import generate_experiment_id

    experiment_id = generate_experiment_id(config)
    checkpoint_dir = Path(results_dir) / experiment_id / "checkpoints"

    # Resume from checkpoint if specified
    if resume_from is not None:
        trainer._ensure_scheduler(len(dataloader))
        restored = load_checkpoint(
            resume_from, model, trainer.optimizer, trainer.scheduler, device
        )
        start_epoch = restored["epoch"] + 1
        all_step_losses = restored["train_losses"]
        edge_compliance_history = restored["edge_compliance_history"]
        rule_compliance_history = restored["rule_compliance_history"]
        log.info("Resumed from epoch %d", restored["epoch"])

    # Training loop
    gate_passed = False
    final_epoch = 0

    for epoch in range(start_epoch, max_epochs):
        final_epoch = epoch + 1

        # Train one epoch
        epoch_losses = trainer.train_epoch(dataloader)
        all_step_losses.extend(epoch_losses)
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # Evaluate compliance
        compliance = evaluate_compliance(
            model, eval_walks, graph_data, jumpers, config, device
        )
        edge_compliance_history.append(compliance.edge_compliance)
        rule_compliance_history.append(compliance.rule_compliance)

        # Log epoch summary
        log.info(
            "Epoch %d: loss=%.4f, edge=%.3f, rule=%.3f",
            epoch + 1,
            avg_loss,
            compliance.edge_compliance,
            compliance.rule_compliance,
        )

        # Gate check
        gate_passed = (
            compliance.edge_compliance > EDGE_COMPLIANCE_THRESHOLD
            and compliance.rule_compliance > RULE_COMPLIANCE_THRESHOLD
        )

        # Checkpoint
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            model=model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            train_losses=all_step_losses,
            edge_compliance_history=edge_compliance_history,
            rule_compliance_history=rule_compliance_history,
            gate_passed=gate_passed,
        )
        cleanup_old_checkpoints(checkpoint_dir, max_keep=3)

        if gate_passed:
            log.info(
                "Gate PASSED at epoch %d: edge=%.3f, rule=%.3f",
                epoch + 1,
                compliance.edge_compliance,
                compliance.rule_compliance,
            )
            break

    # Build curves
    curves = {
        "train_loss": all_step_losses,
        "edge_compliance": edge_compliance_history,
        "rule_compliance": rule_compliance_history,
    }

    # Build metrics for result.json
    final_edge = edge_compliance_history[-1] if edge_compliance_history else 0.0
    final_rule = rule_compliance_history[-1] if rule_compliance_history else 0.0
    final_loss = (
        sum(all_step_losses[-len(epoch_losses) :]) / len(epoch_losses)
        if all_step_losses
        else 0.0
    )

    metrics: dict[str, Any] = {
        "scalars": {
            "final_train_loss": final_loss,
            "final_edge_compliance": final_edge,
            "final_rule_compliance": final_rule,
            "gate_passed": gate_passed,
            "epochs_trained": final_epoch,
        },
        "curves": curves,
    }

    # Build metadata
    metadata: dict[str, Any] = {
        "gate_passed": gate_passed,
    }
    failure_reason = None
    if not gate_passed:
        failure_reason = (
            f"Sufficiency gate not passed after {final_epoch} epochs. "
            f"Final edge compliance: {final_edge:.3f} (threshold: {EDGE_COMPLIANCE_THRESHOLD}), "
            f"Final rule compliance: {final_rule:.3f} (threshold: {RULE_COMPLIANCE_THRESHOLD})"
        )
        metadata["failure_reason"] = failure_reason
        metadata["final_edge_compliance"] = final_edge
        metadata["final_rule_compliance"] = final_rule

    # Write result.json
    try:
        result_id = write_result(
            config=config,
            metrics=metrics,
            metadata=metadata,
            results_dir=results_dir,
        )
        result_path = str(Path(results_dir) / result_id / "result.json")
    except Exception as e:
        log.error("Failed to write result.json: %s", e)
        result_path = None

    return TrainingPipelineResult(
        gate_passed=gate_passed,
        final_epoch=final_epoch,
        curves=curves,
        result_path=result_path,
        failure_reason=failure_reason,
    )
