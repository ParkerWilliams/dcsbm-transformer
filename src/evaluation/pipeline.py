"""Fused evaluation pipeline: autoregressive generation with SVD extraction and behavioral labeling.

Generates sequences autoregressively while simultaneously collecting SVD metrics
across three targets (QK^T, WvWo, AVWo) and behavioral labels in a single
inference pass. Outputs token_metrics.npz and result.json summary.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config.experiment import ExperimentConfig
from src.evaluation.behavioral import RuleOutcome, classify_steps
from src.evaluation.svd_metrics import (
    compute_all_metrics,
    grassmannian_distance,
    guard_matrix_for_svd,
)
from src.evaluation.split import SPLIT_SEED
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.model.types import ExtractionMode

log = logging.getLogger(__name__)

# Metric names computed for singular-value-only targets
SV_METRIC_NAMES = [
    "stable_rank",
    "spectral_entropy",
    "spectral_gap_1_2",
    "spectral_gap_2_3",
    "spectral_gap_4_5",
    "condition_number",
]

# Additional metrics requiring full SVD (U, S, Vh)
FULL_SVD_METRIC_NAMES = [
    "rank1_residual_norm",
    "read_write_alignment",
]

# SVD targets
SVD_TARGETS = ["qkt", "wvwo", "avwo"]


@dataclass
class EvaluationResult:
    """Result of fused evaluation on generated sequences.

    Attributes:
        generated: Generated token IDs, shape [n_sequences, max_steps].
        edge_valid: Edge validity per step, shape [n_sequences, max_steps-1].
        rule_outcome: Rule compliance per step, shape [n_sequences, max_steps-1].
        failure_index: First rule violation step per sequence, shape [n_sequences].
        svd_metrics: Token-level SVD metrics keyed by target.layer.metric.
            Each value has shape [n_sequences, max_steps-1].
        guard_activations: Count of guard activations per target.layer.
        sequence_lengths: Actual length per sequence, shape [n_sequences].
    """

    generated: np.ndarray
    edge_valid: np.ndarray
    rule_outcome: np.ndarray
    failure_index: np.ndarray
    svd_metrics: dict[str, np.ndarray]
    guard_activations: dict[str, int]
    sequence_lengths: np.ndarray


def _compute_avwo_for_layer(
    attention_weights: torch.Tensor,
    values: torch.Tensor,
    model: nn.Module,
    layer_idx: int,
) -> torch.Tensor:
    """Compute AVWo (net residual update) for a single layer.

    AVWo = (A @ V) @ W_o.weight.T represents the actual output added to the
    residual stream. nn.Linear computes x @ W^T internally, so this matches
    the physical computation.

    Args:
        attention_weights: Attention weights for this layer, shape [B, T, T].
        values: Value matrix for this layer, shape [B, T, D].
        model: TransformerLM model (for accessing W_o weights).
        layer_idx: Index of the transformer block.

    Returns:
        AVWo tensor of shape [B, T, D].
    """
    AV = attention_weights @ values  # [B, T, D]
    Wo = model.blocks[layer_idx].attention.W_o.weight  # [D, D]
    return AV @ Wo.T  # [B, T, D] -- matches nn.Linear's x @ W^T


def fused_evaluate(
    model: nn.Module,
    eval_walks: np.ndarray,
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    device: torch.device,
    batch_size: int = 32,
) -> EvaluationResult:
    """Generate sequences with simultaneous SVD metric collection and behavioral labeling.

    Performs autoregressive free-generation using ExtractionMode.SVD_TARGETS,
    computing SVD metrics at every step for positions >= w. Behavioral labels
    are computed after generation completes for each batch.

    Args:
        model: Trained TransformerLM in eval mode.
        eval_walks: Evaluation walk array of shape [N, walk_length].
        graph_data: Graph data with adjacency matrix and block assignments.
        jumpers: List of JumperInfo for rule compliance checking.
        config: Experiment configuration.
        device: Device for computation.
        batch_size: Number of sequences per batch.

    Returns:
        EvaluationResult with generated sequences, behavioral labels, and SVD metrics.
    """
    model.eval()
    w = config.training.w
    n_layers = config.model.n_layers
    n_sequences = eval_walks.shape[0]

    # Build jumper lookup
    jumper_map = {j.vertex_id: j for j in jumpers}

    # Compute generation lengths
    default_length = 4 * w
    max_r = max((j.r for j in jumpers), default=0)
    max_possible_length = default_length + max_r + 1 if max_r > 0 else default_length

    # Compute WvWo SVD once (static weight matrices)
    wvwo = model.get_wvwo()  # [n_layers, D, D]

    # Pre-compute WvWo metrics per layer
    wvwo_layer_metrics: dict[int, dict[str, float]] = {}
    guard_activations: dict[str, int] = {}

    for layer_idx in range(n_layers):
        wvwo_matrix = wvwo[layer_idx]  # [D, D]
        wvwo_clean, guard_fired = guard_matrix_for_svd(wvwo_matrix)
        guard_key = f"wvwo.layer_{layer_idx}"
        guard_activations[guard_key] = 1 if guard_fired else 0

        U, S, Vh = torch.linalg.svd(wvwo_clean, full_matrices=False)
        metrics = compute_all_metrics(S, U=U, Vh=Vh)
        wvwo_layer_metrics[layer_idx] = {
            k: v.item() if v.dim() == 0 else v.cpu().numpy()
            for k, v in metrics.items()
        }

    # Build SVD metric key list
    all_metric_names = SV_METRIC_NAMES + FULL_SVD_METRIC_NAMES + ["grassmannian_distance"]

    # Pre-allocate output arrays (NaN-filled)
    max_steps = max_possible_length
    svd_metric_arrays: dict[str, np.ndarray] = {}

    for target in SVD_TARGETS:
        for layer_idx in range(n_layers):
            for metric_name in all_metric_names:
                key = f"{target}.layer_{layer_idx}.{metric_name}"
                svd_metric_arrays[key] = np.full(
                    (n_sequences, max_steps - 1), np.nan, dtype=np.float32
                )

    all_generated = np.zeros((n_sequences, max_steps), dtype=np.int64)
    all_edge_valid = np.zeros((n_sequences, max_steps - 1), dtype=bool)
    all_rule_outcome = np.full(
        (n_sequences, max_steps - 1), RuleOutcome.NOT_APPLICABLE, dtype=np.int32
    )
    all_failure_index = np.full(n_sequences, -1, dtype=np.int32)
    all_seq_lengths = np.zeros(n_sequences, dtype=np.int32)

    # Process in batches
    for batch_start in range(0, n_sequences, batch_size):
        batch_end = min(batch_start + batch_size, n_sequences)
        B_actual = batch_end - batch_start

        # Seed with first token of each eval walk
        start_tokens = torch.tensor(
            eval_walks[batch_start:batch_end, :1], dtype=torch.long, device=device
        )
        generated = start_tokens  # [B, 1]

        # Per-sequence target lengths (may extend for tail)
        target_lengths = np.full(B_actual, default_length, dtype=np.int32)

        # Track previous step U for Grassmannian distance per layer per target
        # Keys: (target, layer_idx) -> U tensor [B, k, grassmannian_k]
        grassmannian_k = 2
        u_prev: dict[tuple[str, int], torch.Tensor | None] = {}
        for target in ["qkt", "avwo"]:
            for layer_idx in range(n_layers):
                u_prev[(target, layer_idx)] = None

        # Per-step autoregressive generation
        with torch.no_grad():
            for step in range(max_possible_length - 1):
                # Check if all sequences have reached their target length
                current_lengths = generated.shape[1]
                if all(current_lengths >= tl for tl in target_lengths):
                    break

                # Context window: last w tokens
                context = generated[:, -w:]
                output = model(context, mode=ExtractionMode.SVD_TARGETS)

                # Next token via argmax
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)

                # Tail extension check
                for b_idx in range(B_actual):
                    token_val = next_token[b_idx, 0].item()
                    current_pos = generated.shape[1] - 1  # position of newly added token
                    if (
                        token_val in jumper_map
                        and current_pos > default_length - max_r
                        and max_r > 0
                    ):
                        j = jumper_map[token_val]
                        new_target = current_pos + j.r + 1
                        target_lengths[b_idx] = max(
                            target_lengths[b_idx], min(new_target, max_possible_length)
                        )

                # SVD collection (only for step >= w per SVD-06)
                if step >= w - 1:  # step is 0-indexed; position step+1 is where we just generated
                    for layer_idx in range(n_layers):
                        # --- QK^T SVD ---
                        qkt_matrix = output.qkt[:, layer_idx]  # [B, T, T]
                        # Take the last row's context: the QK^T at the prediction position
                        # For SVD, we use the full T x T matrix at this context window
                        qkt_clean, qkt_guard = guard_matrix_for_svd(qkt_matrix)
                        qkt_key = f"qkt.layer_{layer_idx}"
                        guard_activations[qkt_key] = (
                            guard_activations.get(qkt_key, 0) + (1 if qkt_guard else 0)
                        )

                        U_qkt, S_qkt, Vh_qkt = torch.linalg.svd(
                            qkt_clean, full_matrices=False
                        )
                        qkt_metrics = compute_all_metrics(S_qkt, U=U_qkt, Vh=Vh_qkt)

                        # Store QK^T metrics (take mean over batch dimension T for per-sequence scalar)
                        # Actually, each sequence gets the last position's metrics
                        for metric_name, metric_val in qkt_metrics.items():
                            key = f"qkt.layer_{layer_idx}.{metric_name}"
                            if key in svd_metric_arrays:
                                # metric_val is [B] or [B, T] -- we want per-sequence scalar
                                # For batched matrices, take the mean across T dimension
                                val = metric_val.mean(dim=-1) if metric_val.dim() > 1 else metric_val
                                vals_np = val.cpu().numpy()
                                for b_idx in range(B_actual):
                                    svd_metric_arrays[key][
                                        batch_start + b_idx, step
                                    ] = vals_np[b_idx] if vals_np.ndim > 0 else vals_np.item()

                        # QK^T Grassmannian distance
                        u_key = ("qkt", layer_idx)
                        U_curr_k = U_qkt[:, :, :grassmannian_k]  # [B, T, k]
                        # Mean over T to get representative subspace per sequence
                        if u_prev[u_key] is not None:
                            gdist = grassmannian_distance(
                                u_prev[u_key], U_curr_k, k=grassmannian_k
                            )
                            gdist_vals = gdist.mean(dim=-1) if gdist.dim() > 1 else gdist
                            gdist_np = gdist_vals.cpu().numpy()
                            gkey = f"qkt.layer_{layer_idx}.grassmannian_distance"
                            for b_idx in range(B_actual):
                                svd_metric_arrays[gkey][
                                    batch_start + b_idx, step
                                ] = gdist_np[b_idx] if gdist_np.ndim > 0 else gdist_np.item()
                        u_prev[u_key] = U_curr_k.clone()

                        # --- AVWo SVD ---
                        A_layer = output.attention_weights[:, layer_idx]  # [B, T, T]
                        V_layer = output.values[:, layer_idx]  # [B, T, D]
                        avwo_matrix = _compute_avwo_for_layer(
                            A_layer, V_layer, model, layer_idx
                        )  # [B, T, D]

                        avwo_clean, avwo_guard = guard_matrix_for_svd(avwo_matrix)
                        avwo_key = f"avwo.layer_{layer_idx}"
                        guard_activations[avwo_key] = (
                            guard_activations.get(avwo_key, 0) + (1 if avwo_guard else 0)
                        )

                        # SVD on AVWo: [B, T, D] -> treat as batch of T x D matrices per sequence
                        # For per-step metrics, use the full T x D matrix
                        U_avwo, S_avwo, Vh_avwo = torch.linalg.svd(
                            avwo_clean, full_matrices=False
                        )
                        avwo_metrics = compute_all_metrics(
                            S_avwo, U=U_avwo, Vh=Vh_avwo
                        )

                        for metric_name, metric_val in avwo_metrics.items():
                            key = f"avwo.layer_{layer_idx}.{metric_name}"
                            if key in svd_metric_arrays:
                                val = metric_val.mean(dim=-1) if metric_val.dim() > 1 else metric_val
                                vals_np = val.cpu().numpy()
                                for b_idx in range(B_actual):
                                    svd_metric_arrays[key][
                                        batch_start + b_idx, step
                                    ] = vals_np[b_idx] if vals_np.ndim > 0 else vals_np.item()

                        # AVWo Grassmannian distance
                        avwo_u_key = ("avwo", layer_idx)
                        U_avwo_k = U_avwo[:, :, :grassmannian_k] if U_avwo.dim() == 3 else U_avwo[..., :, :grassmannian_k]
                        if u_prev[avwo_u_key] is not None:
                            gdist_avwo = grassmannian_distance(
                                u_prev[avwo_u_key], U_avwo_k, k=grassmannian_k
                            )
                            gdist_avwo_vals = gdist_avwo.mean(dim=-1) if gdist_avwo.dim() > 1 else gdist_avwo
                            gdist_avwo_np = gdist_avwo_vals.cpu().numpy()
                            gkey_avwo = f"avwo.layer_{layer_idx}.grassmannian_distance"
                            for b_idx in range(B_actual):
                                svd_metric_arrays[gkey_avwo][
                                    batch_start + b_idx, step
                                ] = gdist_avwo_np[b_idx] if gdist_avwo_np.ndim > 0 else gdist_avwo_np.item()
                        u_prev[avwo_u_key] = U_avwo_k.clone()

                    # --- WvWo metrics (static, broadcast) ---
                    for layer_idx in range(n_layers):
                        for metric_name, val in wvwo_layer_metrics[layer_idx].items():
                            key = f"wvwo.layer_{layer_idx}.{metric_name}"
                            if key in svd_metric_arrays:
                                for b_idx in range(B_actual):
                                    svd_metric_arrays[key][
                                        batch_start + b_idx, step
                                    ] = val

                        # WvWo Grassmannian distance is NaN (static matrix)
                        gkey_wvwo = f"wvwo.layer_{layer_idx}.grassmannian_distance"
                        # Already NaN from initialization, no action needed

        # After generation: behavioral classification
        gen_np = generated.cpu().numpy()
        for b_idx in range(B_actual):
            seq_len = min(int(target_lengths[b_idx]), gen_np.shape[1])
            all_seq_lengths[batch_start + b_idx] = seq_len
            all_generated[batch_start + b_idx, :seq_len] = gen_np[b_idx, :seq_len]

        # Classify behavioral labels
        edge_valid, rule_outcome, failure_index = classify_steps(
            generated, graph_data, jumper_map
        )

        # Copy into output arrays
        n_steps_classified = edge_valid.shape[1]
        for b_idx in range(B_actual):
            seq_len = int(target_lengths[b_idx])
            copy_len = min(n_steps_classified, seq_len - 1, max_steps - 1)
            all_edge_valid[batch_start + b_idx, :copy_len] = edge_valid[b_idx, :copy_len]
            all_rule_outcome[batch_start + b_idx, :copy_len] = rule_outcome[b_idx, :copy_len]
            all_failure_index[batch_start + b_idx] = failure_index[b_idx]

    return EvaluationResult(
        generated=all_generated,
        edge_valid=all_edge_valid,
        rule_outcome=all_rule_outcome,
        failure_index=all_failure_index,
        svd_metrics=svd_metric_arrays,
        guard_activations=guard_activations,
        sequence_lengths=all_seq_lengths,
    )


def save_evaluation_results(
    result: EvaluationResult,
    output_dir: str | Path,
    split_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    """Save evaluation results to NPZ and return summary dict for result.json.

    Writes token_metrics.npz containing all SVD metric arrays, behavioral
    arrays, and failure_index. Returns a summary dict with aggregate statistics.

    Args:
        result: EvaluationResult from fused_evaluate.
        output_dir: Directory to write token_metrics.npz into.
        split_labels: Optional array of split assignments ('exploratory' or
            'confirmatory') per walk. When provided, split data is stored in
            NPZ and split metadata is included in the summary dict.
            See docs/pre-registration.md Section 5.

    Returns:
        Summary dict suitable for inclusion in result.json metrics section.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build NPZ contents
    npz_data: dict[str, np.ndarray] = {}

    # SVD metric arrays
    npz_data.update(result.svd_metrics)

    # Behavioral arrays
    npz_data["edge_valid"] = result.edge_valid
    npz_data["rule_outcome"] = result.rule_outcome
    npz_data["failure_index"] = result.failure_index
    npz_data["sequence_lengths"] = result.sequence_lengths
    npz_data["generated"] = result.generated

    # Held-out split labels (Phase 11: Pre-Registration Framework)
    if split_labels is not None:
        # Store as integer array: 0=exploratory, 1=confirmatory
        split_int = np.where(split_labels == "confirmatory", 1, 0).astype(np.int8)
        npz_data["split"] = split_int

    # Write NPZ
    npz_path = output_path / "token_metrics.npz"
    np.savez_compressed(str(npz_path), **npz_data)

    # Compute aggregate summary
    summary: dict[str, Any] = {"scalars": {}}

    # Per-metric mean/std (across valid, non-NaN positions)
    metric_stats: dict[str, dict[str, float]] = {}
    for key, arr in result.svd_metrics.items():
        valid_mask = ~np.isnan(arr)
        if valid_mask.any():
            metric_stats[key] = {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
            }
        else:
            metric_stats[key] = {"mean": float("nan"), "std": float("nan")}

    summary["scalars"]["svd_metric_stats"] = metric_stats
    summary["scalars"]["guard_activations"] = result.guard_activations
    summary["scalars"]["failure_index_list"] = result.failure_index.tolist()
    summary["scalars"]["n_sequences"] = int(result.sequence_lengths.shape[0])
    summary["scalars"]["n_violations"] = int((result.failure_index >= 0).sum())

    # Split assignment metadata
    if split_labels is not None:
        n_exploratory = int((split_labels == "exploratory").sum())
        n_confirmatory = int((split_labels == "confirmatory").sum())
        violation_mask = result.failure_index >= 0
        summary["scalars"]["split_assignment"] = {
            "split_seed": SPLIT_SEED,
            "n_exploratory": n_exploratory,
            "n_confirmatory": n_confirmatory,
            "n_exploratory_violations": int(
                (violation_mask & (split_labels == "exploratory")).sum()
            ),
            "n_confirmatory_violations": int(
                (violation_mask & (split_labels == "confirmatory")).sum()
            ),
        }

    return summary
