"""Compliance evaluation via greedy generation on held-out walks.

Generates sequences via argmax decoding and computes edge compliance
(fraction of valid edges) and rule compliance (fraction of correct
block arrivals at step+r from jumper vertices).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.config.experiment import ExperimentConfig
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.model.types import ExtractionMode


@dataclass(frozen=True)
class ComplianceResult:
    """Result of compliance evaluation on generated sequences.

    Attributes:
        edge_compliance: Fraction of consecutive token pairs that are valid edges.
        rule_compliance: Fraction of jumper encounters where walk lands in correct
            target block at step+r. 1.0 if no rule checks were possible.
        n_sequences: Number of sequences evaluated.
        n_edge_checks: Total number of edge validity checks performed.
        n_rule_checks: Total number of rule compliance checks performed.
    """

    edge_compliance: float
    rule_compliance: float
    n_sequences: int
    n_edge_checks: int
    n_rule_checks: int


def greedy_generate(
    model: nn.Module,
    start_tokens: torch.Tensor,
    length: int,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate sequences via argmax (greedy) decoding.

    Args:
        model: TransformerLM in eval mode.
        start_tokens: Seed tokens of shape [B, 1].
        length: Total length of generated sequences (including seed).
        max_seq_len: Model's maximum sequence length (context window).
        device: Device for computation.

    Returns:
        Generated sequences of shape [B, length].
    """
    model.eval()
    generated = start_tokens.to(device)

    with torch.no_grad():
        for _ in range(length - start_tokens.shape[1]):
            # Use last max_seq_len tokens for context
            context = generated[:, -max_seq_len:]
            output = model(context, mode=ExtractionMode.NONE)
            # Argmax decoding
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def evaluate_compliance(
    model: nn.Module,
    eval_walks: np.ndarray,
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    device: torch.device,
    n_sequences: int = 1000,
) -> ComplianceResult:
    """Evaluate edge and rule compliance via greedy generation.

    1. Select n_sequences eval walks (or all if fewer).
    2. Use the first token of each as the seed.
    3. Generate walk_length tokens via argmax.
    4. Check edge validity and rule compliance.

    Args:
        model: Trained TransformerLM.
        eval_walks: Evaluation walk array of shape (N, walk_length).
        graph_data: Graph data with adjacency matrix and block assignments.
        jumpers: List of JumperInfo for rule compliance checking.
        config: Experiment config (for walk_length and r).
        device: Device for computation.
        n_sequences: Number of sequences to generate and evaluate.

    Returns:
        ComplianceResult with edge and rule compliance fractions.
    """
    n_available = eval_walks.shape[0]
    n_eval = min(n_sequences, n_available)
    walk_length = config.training.walk_length
    r = config.training.r

    # Extract seed tokens (first token of each eval walk)
    seed_tokens = torch.tensor(
        eval_walks[:n_eval, :1], dtype=torch.long
    )

    # Generate sequences via greedy decoding
    generated = greedy_generate(
        model, seed_tokens, walk_length, model.max_seq_len, device
    )
    seqs = generated.cpu().numpy()

    # Build jumper lookup: vertex_id -> JumperInfo
    jumper_map = {j.vertex_id: j for j in jumpers}

    # CSR arrays for edge checking
    indptr = graph_data.adjacency.indptr
    indices = graph_data.adjacency.indices
    block_assignments = graph_data.block_assignments

    total_edge_checks = 0
    valid_edges = 0
    total_rule_checks = 0
    rule_compliant = 0

    for seq in seqs:
        seq_len = len(seq)
        for t in range(seq_len - 1):
            u = int(seq[t])
            v = int(seq[t + 1])

            # Edge validity check
            neighbors = indices[indptr[u] : indptr[u + 1]]
            total_edge_checks += 1
            if v in neighbors:
                valid_edges += 1

            # Rule compliance check: if u is a jumper, check step t+r
            if u in jumper_map and t + r < seq_len:
                total_rule_checks += 1
                target_block = jumper_map[u].target_block
                arrival_vertex = int(seq[t + r])
                actual_block = int(block_assignments[arrival_vertex])
                if actual_block == target_block:
                    rule_compliant += 1

    edge_compliance = valid_edges / max(1, total_edge_checks)
    rule_compliance = (
        rule_compliant / total_rule_checks if total_rule_checks > 0 else 1.0
    )

    return ComplianceResult(
        edge_compliance=edge_compliance,
        rule_compliance=rule_compliance,
        n_sequences=n_eval,
        n_edge_checks=total_edge_checks,
        n_rule_checks=total_rule_checks,
    )
