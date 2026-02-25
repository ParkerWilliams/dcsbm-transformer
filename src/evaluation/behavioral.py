"""Behavioral classification for generated sequences.

Classifies each generation step into one of 4 classes based on
edge validity (against DCSBM adjacency) and rule compliance
(against jumper constraints). Produces failure_index annotation
marking the first rule violation per sequence.
"""

from enum import IntEnum

import numpy as np
import torch

from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData


class RuleOutcome(IntEnum):
    """Outcome of rule compliance check at a generation step.

    NOT_APPLICABLE: No jumper rule deadline at this step.
    FOLLOWED: Jumper rule deadline met, correct block reached.
    VIOLATED: Jumper rule deadline met, wrong block reached.
    """

    NOT_APPLICABLE = 0
    FOLLOWED = 1
    VIOLATED = 2


def classify_steps(
    generated: torch.Tensor,
    graph_data: GraphData,
    jumper_map: dict[int, JumperInfo],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify all steps in all sequences for edge validity and rule compliance.

    For each consecutive pair of tokens (u, v) at step t:
    - Edge validity: True if v is a neighbor of u in the CSR adjacency matrix.
    - Rule compliance: If u is a jumper vertex, a constraint is added with
      deadline = t + jumper.r. When step t+1 == deadline, the block of v is
      checked against the target block. FOLLOWED if match, VIOLATED otherwise.
    - failure_index: The first step t where rule_outcome[t] == VIOLATED, or -1.

    Generation continues after violations (no early stopping).

    Args:
        generated: Generated token sequences of shape [B, L].
        graph_data: Graph data with adjacency matrix and block assignments.
        jumper_map: Mapping from vertex_id to JumperInfo.

    Returns:
        Tuple of (edge_valid, rule_outcome, failure_index):
        - edge_valid: bool array [B, L-1]
        - rule_outcome: int array [B, L-1] (RuleOutcome values)
        - failure_index: int array [B] (-1 for no failure)
    """
    seqs = generated.cpu().numpy()
    B, L = seqs.shape

    edge_valid = np.zeros((B, L - 1), dtype=bool)
    rule_outcome = np.full((B, L - 1), RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
    failure_index = np.full(B, -1, dtype=np.int32)

    indptr = graph_data.adjacency.indptr
    indices = graph_data.adjacency.indices
    block_assignments = graph_data.block_assignments

    for b in range(B):
        # Active constraints: list of (deadline_step, target_block) tuples
        active_constraints: list[tuple[int, int]] = []

        for t in range(L - 1):
            u = int(seqs[b, t])
            v = int(seqs[b, t + 1])

            # Edge validity check via CSR lookup
            neighbors = indices[indptr[u] : indptr[u + 1]]
            edge_valid[b, t] = v in neighbors

            # Track jumper encounters from the generated path
            if u in jumper_map:
                j = jumper_map[u]
                active_constraints.append((t + j.r, j.target_block))

            # Check rule deadlines: t+1 is the position of v
            for deadline, target_block in active_constraints:
                if t + 1 == deadline:
                    actual_block = int(block_assignments[v])
                    if actual_block == target_block:
                        rule_outcome[b, t] = RuleOutcome.FOLLOWED
                    else:
                        rule_outcome[b, t] = RuleOutcome.VIOLATED
                        if failure_index[b] == -1:
                            failure_index[b] = t
                    break  # Only one constraint resolves per step

    return edge_valid, rule_outcome, failure_index
