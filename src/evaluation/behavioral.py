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
    raise NotImplementedError
