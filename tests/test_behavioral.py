"""Unit tests for behavioral classification: edge validity, rule compliance, failure_index.

Uses small synthetic graph fixtures with known edges and jumper configurations
to verify deterministic classification outcomes.
"""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.evaluation.behavioral import RuleOutcome, classify_steps
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_graph():
    """5-vertex directed graph with known edges.

    Edges: 0->1, 0->2, 1->2, 1->3, 2->3, 2->4, 3->0, 3->4, 4->0, 4->1
    Self-loops: 0->0, 1->1, 2->2, 3->3, 4->4
    Blocks: [0, 0, 1, 1, 2]
    """
    n = 5
    K = 3
    block_assignments = np.array([0, 0, 1, 1, 2], dtype=np.int32)

    rows = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    cols = [0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4, 0, 1, 4]
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

    return GraphData(
        adjacency=adjacency,
        block_assignments=block_assignments,
        theta=np.ones(n),
        n=n,
        K=K,
        block_size=2,  # approximate
        generation_seed=42,
        attempt=0,
    )


@pytest.fixture
def jumper_map():
    """Vertex 0 is a jumper: r=2, target_block=1 (vertices 2, 3)."""
    return {
        0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2),
    }


@pytest.fixture
def no_jumpers():
    """Empty jumper map."""
    return {}


# ---------------------------------------------------------------------------
# TestEdgeValidity
# ---------------------------------------------------------------------------
class TestEdgeValidity:
    """Edge validity: checked against CSR adjacency at every step."""

    def test_all_valid_edges(self, small_graph, no_jumpers):
        """Sequence following only valid edges: all edge_valid=True."""
        # Path: 0 -> 1 -> 2 -> 3 -> 4 (all valid edges)
        generated = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        edge_valid, _, _ = classify_steps(generated, small_graph, no_jumpers)
        assert edge_valid.shape == (1, 4)
        assert np.all(edge_valid[0] == True)  # noqa: E712

    def test_one_invalid_edge(self, small_graph, no_jumpers):
        """Sequence with one invalid edge: that step edge_valid=False."""
        # Path: 0 -> 1 -> 4 -> 0 -> 2
        # 0->1 valid, 1->4 INVALID (1 connects to 2,3 not 4), 4->0 valid, 0->2 valid
        generated = torch.tensor([[0, 1, 4, 0, 2]], dtype=torch.long)
        edge_valid, _, _ = classify_steps(generated, small_graph, no_jumpers)
        assert edge_valid[0, 0] == True  # 0->1 valid
        assert edge_valid[0, 1] == False  # 1->4 invalid
        assert edge_valid[0, 2] == True  # 4->0 valid
        assert edge_valid[0, 3] == True  # 0->2 valid

    def test_all_invalid_edges(self, small_graph, no_jumpers):
        """Sequence where every edge is invalid."""
        # Path: 1 -> 0 -> 3 -> 2 -> 0
        # 1->0 INVALID (1 connects to 1,2,3 not 0), 0->3 INVALID (0 connects to 0,1,2)
        # 3->2 INVALID (3 connects to 0,3,4), 2->0 INVALID (2 connects to 2,3,4)
        generated = torch.tensor([[1, 0, 3, 2, 0]], dtype=torch.long)
        edge_valid, _, _ = classify_steps(generated, small_graph, no_jumpers)
        assert not np.any(edge_valid[0])

    def test_single_step_sequence(self, small_graph, no_jumpers):
        """Single-step sequence (2 tokens): one edge check."""
        generated = torch.tensor([[0, 1]], dtype=torch.long)
        edge_valid, _, _ = classify_steps(generated, small_graph, no_jumpers)
        assert edge_valid.shape == (1, 1)
        assert edge_valid[0, 0] == True


# ---------------------------------------------------------------------------
# TestRuleCompliance
# ---------------------------------------------------------------------------
class TestRuleCompliance:
    """Rule compliance: jumper encounters tracked, deadlines checked."""

    def test_rule_followed(self, small_graph, jumper_map):
        """Encounter jumper vertex 0, at step+r land in correct block -> FOLLOWED."""
        # Jumper at vertex 0, r=2, target_block=1 (vertices 2, 3)
        # Path: 0 -> 1 -> 2 -> 3 -> 4
        # Step 0: vertex 0 is jumper, deadline at step 0+2=2
        # Step 2: v = token at index 2+1=3 which is vertex 3, block_assignments[3]=1 == target_block=1 -> FOLLOWED
        # Wait - let's be precise about the indexing:
        # Step t=0: u=0 (jumper), v=1, constraint added: deadline = 0 + r = 2 (meaning at step t+1=2, check block of v)
        # Step t=1: u=1, v=2, check if t+1=2 == deadline? Yes! v=2, block_assignments[2]=1 == 1 -> FOLLOWED
        # Actually, let's re-read: deadline means at step t where t+1 == deadline:
        # constraint is (deadline_step=0+2=2, target_block=1)
        # At step t=1: t+1=2 == deadline, v=2, block[2]=1 -> FOLLOWED
        generated = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, small_graph, jumper_map)
        # Step 0: NOT_APPLICABLE (no deadline resolves, even though jumper encountered)
        # Step 1: FOLLOWED (deadline 2 resolves, vertex 2 is in block 1)
        assert rule_outcome[0, 0] == RuleOutcome.NOT_APPLICABLE
        assert rule_outcome[0, 1] == RuleOutcome.FOLLOWED

    def test_rule_violated(self, small_graph, jumper_map):
        """Encounter jumper, at step+r land in wrong block -> VIOLATED."""
        # Jumper at vertex 0, r=2, target_block=1
        # Path: 0 -> 1 -> 2 -> 4 -> 0
        # Step 0: u=0 (jumper), constraint: deadline=2
        # Step 1: t+1=2 == deadline, v=2, block[2]=1 -> FOLLOWED
        # Need a case where the arrival is in wrong block.
        # Path: 0 -> 0 -> 0 -> ...
        # Step 0: u=0 (jumper), v=0, constraint deadline=2
        # Step 1: t+1=2 == deadline, v=0, block[0]=0 != 1 -> VIOLATED
        generated = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, small_graph, jumper_map)
        assert rule_outcome[0, 1] == RuleOutcome.VIOLATED

    def test_no_jumper_encounter(self, small_graph, jumper_map):
        """Sequence with no jumper encounter: all NOT_APPLICABLE."""
        # Path: 1 -> 2 -> 3 -> 4 -> 4 (vertex 0 never visited)
        generated = torch.tensor([[1, 2, 3, 4, 4]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, small_graph, jumper_map)
        assert np.all(rule_outcome[0] == RuleOutcome.NOT_APPLICABLE)

    def test_multiple_jumper_encounters(self, small_graph):
        """Multiple jumper encounters in same sequence: each checked independently."""
        # Two jumpers: vertex 0 (r=2, target=1) and vertex 1 (r=3, target=2)
        jmap = {
            0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2),
            1: JumperInfo(vertex_id=1, source_block=0, target_block=2, r=3),
        }
        # Path: 0 -> 1 -> 2 -> 3 -> 4 -> 0 -> 1 -> 2
        # Step 0: u=0 jumper, deadline=2
        # Step 1: u=1 jumper, deadline=4; also t+1=2==deadline(0), v=2, block[2]=1==target -> FOLLOWED
        # Step 4: t+1=5? No, deadline(1)=4, at step t=3: t+1=4==deadline, v=4, block[4]=2==target -> FOLLOWED
        generated = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, small_graph, jmap)
        assert rule_outcome[0, 1] == RuleOutcome.FOLLOWED  # deadline from vertex 0
        assert rule_outcome[0, 3] == RuleOutcome.FOLLOWED  # deadline from vertex 1

    def test_jumper_too_late_for_resolution(self, small_graph, jumper_map):
        """Jumper encountered too late: step+r >= seq_len -> stays NOT_APPLICABLE."""
        # Short sequence, jumper at vertex 0 with r=2
        # Path: 1 -> 0 -> 1 (only 3 tokens, 2 steps)
        # Step 1: u=0 jumper, deadline=1+2=3, but seq only has 2 steps (indices 0,1)
        # Deadline 3 never checked -> NOT_APPLICABLE
        generated = torch.tensor([[1, 0, 1]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, small_graph, jumper_map)
        # Step 0: NOT_APPLICABLE (u=1, not a jumper)
        # Step 1: NOT_APPLICABLE (jumper encountered but deadline unreachable)
        assert np.all(rule_outcome[0] == RuleOutcome.NOT_APPLICABLE)


# ---------------------------------------------------------------------------
# TestFailureIndex
# ---------------------------------------------------------------------------
class TestFailureIndex:
    """failure_index: first step where rule_outcome == VIOLATED."""

    def test_fully_correct_sequence(self, small_graph, jumper_map):
        """Fully correct sequence: failure_index = -1."""
        # No violation path
        generated = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        _, _, failure_index = classify_steps(generated, small_graph, jumper_map)
        assert failure_index[0] == -1

    def test_violation_failure_index(self, small_graph, jumper_map):
        """Rule violation at deadline step: failure_index = deadline step."""
        # Path: 0 -> 0 -> 0 -> 0 -> 0
        # Step 0: u=0 jumper, deadline=2
        # Step 1: t+1=2 == deadline, v=0, block[0]=0 != 1 -> VIOLATED
        # failure_index = t+1 = 2 (the deadline step)
        generated = torch.tensor([[0, 0, 0, 0, 0]], dtype=torch.long)
        _, _, failure_index = classify_steps(generated, small_graph, jumper_map)
        # The failure_index records the step where the violation is detected
        # Based on plan: "failure_index records the first step where rule_outcome==VIOLATED"
        # The violation happens at step t=1 (the transition being classified)
        assert failure_index[0] >= 0  # There IS a failure
        assert failure_index[0] == 1  # Violation detected at step t=1

    def test_multiple_violations_first_recorded(self, small_graph):
        """Multiple violations: failure_index = first one."""
        # Jumper at 0 (r=2, target=1) and 1 (r=2, target=2)
        jmap = {
            0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2),
            1: JumperInfo(vertex_id=1, source_block=0, target_block=2, r=2),
        }
        # Path: 0 -> 1 -> 0 -> 0 -> 0
        # Step 0: u=0 jumper, deadline=2
        # Step 1: u=1 jumper, deadline=3; t+1=2==deadline(0), v=0, block[0]=0 != 1 -> VIOLATED (first)
        # Step 2: t+1=3==deadline(1), v=0, block[0]=0 != 2 -> VIOLATED
        generated = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.long)
        _, _, failure_index = classify_steps(generated, small_graph, jmap)
        assert failure_index[0] == 1  # First violation at step 1

    def test_edge_invalidity_no_failure_index(self, small_graph, jumper_map):
        """Edge invalidity does NOT affect failure_index (rule violations only)."""
        # Path with invalid edge but no rule violation
        # 1 -> 0 -> 1 -> 2 -> 3 (1->0 is INVALID edge, but no jumper rules triggered)
        generated = torch.tensor([[1, 0, 1, 2, 3]], dtype=torch.long)
        edge_valid, _, failure_index = classify_steps(generated, small_graph, jumper_map)
        assert edge_valid[0, 0] == False  # 1->0 is invalid
        assert failure_index[0] == -1  # No rule violation

    def test_no_jumper_encounters(self, small_graph, no_jumpers):
        """No jumper encounters: failure_index = -1."""
        generated = torch.tensor([[1, 2, 3, 4, 4]], dtype=torch.long)
        _, _, failure_index = classify_steps(generated, small_graph, no_jumpers)
        assert failure_index[0] == -1


# ---------------------------------------------------------------------------
# TestBatchedClassification
# ---------------------------------------------------------------------------
class TestBatchedClassification:
    """Batch processing across multiple sequences."""

    def test_batch_of_three(self, small_graph, jumper_map):
        """classify_steps with batch of 3 sequences."""
        generated = torch.tensor([
            [0, 1, 2, 3, 4],  # correct (jumper at 0, r=2, arrives at block 1)
            [0, 0, 0, 0, 0],  # violation (jumper at 0, r=2, stays in block 0)
            [1, 2, 3, 4, 4],  # no jumper encounter
        ], dtype=torch.long)
        edge_valid, rule_outcome, failure_index = classify_steps(
            generated, small_graph, jumper_map
        )
        assert edge_valid.shape == (3, 4)
        assert rule_outcome.shape == (3, 4)
        assert failure_index.shape == (3,)

    def test_batch_mixed_outcomes(self, small_graph, jumper_map):
        """Mixed batch: one correct, one with violation, one with no jumpers."""
        generated = torch.tensor([
            [0, 1, 2, 3, 4],  # FOLLOWED at step 1
            [0, 0, 0, 0, 0],  # VIOLATED at step 1
            [1, 2, 3, 4, 4],  # no jumper encounter
        ], dtype=torch.long)
        _, rule_outcome, failure_index = classify_steps(
            generated, small_graph, jumper_map
        )
        assert failure_index[0] == -1  # No violation
        assert failure_index[1] >= 0  # Has violation
        assert failure_index[2] == -1  # No jumper encounter


# ---------------------------------------------------------------------------
# TestContinuationAfterViolation
# ---------------------------------------------------------------------------
class TestContinuationAfterViolation:
    """Generation continues classifying after first violation (no early stopping)."""

    def test_continues_after_violation(self, small_graph):
        """After violation, subsequent jumper encounters still classified."""
        jmap = {
            0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2),
        }
        # Path: 0 -> 0 -> 0 -> 0 -> 1 -> 2 -> 0 -> 0 -> 0
        # Step 0: u=0 jumper, deadline=2
        # Step 1: t+1=2==deadline, v=0, block[0]=0 != 1 -> VIOLATED (failure_index=1)
        # Step 2: u=0 jumper again, deadline=4
        # Step 3: t+1=4==deadline, v=1, block[1]=0 != 1 -> VIOLATED (classification continues!)
        # (Also step 4: u=1 not jumper, step 5: u=2 not jumper)
        # Step 5: u=0 jumper, deadline=7
        # Step 6: t+1=7==deadline, v=0, block[0]=0 != 1 -> VIOLATED
        generated = torch.tensor([[0, 0, 0, 0, 1, 2, 0, 0, 0]], dtype=torch.long)
        _, rule_outcome, failure_index = classify_steps(generated, small_graph, jmap)
        # First violation at step 1
        assert failure_index[0] == 1
        # But classification CONTINUES: step 3 also has a violation
        assert rule_outcome[0, 3] == RuleOutcome.VIOLATED
        # And step 7 (index 7) also has violation
        assert rule_outcome[0, 7] == RuleOutcome.VIOLATED

    def test_violation_then_followed(self, small_graph):
        """Violation followed by a correct jumper encounter."""
        jmap = {
            0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2),
        }
        # Path: 0 -> 0 -> 0 -> 0 -> 0 -> 0 -> 2 (len=7)
        # Step 0: u=0 jumper, deadline=2
        # Step 1: t+1=2, v=0, block[0]=0 != 1 -> VIOLATED
        # Step 2: u=0 jumper, deadline=4
        # Step 3: t+1=4, v=0, block[0]=0 != 1 -> VIOLATED
        # Step 4: u=0 jumper, deadline=6
        # Step 5: t+1=6, v=2, block[2]=1 == 1 -> FOLLOWED!
        generated = torch.tensor([[0, 0, 0, 0, 0, 0, 2]], dtype=torch.long)
        _, rule_outcome, failure_index = classify_steps(generated, small_graph, jmap)
        assert failure_index[0] == 1  # First violation
        assert rule_outcome[0, 1] == RuleOutcome.VIOLATED
        assert rule_outcome[0, 5] == RuleOutcome.FOLLOWED  # Later success
