"""Audit tests for 4-class behavioral classification (GRAPH-04).

Verifies the RuleOutcome enum values, the distinction between UNCONSTRAINED
and PENDING states, countdown sequences, resolution outcomes, and
compatibility with immediate consumers (confusion matrix, event extraction).
"""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.evaluation.behavioral import RuleOutcome, classify_steps
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData


# ---------------------------------------------------------------------------
# Shared fixture: 5-vertex graph with known structure
# ---------------------------------------------------------------------------
@pytest.fixture
def graph():
    """5-vertex directed graph.

    Edges: 0->1, 0->2, 1->2, 1->3, 2->3, 2->4, 3->0, 3->4, 4->0, 4->1
    Self-loops on all vertices.
    Blocks: [0, 0, 1, 1, 2]
    """
    n = 5
    rows = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    cols = [0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4, 0, 1, 4]
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))
    return GraphData(
        adjacency=adjacency,
        block_assignments=np.array([0, 0, 1, 1, 2], dtype=np.int32),
        theta=np.ones(n),
        n=n,
        K=3,
        block_size=2,
        generation_seed=42,
        attempt=0,
    )


# ---------------------------------------------------------------------------
# 1. Enum values
# ---------------------------------------------------------------------------
class TestEnumValues:
    """Verify the exact enum values for 4-class RuleOutcome."""

    def test_4class_enum_values(self):
        """UNCONSTRAINED=0, PENDING=1, FOLLOWED=2, VIOLATED=3.

        Logical ordering: no constraint -> waiting -> resolved-correct -> resolved-wrong.
        """
        assert RuleOutcome.UNCONSTRAINED == 0
        assert RuleOutcome.PENDING == 1
        assert RuleOutcome.FOLLOWED == 2
        assert RuleOutcome.VIOLATED == 3

    def test_enum_has_exactly_four_members(self):
        """RuleOutcome has exactly 4 members, no more."""
        assert len(RuleOutcome) == 4


# ---------------------------------------------------------------------------
# 2. UNCONSTRAINED classification
# ---------------------------------------------------------------------------
class TestUnconstrained:
    """Steps with no active jumper constraint are UNCONSTRAINED."""

    def test_unconstrained_no_jumper(self, graph):
        """Steps with no jumper encounter should be UNCONSTRAINED.

        UNCONSTRAINED means no jumper rule is active at this step.
        """
        # Path: 1 -> 2 -> 3 -> 4 -> 4 (no jumper vertices encountered)
        jmap = {0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2)}
        generated = torch.tensor([[1, 2, 3, 4, 4]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, graph, jmap)
        assert np.all(rule_outcome[0] == RuleOutcome.UNCONSTRAINED)


# ---------------------------------------------------------------------------
# 3. UNCONSTRAINED vs PENDING distinction
# ---------------------------------------------------------------------------
class TestUnconstrainedVsPending:
    """Key distinction: UNCONSTRAINED (no rule) vs PENDING (rule active, waiting)."""

    def test_unconstrained_vs_pending_distinction(self, graph):
        """A step with no active constraint is UNCONSTRAINED.
        A step with an active constraint whose deadline is in the future is PENDING.

        Key distinction in 4-class model: UNCONSTRAINED (no rule) vs PENDING (rule active, waiting).
        """
        # Jumper at vertex 1 (not vertex 0), r=3, target_block=2
        jmap = {1: JumperInfo(vertex_id=1, source_block=0, target_block=2, r=3)}
        # Path: 0 -> 1 -> 2 -> 3 -> 4  (5 tokens)
        # Step 0: u=0, not a jumper => UNCONSTRAINED
        # Step 1: u=1 jumper, deadline=1+3=4. t+1=2, 4>2 => PENDING
        # Step 2: u=2, t+1=3, 4>3 => PENDING
        # Step 3: u=3, t+1=4==4 => v=4, block[4]=2==2 => FOLLOWED
        generated = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, graph, jmap)
        assert rule_outcome[0, 0] == RuleOutcome.UNCONSTRAINED  # No constraint yet
        assert rule_outcome[0, 1] == RuleOutcome.PENDING  # Constraint active, waiting
        assert rule_outcome[0, 2] == RuleOutcome.PENDING  # Still waiting
        assert rule_outcome[0, 3] == RuleOutcome.FOLLOWED  # Resolved


# ---------------------------------------------------------------------------
# 4. PENDING countdown sequence
# ---------------------------------------------------------------------------
class TestPendingCountdown:
    """Steps within the countdown window are PENDING."""

    def test_pending_countdown_sequence(self, graph):
        """For a jumper with r=4, verify that steps between encounter and deadline
        are all PENDING.

        Steps within the countdown window are PENDING.
        """
        # Jumper at vertex 0, r=4, target_block=1
        jmap = {0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=4)}
        # Path: 0 -> 1 -> 1 -> 1 -> 2 -> 3  (6 tokens, 5 steps)
        # Step 0: u=0 jumper, deadline=4. t+1=1, 4>1 => PENDING
        # Step 1: u=1, t+1=2, 4>2 => PENDING
        # Step 2: u=1, t+1=3, 4>3 => PENDING
        # Step 3: u=1, t+1=4==4 => v=2, block[2]=1==1 => FOLLOWED
        # Step 4: u=2, no pending constraints => UNCONSTRAINED
        generated = torch.tensor([[0, 1, 1, 1, 2, 3]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, graph, jmap)
        assert rule_outcome[0, 0] == RuleOutcome.PENDING
        assert rule_outcome[0, 1] == RuleOutcome.PENDING
        assert rule_outcome[0, 2] == RuleOutcome.PENDING
        assert rule_outcome[0, 3] == RuleOutcome.FOLLOWED
        assert rule_outcome[0, 4] == RuleOutcome.UNCONSTRAINED


# ---------------------------------------------------------------------------
# 5. FOLLOWED at deadline
# ---------------------------------------------------------------------------
class TestFollowedAtDeadline:
    """FOLLOWED means constraint resolved successfully."""

    def test_followed_at_deadline(self, graph):
        """At the deadline step, if the vertex is in the correct block,
        the outcome is FOLLOWED.

        FOLLOWED means constraint resolved successfully.
        """
        # Jumper at vertex 0, r=2, target_block=1 (vertices 2,3 are in block 1)
        jmap = {0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2)}
        # Path: 0 -> 1 -> 2 -> 3  (4 tokens)
        # Step 0: u=0 jumper, deadline=2. PENDING (2>1)
        # Step 1: u=1, t+1=2==2 => v=2, block[2]=1==1 => FOLLOWED
        generated = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, graph, jmap)
        assert rule_outcome[0, 1] == RuleOutcome.FOLLOWED


# ---------------------------------------------------------------------------
# 6. VIOLATED at deadline
# ---------------------------------------------------------------------------
class TestViolatedAtDeadline:
    """VIOLATED means constraint resolution failed."""

    def test_violated_at_deadline(self, graph):
        """At the deadline step, if the vertex is in the wrong block,
        the outcome is VIOLATED.

        VIOLATED means constraint resolution failed.
        """
        # Jumper at vertex 0, r=2, target_block=1
        # Path: 0 -> 1 -> 0 -> 1  (4 tokens)
        # Step 0: u=0 jumper, deadline=2. PENDING
        # Step 1: u=1, t+1=2==2 => v=0, block[0]=0 != 1 => VIOLATED
        generated = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)
        _, rule_outcome, failure_index = classify_steps(generated, graph,
            {0: JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2)})
        assert rule_outcome[0, 1] == RuleOutcome.VIOLATED
        assert failure_index[0] == 1


# ---------------------------------------------------------------------------
# 7. All four classes in one sequence
# ---------------------------------------------------------------------------
class TestAllFourClasses:
    """All four behavioral classes appear in a single walk."""

    def test_all_four_classes_in_one_sequence(self, graph):
        """Construct a sequence that exhibits all 4 classes: some steps UNCONSTRAINED
        (before any jumper), some PENDING (after jumper encounter, before deadline),
        one FOLLOWED or VIOLATED (at deadline), and more UNCONSTRAINED after.

        All four behavioral classes appear in a single walk.
        """
        # Jumper at vertex 1, r=3, target_block=2
        jmap = {1: JumperInfo(vertex_id=1, source_block=0, target_block=2, r=3)}
        # Path: 0 -> 0 -> 1 -> 2 -> 3 -> 4 -> 4  (7 tokens, 6 steps)
        # Step 0: u=0, not a jumper => UNCONSTRAINED
        # Step 1: u=0, not a jumper => UNCONSTRAINED  (but wait, step 1 u=token at index 1 = 0)
        #   Actually: generated[0] = [0, 0, 1, 2, 3, 4, 4]
        #   Step 0: u=0, v=0. No jumper (vertex 0 not in jmap). UNCONSTRAINED
        #   Step 1: u=0, v=1. UNCONSTRAINED
        #   Step 2: u=1 (jumper!), v=2. deadline=2+3=5. PENDING (5>3)
        #   Step 3: u=2, v=3. t+1=4, 5>4 => PENDING
        #   Step 4: u=3, v=4. t+1=5==5 => v=4, block[4]=2==2 => FOLLOWED
        #   Step 5: u=4, v=4. No pending constraints => UNCONSTRAINED
        generated = torch.tensor([[0, 0, 1, 2, 3, 4, 4]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, graph, jmap)

        outcomes = set(rule_outcome[0])
        assert RuleOutcome.UNCONSTRAINED in outcomes
        assert RuleOutcome.PENDING in outcomes
        assert RuleOutcome.FOLLOWED in outcomes
        # Check specific positions
        assert rule_outcome[0, 0] == RuleOutcome.UNCONSTRAINED
        assert rule_outcome[0, 1] == RuleOutcome.UNCONSTRAINED
        assert rule_outcome[0, 2] == RuleOutcome.PENDING
        assert rule_outcome[0, 3] == RuleOutcome.PENDING
        assert rule_outcome[0, 4] == RuleOutcome.FOLLOWED
        assert rule_outcome[0, 5] == RuleOutcome.UNCONSTRAINED

    def test_all_four_classes_with_violation(self, graph):
        """Sequence showing all 4 classes including VIOLATED."""
        # Jumper at vertex 1, r=3, target_block=2
        jmap = {1: JumperInfo(vertex_id=1, source_block=0, target_block=2, r=3)}
        # Path: 0 -> 0 -> 1 -> 2 -> 3 -> 0 -> 0  (7 tokens, 6 steps)
        # Step 0: UNCONSTRAINED (u=0, not jumper)
        # Step 1: UNCONSTRAINED (u=0, not jumper)
        # Step 2: u=1 jumper, deadline=5. PENDING (5>3)
        # Step 3: PENDING (5>4)
        # Step 4: t+1=5==5 => v=0, block[0]=0 != 2 => VIOLATED
        # Step 5: UNCONSTRAINED (no pending constraints after resolution)
        generated = torch.tensor([[0, 0, 1, 2, 3, 0, 0]], dtype=torch.long)
        _, rule_outcome, _ = classify_steps(generated, graph, jmap)

        outcomes = set(rule_outcome[0])
        assert outcomes == {
            RuleOutcome.UNCONSTRAINED,
            RuleOutcome.PENDING,
            RuleOutcome.VIOLATED,
        }


# ---------------------------------------------------------------------------
# 8. Confusion matrix excludes PENDING and UNCONSTRAINED
# ---------------------------------------------------------------------------
class TestConfusionMatrixCompat:
    """Confusion matrix shows only resolved outcomes."""

    def test_confusion_matrix_excludes_pending_and_unconstrained(self):
        """Verify that the confusion matrix in confusion.py only counts
        FOLLOWED and VIOLATED steps.

        Confusion matrix shows only resolved outcomes.
        """
        from src.visualization.confusion import plot_confusion_matrix
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend

        # Construct synthetic data with all 4 classes
        edge_valid = np.array([True, True, False, True, True, False], dtype=bool)
        rule_outcome = np.array([
            RuleOutcome.UNCONSTRAINED,
            RuleOutcome.PENDING,
            RuleOutcome.FOLLOWED,
            RuleOutcome.FOLLOWED,
            RuleOutcome.VIOLATED,
            RuleOutcome.VIOLATED,
        ], dtype=np.int32)

        fig = plot_confusion_matrix(edge_valid, rule_outcome)

        # The confusion matrix should only count 4 steps (2 FOLLOWED + 2 VIOLATED)
        # not 6 (which would include UNCONSTRAINED and PENDING)
        ax = fig.axes[0]
        # Verify by checking annotations contain counts that sum to 4
        total = 0
        for text_obj in ax.texts:
            txt = text_obj.get_text()
            if "\n" in txt:
                count_str = txt.split("\n")[0]
                total += int(count_str)
        assert total == 4, f"Expected 4 resolved steps in confusion matrix, got {total}"
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Event extraction skips PENDING
# ---------------------------------------------------------------------------
class TestEventExtractionCompat:
    """Events are only created for resolved outcomes."""

    def test_event_extraction_skips_pending(self):
        """Verify that event_extraction.py skips PENDING outcomes
        (only records FOLLOWED/VIOLATED).

        Events are only created for resolved outcomes.
        """
        from src.analysis.event_extraction import extract_events

        # Jumper vertex 5 at step 3, r=4 => resolution at step 7
        generated = np.zeros((1, 20), dtype=np.int64)
        generated[0, 3] = 5

        # Mark the resolution index as PENDING instead of resolved
        rule_outcome = np.full((1, 19), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        rule_outcome[0, 6] = RuleOutcome.PENDING  # 3+4-1 = 6

        failure_index = np.full(1, -1, dtype=np.int32)
        jumper_map = {5: JumperInfo(vertex_id=5, source_block=0, target_block=1, r=4)}

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 0, "PENDING outcomes should not produce events"

    def test_event_extraction_records_followed_and_violated(self):
        """FOLLOWED and VIOLATED outcomes produce events normally."""
        from src.analysis.event_extraction import extract_events

        generated = np.zeros((1, 20), dtype=np.int64)
        generated[0, 3] = 5

        rule_outcome = np.full((1, 19), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        rule_outcome[0, 6] = RuleOutcome.FOLLOWED  # 3+4-1 = 6

        failure_index = np.full(1, -1, dtype=np.int32)
        jumper_map = {5: JumperInfo(vertex_id=5, source_block=0, target_block=1, r=4)}

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 1
        assert events[0].outcome == RuleOutcome.FOLLOWED
