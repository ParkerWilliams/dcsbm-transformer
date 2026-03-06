"""Audit tests for event extraction boundary correctness (AUROC-04).

Verifies that extract_events correctly identifies resolution steps from
behavioral labels, that only FOLLOWED/VIOLATED events are yielded, that
is_first_violation is correctly determined, that contamination filtering
applies asymmetrically (only violations contaminate), and that the
cross-module seam between behavioral.py and event_extraction.py is consistent.
"""

import numpy as np
import pytest

from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
)
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


def _make_jumper(vertex_id: int, r: int, source_block: int = 0, target_block: int = 1) -> JumperInfo:
    """Helper to create a JumperInfo with minimal boilerplate."""
    return JumperInfo(vertex_id=vertex_id, source_block=source_block, target_block=target_block, r=r)


# ---------------------------------------------------------------------------
# 1. Outcome filtering: only FOLLOWED and VIOLATED
# ---------------------------------------------------------------------------
class TestExtractEventsOutcomeFiltering:
    """Verify extract_events only yields events for FOLLOWED and VIOLATED outcomes,
    never UNCONSTRAINED or PENDING."""

    def test_only_followed_and_violated_events_extracted(self):
        """Minimal setup: one jumper vertex, test all 4 outcome values.

        Jumper vertex_id=5, r=3, placed at step t=2.
        resolution_step = 2 + 3 = 5, outcome_idx = 4.
        Walk shape: (1, 10), rule_outcome shape: (1, 9)."""
        jumper_map = {5: _make_jumper(5, r=3)}

        # Walk with jumper at position 2
        generated = np.zeros((1, 10), dtype=np.int64)
        generated[0, 2] = 5  # jumper vertex at step t=2

        rule_outcome = np.full((1, 9), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        failure_index = np.full(1, -1, dtype=np.int32)

        # Case 1: FOLLOWED at outcome_idx=4
        rule_outcome[0, 4] = RuleOutcome.FOLLOWED
        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 1
        assert events[0].outcome == RuleOutcome.FOLLOWED

        # Case 2: VIOLATED at outcome_idx=4
        rule_outcome[0, 4] = RuleOutcome.VIOLATED
        failure_index[0] = 4  # first violation at this step
        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 1
        assert events[0].outcome == RuleOutcome.VIOLATED

        # Case 3: PENDING at outcome_idx=4 -> no event
        rule_outcome[0, 4] = RuleOutcome.PENDING
        failure_index[0] = -1
        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 0, "PENDING should not produce an event"

        # Case 4: UNCONSTRAINED at outcome_idx=4 -> no event
        rule_outcome[0, 4] = RuleOutcome.UNCONSTRAINED
        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 0, "UNCONSTRAINED should not produce an event"

    def test_multiple_encounters_in_single_walk(self):
        """Walk with 2 jumper encounters at different positions.
        One resolves FOLLOWED, the other VIOLATED. Both should be extracted."""
        # Jumper A: vertex_id=5, r=3 at step t=1
        # Jumper B: vertex_id=7, r=2 at step t=6
        jumper_map = {
            5: _make_jumper(5, r=3),
            7: _make_jumper(7, r=2),
        }

        generated = np.zeros((1, 15), dtype=np.int64)
        generated[0, 1] = 5  # encounter A at t=1, resolution_step=4, outcome_idx=3
        generated[0, 6] = 7  # encounter B at t=6, resolution_step=8, outcome_idx=7

        rule_outcome = np.full((1, 14), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        rule_outcome[0, 3] = RuleOutcome.FOLLOWED   # encounter A resolves FOLLOWED
        rule_outcome[0, 7] = RuleOutcome.VIOLATED    # encounter B resolves VIOLATED
        failure_index = np.array([7], dtype=np.int32)  # first violation at idx 7

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 2

        # Sort by encounter_step for deterministic checking
        events_sorted = sorted(events, key=lambda e: e.encounter_step)

        assert events_sorted[0].outcome == RuleOutcome.FOLLOWED
        assert events_sorted[0].encounter_step == 1
        assert events_sorted[0].r_value == 3

        assert events_sorted[1].outcome == RuleOutcome.VIOLATED
        assert events_sorted[1].encounter_step == 6
        assert events_sorted[1].r_value == 2

    def test_resolution_step_calculation(self):
        """Verify resolution_step = encounter_step + r.

        Jumper at step t=3, r=4 -> resolution_step should be 7."""
        jumper_map = {10: _make_jumper(10, r=4)}

        generated = np.zeros((1, 15), dtype=np.int64)
        generated[0, 3] = 10  # encounter at t=3

        rule_outcome = np.full((1, 14), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        # outcome_idx = resolution_step - 1 = 7 - 1 = 6
        rule_outcome[0, 6] = RuleOutcome.FOLLOWED
        failure_index = np.full(1, -1, dtype=np.int32)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 1
        assert events[0].encounter_step == 3
        assert events[0].resolution_step == 7, (
            f"resolution_step should be encounter_step + r = 3 + 4 = 7, got {events[0].resolution_step}"
        )
        assert events[0].r_value == 4

    def test_resolution_step_out_of_bounds_skipped(self):
        """When encounter_step + r exceeds rule_outcome array length,
        the encounter should be skipped (no event created).

        Walk length 8 -> rule_outcome has 7 columns (indices 0-6).
        Jumper at t=5, r=4 -> resolution_step=9, outcome_idx=8 > 6 -> skip."""
        jumper_map = {5: _make_jumper(5, r=4)}

        generated = np.zeros((1, 8), dtype=np.int64)
        generated[0, 5] = 5  # encounter at t=5, resolution_step=9 -> out of bounds

        rule_outcome = np.full((1, 7), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        failure_index = np.full(1, -1, dtype=np.int32)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 0, (
            "Encounter with resolution_step beyond rule_outcome bounds should be skipped"
        )


# ---------------------------------------------------------------------------
# 2. is_first_violation flag
# ---------------------------------------------------------------------------
class TestIsFirstViolation:
    """Verify is_first_violation is True only for the first violation in each walk."""

    def test_first_violation_flag_single_violation(self):
        """Walk with exactly one violation. is_first_violation should be True."""
        jumper_map = {5: _make_jumper(5, r=3)}

        generated = np.zeros((1, 10), dtype=np.int64)
        generated[0, 2] = 5  # encounter at t=2, resolution_step=5, outcome_idx=4

        rule_outcome = np.full((1, 9), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        rule_outcome[0, 4] = RuleOutcome.VIOLATED
        failure_index = np.array([4], dtype=np.int32)  # first violation at outcome_idx=4

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 1
        assert events[0].is_first_violation is True

    def test_first_violation_flag_multiple_violations(self):
        """Walk with 2+ violations. Only the event matching failure_index
        should have is_first_violation=True.

        First violation: jumper at t=2, r=3, outcome_idx=4
        Second violation: jumper at t=6, r=2, outcome_idx=7"""
        jumper_map = {
            5: _make_jumper(5, r=3),
            7: _make_jumper(7, r=2),
        }

        generated = np.zeros((1, 15), dtype=np.int64)
        generated[0, 2] = 5  # first encounter at t=2
        generated[0, 6] = 7  # second encounter at t=6

        rule_outcome = np.full((1, 14), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        rule_outcome[0, 4] = RuleOutcome.VIOLATED  # first violation at outcome_idx=4
        rule_outcome[0, 7] = RuleOutcome.VIOLATED  # second violation at outcome_idx=7

        # failure_index points to the FIRST violation's outcome_idx
        failure_index = np.array([4], dtype=np.int32)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 2

        events_sorted = sorted(events, key=lambda e: e.encounter_step)
        # First violation (outcome_idx=4 matches failure_index=4)
        assert events_sorted[0].is_first_violation is True
        # Second violation (outcome_idx=7 does NOT match failure_index=4)
        assert events_sorted[1].is_first_violation is False

    def test_followed_event_never_first_violation(self):
        """FOLLOWED events must always have is_first_violation=False,
        regardless of failure_index value."""
        jumper_map = {5: _make_jumper(5, r=3)}

        generated = np.zeros((1, 10), dtype=np.int64)
        generated[0, 2] = 5  # encounter at t=2, outcome_idx=4

        rule_outcome = np.full((1, 9), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        rule_outcome[0, 4] = RuleOutcome.FOLLOWED

        # Even if failure_index happens to match (shouldn't happen in practice,
        # but test robustness), FOLLOWED must NOT be marked as first violation
        failure_index = np.array([4], dtype=np.int32)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 1
        assert events[0].outcome == RuleOutcome.FOLLOWED
        assert events[0].is_first_violation is False, (
            "FOLLOWED events must never have is_first_violation=True"
        )


# ---------------------------------------------------------------------------
# 3. Contamination filter: asymmetric logic
# ---------------------------------------------------------------------------
class TestContaminationFilter:
    """Per CONTEXT.md locked decisions: only violations contaminate subsequent
    encounters. FOLLOWED events do NOT set last_violation_end."""

    def test_violation_then_nearby_encounter_excluded(self):
        """Violation at encounter_step=5, r=3 (resolution_step=8).
        Second encounter at encounter_step=7 (< 8 = last_violation_end).
        Second event should be EXCLUDED."""
        events = [
            AnalysisEvent(
                walk_idx=0, encounter_step=5, resolution_step=8,
                r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
            ),
            AnalysisEvent(
                walk_idx=0, encounter_step=7, resolution_step=10,
                r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
        ]

        filtered, audit = filter_contaminated_events(events)
        # Only the first event should survive
        assert len(filtered) == 1
        assert filtered[0].encounter_step == 5
        assert audit["excluded_encounters"] == 1

    def test_followed_then_nearby_encounter_NOT_excluded(self):
        """FOLLOWED at encounter_step=5, r=3 (resolution_step=8).
        Second encounter at encounter_step=7 (< 8 but FOLLOWED does NOT
        set last_violation_end). Second event should NOT be excluded.

        This is the asymmetric contamination logic: only violations contaminate."""
        events = [
            AnalysisEvent(
                walk_idx=0, encounter_step=5, resolution_step=8,
                r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
            AnalysisEvent(
                walk_idx=0, encounter_step=7, resolution_step=10,
                r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
        ]

        filtered, audit = filter_contaminated_events(events)
        # Both events should survive (FOLLOWED does NOT contaminate)
        assert len(filtered) == 2
        assert audit["excluded_encounters"] == 0

    def test_violation_with_encounter_outside_window_NOT_excluded(self):
        """Violation at encounter_step=5, r=3 (resolution_step=8).
        Second encounter at encounter_step=9 (>= 8 = last_violation_end).
        Second event should NOT be excluded (encounter is after the violation window)."""
        events = [
            AnalysisEvent(
                walk_idx=0, encounter_step=5, resolution_step=8,
                r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
            ),
            AnalysisEvent(
                walk_idx=0, encounter_step=9, resolution_step=12,
                r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
        ]

        filtered, audit = filter_contaminated_events(events)
        # Both events should survive (second encounter starts after violation window)
        assert len(filtered) == 2
        assert audit["excluded_encounters"] == 0

    def test_violation_boundary_encounter_at_exact_end_NOT_excluded(self):
        """Violation at encounter_step=5, r=3 (resolution_step=8).
        Encounter at encounter_step=8 (== last_violation_end).
        Should NOT be excluded: condition is strict < not <=."""
        events = [
            AnalysisEvent(
                walk_idx=0, encounter_step=5, resolution_step=8,
                r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
            ),
            AnalysisEvent(
                walk_idx=0, encounter_step=8, resolution_step=11,
                r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
        ]

        filtered, audit = filter_contaminated_events(events)
        # encounter_step=8 is NOT < last_violation_end=8, so NOT excluded
        assert len(filtered) == 2
        assert audit["excluded_encounters"] == 0

    def test_contamination_audit_statistics(self):
        """Verify the audit dict returned by filter_contaminated_events
        contains correct counts and per_r breakdown."""
        events = [
            # Walk 0: violation then contaminated encounter
            AnalysisEvent(
                walk_idx=0, encounter_step=2, resolution_step=5,
                r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
            ),
            AnalysisEvent(
                walk_idx=0, encounter_step=4, resolution_step=6,
                r_value=2, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
            # Walk 1: no contamination (FOLLOWED first)
            AnalysisEvent(
                walk_idx=1, encounter_step=2, resolution_step=5,
                r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
            AnalysisEvent(
                walk_idx=1, encounter_step=4, resolution_step=6,
                r_value=2, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
        ]

        filtered, audit = filter_contaminated_events(events)

        assert audit["total_encounters"] == 4
        assert audit["excluded_encounters"] == 1
        assert audit["exclusion_rate"] == pytest.approx(0.25)
        assert audit["flagged"] is False  # 0.25 <= 0.3

        # per_r breakdown
        assert 2 in audit["per_r"]
        assert 3 in audit["per_r"]
        # r=2: 2 total, 1 excluded (walk 0's encounter at t=4)
        assert audit["per_r"][2]["total"] == 2
        assert audit["per_r"][2]["excluded"] == 1
        # r=3: 2 total, 0 excluded
        assert audit["per_r"][3]["total"] == 2
        assert audit["per_r"][3]["excluded"] == 0

    def test_contamination_empty_events(self):
        """filter_contaminated_events with empty input should return empty
        with zeroed audit dict."""
        filtered, audit = filter_contaminated_events([])
        assert filtered == []
        assert audit["total_encounters"] == 0
        assert audit["excluded_encounters"] == 0
        assert audit["exclusion_rate"] == 0.0
        assert audit["flagged"] is False
        assert audit["per_r"] == {}


# ---------------------------------------------------------------------------
# 4. Cross-module seam: behavioral.py -> event_extraction.py
# ---------------------------------------------------------------------------
class TestCrossModuleSeam:
    """End-to-end seam tests verifying the contract between behavioral.py
    classification and event_extraction.py event creation."""

    def test_behavioral_to_event_extraction_seam(self):
        """Construct a minimal scenario with numpy arrays directly:
        - generated with a known jumper vertex at step t
        - rule_outcome with VIOLATED at the expected index (t + r - 1)
        - failure_index set to (t + r - 1) for the walk
        - jumper_map with the jumper's JumperInfo

        Verify the full chain:
        - event.encounter_step == t
        - event.resolution_step == t + r
        - event.outcome == RuleOutcome.VIOLATED
        - event.is_first_violation == True (failure_index matches outcome_idx)."""
        t = 3  # encounter step
        r = 4  # jump length
        resolution_step = t + r  # = 7
        outcome_idx = resolution_step - 1  # = 6

        jumper_map = {20: _make_jumper(20, r=r)}

        generated = np.zeros((1, 15), dtype=np.int64)
        generated[0, t] = 20  # place jumper at step t

        rule_outcome = np.full((1, 14), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        rule_outcome[0, outcome_idx] = RuleOutcome.VIOLATED

        failure_index = np.array([outcome_idx], dtype=np.int32)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)

        assert len(events) == 1
        ev = events[0]
        assert ev.encounter_step == t, f"encounter_step: expected {t}, got {ev.encounter_step}"
        assert ev.resolution_step == resolution_step, (
            f"resolution_step: expected {resolution_step}, got {ev.resolution_step}"
        )
        assert ev.outcome == RuleOutcome.VIOLATED
        assert ev.is_first_violation is True, (
            "failure_index matches outcome_idx, so is_first_violation should be True"
        )

    def test_outcome_index_is_resolution_step_minus_one(self):
        """Explicitly verify that extract_events looks up rule_outcome at
        index resolution_step - 1 (not resolution_step).

        Plant VIOLATED at resolution_step - 1 and UNCONSTRAINED at
        resolution_step (if in bounds). Verify the event outcome is VIOLATED.

        This catches off-by-one errors in the indexing chain."""
        t = 2  # encounter step
        r = 3  # jump length
        resolution_step = t + r  # = 5
        outcome_idx = resolution_step - 1  # = 4

        jumper_map = {15: _make_jumper(15, r=r)}

        generated = np.zeros((1, 12), dtype=np.int64)
        generated[0, t] = 15

        rule_outcome = np.full((1, 11), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        # Plant VIOLATED at outcome_idx = resolution_step - 1 = 4
        rule_outcome[0, outcome_idx] = RuleOutcome.VIOLATED
        # Plant UNCONSTRAINED at resolution_step = 5 (to catch off-by-one)
        rule_outcome[0, resolution_step] = RuleOutcome.UNCONSTRAINED

        failure_index = np.array([outcome_idx], dtype=np.int32)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)

        # Should find VIOLATED at outcome_idx=4, NOT UNCONSTRAINED at index=5
        assert len(events) == 1
        assert events[0].outcome == RuleOutcome.VIOLATED, (
            "extract_events should look up rule_outcome at resolution_step - 1, "
            f"not resolution_step. Got outcome={events[0].outcome}"
        )

    def test_multi_walk_independent_failure_index(self):
        """Verify is_first_violation is tracked per-walk, not globally.

        Walk 0: violation at outcome_idx=4
        Walk 1: violation at outcome_idx=6
        Each should have is_first_violation=True for their own first violation."""
        jumper_map = {
            5: _make_jumper(5, r=3),
            7: _make_jumper(7, r=2),
        }

        generated = np.zeros((2, 12), dtype=np.int64)
        generated[0, 2] = 5  # Walk 0: encounter at t=2, resolution=5, outcome_idx=4
        generated[1, 5] = 7  # Walk 1: encounter at t=5, resolution=7, outcome_idx=6

        rule_outcome = np.full((2, 11), RuleOutcome.UNCONSTRAINED, dtype=np.int32)
        rule_outcome[0, 4] = RuleOutcome.VIOLATED
        rule_outcome[1, 6] = RuleOutcome.VIOLATED

        failure_index = np.array([4, 6], dtype=np.int32)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 2

        events_by_walk = {ev.walk_idx: ev for ev in events}
        assert events_by_walk[0].is_first_violation is True
        assert events_by_walk[1].is_first_violation is True
