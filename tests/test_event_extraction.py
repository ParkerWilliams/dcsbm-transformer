"""Tests for event extraction and contamination filtering.

Tests use synthetic data (no model needed). The indexing convention follows
behavioral.py: when a jumper vertex appears at generated[walk, t], the constraint
deadline is t + r. The rule_outcome is recorded at index deadline - 1 = t + r - 1.
We define resolution_step = encounter_step + r (the deadline), and cross-reference
rule_outcome at resolution_step - 1.
"""

import numpy as np
import pytest

from src.analysis.event_extraction import (
    AnalysisEvent,
    extract_events,
    filter_contaminated_events,
    stratify_by_r,
)
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


def _make_jumper_map(*jumpers: JumperInfo) -> dict[int, JumperInfo]:
    """Build a jumper_map dict from JumperInfo objects."""
    return {j.vertex_id: j for j in jumpers}


class TestExtractEvents:
    """Tests for extract_events function."""

    def test_extract_events_finds_jumper_encounters(self):
        """Build a synthetic generated array with known jumper vertices.
        Verify that extract_events returns AnalysisEvent records with
        correct walk_idx, encounter_step, resolution_step, r_value, and outcome.
        """
        # One sequence, length 20.
        # Jumper vertex 5 with r=4, source_block=0, target_block=1.
        # Place jumper at step 3 => resolution_step = 3 + 4 = 7
        # rule_outcome at index 6 (resolution_step - 1) = FOLLOWED
        generated = np.zeros((1, 20), dtype=np.int64)
        generated[0, 3] = 5  # jumper vertex at step 3

        rule_outcome = np.full((1, 19), RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
        rule_outcome[0, 6] = RuleOutcome.FOLLOWED  # step 6 = deadline-1 = 3+4-1

        failure_index = np.full(1, -1, dtype=np.int32)

        jumper = JumperInfo(vertex_id=5, source_block=0, target_block=1, r=4)
        jumper_map = _make_jumper_map(jumper)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)

        assert len(events) == 1
        ev = events[0]
        assert ev.walk_idx == 0
        assert ev.encounter_step == 3
        assert ev.resolution_step == 7  # encounter_step + r
        assert ev.r_value == 4
        assert ev.outcome == RuleOutcome.FOLLOWED
        assert ev.is_first_violation is False

    def test_extract_events_first_violation_only(self):
        """Walk with two violations: verify is_first_violation=True only for the first."""
        # Two jumpers: vertex 5 (r=3) at step 2, vertex 7 (r=3) at step 10
        generated = np.zeros((1, 30), dtype=np.int64)
        generated[0, 2] = 5   # encounter at step 2, resolution = 5
        generated[0, 10] = 7  # encounter at step 10, resolution = 13

        rule_outcome = np.full((1, 29), RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
        rule_outcome[0, 4] = RuleOutcome.VIOLATED   # step 4 = 2+3-1
        rule_outcome[0, 12] = RuleOutcome.VIOLATED  # step 12 = 10+3-1

        # failure_index = first violation step = 4
        failure_index = np.array([4], dtype=np.int32)

        j5 = JumperInfo(vertex_id=5, source_block=0, target_block=1, r=3)
        j7 = JumperInfo(vertex_id=7, source_block=0, target_block=1, r=3)
        jumper_map = _make_jumper_map(j5, j7)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)

        violations = [e for e in events if e.outcome == RuleOutcome.VIOLATED]
        assert len(violations) == 2

        first_viol = [e for e in violations if e.encounter_step == 2][0]
        second_viol = [e for e in violations if e.encounter_step == 10][0]

        assert first_viol.is_first_violation is True
        assert second_viol.is_first_violation is False

    def test_no_events_returns_empty(self):
        """No jumper vertices in generated sequences. Returns empty list."""
        generated = np.zeros((2, 10), dtype=np.int64)
        rule_outcome = np.full((2, 9), RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
        failure_index = np.full(2, -1, dtype=np.int32)

        # Jumper vertex 99 is never in the generated array
        jumper = JumperInfo(vertex_id=99, source_block=0, target_block=1, r=3)
        jumper_map = _make_jumper_map(jumper)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert events == []

    def test_resolution_step_alignment(self):
        """Verify resolution_step = encounter_step + r exactly (off-by-one guard)."""
        # Jumper vertex 10 with r=5 at step 8 => resolution = 13
        generated = np.zeros((1, 25), dtype=np.int64)
        generated[0, 8] = 10

        rule_outcome = np.full((1, 24), RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
        rule_outcome[0, 12] = RuleOutcome.FOLLOWED  # 8+5-1 = 12

        failure_index = np.full(1, -1, dtype=np.int32)

        jumper = JumperInfo(vertex_id=10, source_block=0, target_block=1, r=5)
        jumper_map = _make_jumper_map(jumper)

        events = extract_events(generated, rule_outcome, failure_index, jumper_map)
        assert len(events) == 1
        ev = events[0]
        assert ev.resolution_step == ev.encounter_step + ev.r_value
        assert ev.resolution_step == 13


class TestContaminationFilter:
    """Tests for filter_contaminated_events function."""

    def test_contamination_filter_excludes_overlapping(self):
        """Two encounters in the same walk where encounter B's countdown window
        overlaps violation A's window (B.encounter_step < A.resolution_step).
        Verify B is excluded and exclusion_count=1.
        """
        # Event A: violation at encounter_step=2, r=5, resolution=7
        # Event B: encounter at encounter_step=5, r=3, resolution=8
        # B.encounter_step (5) < A.resolution_step (7) => B excluded
        event_a = AnalysisEvent(
            walk_idx=0, encounter_step=2, resolution_step=7,
            r_value=5, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
        )
        event_b = AnalysisEvent(
            walk_idx=0, encounter_step=5, resolution_step=8,
            r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
        )

        filtered, audit = filter_contaminated_events([event_a, event_b])

        assert len(filtered) == 1
        assert filtered[0] == event_a
        assert audit["excluded_encounters"] == 1
        assert audit["exclusion_rate"] == pytest.approx(0.5)

    def test_contamination_filter_keeps_non_overlapping(self):
        """Two encounters where B.encounter_step >= A.resolution_step.
        Verify both kept.
        """
        # Event A: violation at encounter_step=2, r=3, resolution=5
        # Event B: encounter at encounter_step=5, r=3, resolution=8
        # B.encounter_step (5) >= A.resolution_step (5) => both kept
        event_a = AnalysisEvent(
            walk_idx=0, encounter_step=2, resolution_step=5,
            r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
        )
        event_b = AnalysisEvent(
            walk_idx=0, encounter_step=5, resolution_step=8,
            r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
        )

        filtered, audit = filter_contaminated_events([event_a, event_b])

        assert len(filtered) == 2
        assert audit["excluded_encounters"] == 0

    def test_contamination_filter_successful_prior_does_not_contaminate(self):
        """Encounter A is FOLLOWED (success), encounter B starts within A's window.
        Verify B is NOT excluded (only violations contaminate per CONTEXT.md).
        """
        # Event A: FOLLOWED at encounter_step=2, r=5, resolution=7
        # Event B: encounter at encounter_step=5, r=3, resolution=8
        # A is FOLLOWED => does NOT contaminate => B kept
        event_a = AnalysisEvent(
            walk_idx=0, encounter_step=2, resolution_step=7,
            r_value=5, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
        )
        event_b = AnalysisEvent(
            walk_idx=0, encounter_step=5, resolution_step=8,
            r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
        )

        filtered, audit = filter_contaminated_events([event_a, event_b])

        assert len(filtered) == 2
        assert audit["excluded_encounters"] == 0

    def test_contamination_audit_threshold(self):
        """>30% exclusion rate produces flagged=True."""
        # Create 3 events in same walk: first is violation, next 2 are within window
        events = [
            AnalysisEvent(
                walk_idx=0, encounter_step=0, resolution_step=100,
                r_value=100, outcome=RuleOutcome.VIOLATED, is_first_violation=True,
            ),
            AnalysisEvent(
                walk_idx=0, encounter_step=10, resolution_step=40,
                r_value=30, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
            AnalysisEvent(
                walk_idx=0, encounter_step=20, resolution_step=50,
                r_value=30, outcome=RuleOutcome.FOLLOWED, is_first_violation=False,
            ),
        ]

        filtered, audit = filter_contaminated_events(events)

        # 2 out of 3 excluded => 66.7%
        assert audit["exclusion_rate"] > 0.3
        assert audit["flagged"] is True


class TestStratifyByR:
    """Tests for stratify_by_r function."""

    def test_stratify_by_r(self):
        """Events with r=32 and r=45. Verify correct grouping."""
        events = [
            AnalysisEvent(walk_idx=0, encounter_step=5, resolution_step=37,
                          r_value=32, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
            AnalysisEvent(walk_idx=1, encounter_step=3, resolution_step=48,
                          r_value=45, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
            AnalysisEvent(walk_idx=2, encounter_step=10, resolution_step=42,
                          r_value=32, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
        ]

        stratified = stratify_by_r(events)

        assert set(stratified.keys()) == {32, 45}
        assert len(stratified[32]) == 2
        assert len(stratified[45]) == 1
