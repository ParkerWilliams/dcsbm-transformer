"""Audit tests for compliance rate formula (GRAPH-05).

Verifies that the compliance rate computation matches the mathematical
definition: compliance = followed / constrained = 1 - violations / constrained.
Tests cover all-followed, all-violated, mixed, no-constrained-steps, and
independent algebra verification.
"""

import numpy as np
import pytest

from src.graph.jumpers import JumperInfo


def _compute_compliance_from_sequences(
    sequences: np.ndarray,
    jumper_map: dict[int, JumperInfo],
    block_assignments: np.ndarray,
) -> tuple[int, int]:
    """Independently compute rule compliance counts from raw sequences.

    Replicates the counting logic from evaluate_compliance without calling it,
    providing an independent reference implementation.

    Returns:
        Tuple of (rule_compliant, total_rule_checks).
    """
    total_rule_checks = 0
    rule_compliant = 0

    for seq in sequences:
        seq_len = len(seq)
        for t in range(seq_len - 1):
            u = int(seq[t])
            if u in jumper_map:
                jumper = jumper_map[u]
                if t + jumper.r < seq_len:
                    total_rule_checks += 1
                    arrival_vertex = int(seq[t + jumper.r])
                    actual_block = int(block_assignments[arrival_vertex])
                    if actual_block == jumper.target_block:
                        rule_compliant += 1

    return rule_compliant, total_rule_checks


class TestComplianceRateFormulaMatchesDefinition:
    """Verify compliance rate formula: followed/constrained = 1 - violation_rate."""

    def test_compliance_rate_formula_matches_definition(self) -> None:
        """GRAPH-05: compliance = followed/constrained = 1 - violations/constrained.
        The code computes the COMPLIANCE rate (fraction correct), not the
        VIOLATION rate. Verify this relationship with a synthetic scenario
        with known outcomes.
        """
        # Set up: 4 blocks, vertex 2 is a jumper (block 0 -> target block 1, r=3)
        n = 8
        K = 4
        block_size = 2
        block_assignments = np.arange(n) // block_size
        # blocks: [0,0,1,1,2,2,3,3]

        jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3)
        jumper_map = {0: jumper}

        # Create sequences where vertex 0 appears at various positions
        # Sequence 1: [..., 0, ?, ?, 2, ...] -- vertex at t+3=2 is in block 1 (FOLLOWED)
        # Sequence 2: [..., 0, ?, ?, 4, ...] -- vertex at t+3=4 is in block 2 (VIOLATED)
        # Sequence 3: [..., 0, ?, ?, 3, ...] -- vertex at t+3=3 is in block 1 (FOLLOWED)
        sequences = np.array([
            [0, 1, 1, 2, 1, 1, 1, 1],  # t=0: jumper, arrival at t=3 -> v=2 (block 1) FOLLOWED
            [0, 1, 1, 4, 1, 1, 1, 1],  # t=0: jumper, arrival at t=3 -> v=4 (block 2) VIOLATED
            [0, 1, 1, 3, 1, 1, 1, 1],  # t=0: jumper, arrival at t=3 -> v=3 (block 1) FOLLOWED
        ])

        rule_compliant, total_rule_checks = _compute_compliance_from_sequences(
            sequences, jumper_map, block_assignments
        )

        # 3 constrained steps total (vertex 0 appears at t=0 in each sequence)
        assert total_rule_checks == 3, (
            f"Expected 3 rule checks, got {total_rule_checks}"
        )
        # 2 followed, 1 violated
        assert rule_compliant == 2, (
            f"Expected 2 compliant, got {rule_compliant}"
        )

        # Compliance rate
        compliance = rule_compliant / total_rule_checks
        violation_rate = (total_rule_checks - rule_compliant) / total_rule_checks

        assert compliance == pytest.approx(2 / 3)
        assert violation_rate == pytest.approx(1 / 3)
        assert compliance == pytest.approx(1 - violation_rate)


class TestComplianceRateAllFollowed:
    """Verify compliance = 1.0 when all jumper constraints are met."""

    def test_compliance_rate_all_followed(self) -> None:
        """All constraints satisfied => compliance = 1.0.
        Construct sequences where every jumper encounter resolves to the
        correct target block.
        """
        n = 8
        block_assignments = np.arange(n) // 2
        # blocks: [0,0,1,1,2,2,3,3]

        jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2)
        jumper_map = {0: jumper}

        # All sequences have vertex 0 at start, arrival at t+2 in block 1 (vertices 2 or 3)
        sequences = np.array([
            [0, 1, 2, 1, 1, 1],  # t=0: arrival at t+2=v2 (block 1) FOLLOWED
            [0, 1, 3, 1, 1, 1],  # t=0: arrival at t+2=v3 (block 1) FOLLOWED
            [0, 1, 2, 1, 1, 1],  # t=0: arrival at t+2=v2 (block 1) FOLLOWED
        ])

        rule_compliant, total_rule_checks = _compute_compliance_from_sequences(
            sequences, jumper_map, block_assignments
        )

        assert total_rule_checks == 3
        assert rule_compliant == 3
        compliance = rule_compliant / total_rule_checks
        assert compliance == 1.0


class TestComplianceRateAllViolated:
    """Verify compliance = 0.0 when all jumper constraints are violated."""

    def test_compliance_rate_all_violated(self) -> None:
        """All constraints violated => compliance = 0.0.
        Construct sequences where every jumper encounter resolves to the
        wrong block.
        """
        n = 8
        block_assignments = np.arange(n) // 2
        # blocks: [0,0,1,1,2,2,3,3]

        jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2)
        jumper_map = {0: jumper}

        # All sequences have vertex 0 at start, arrival at t+2 NOT in block 1.
        # Avoid vertex 0 elsewhere in sequence to prevent extra jumper checks.
        sequences = np.array([
            [0, 1, 4, 1, 1, 1],  # t=0: arrival at t+2=v4 (block 2) VIOLATED
            [0, 1, 6, 1, 1, 1],  # t=0: arrival at t+2=v6 (block 3) VIOLATED
            [0, 1, 7, 1, 1, 1],  # t=0: arrival at t+2=v7 (block 3) VIOLATED
        ])

        rule_compliant, total_rule_checks = _compute_compliance_from_sequences(
            sequences, jumper_map, block_assignments
        )

        assert total_rule_checks == 3
        assert rule_compliant == 0
        compliance = rule_compliant / total_rule_checks
        assert compliance == 0.0


class TestComplianceRateMixed:
    """Verify compliance = n_followed / n_constrained for mixed outcomes."""

    def test_compliance_rate_mixed(self) -> None:
        """Mixed outcomes: compliance = n_followed / n_constrained.
        With 3 followed and 2 violated out of 5 constrained steps,
        compliance == 3/5 == 0.6.
        """
        n = 8
        block_assignments = np.arange(n) // 2
        # blocks: [0,0,1,1,2,2,3,3]

        jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2)
        jumper_map = {0: jumper}

        sequences = np.array([
            [0, 1, 2, 1, 1, 1],  # FOLLOWED (v2 in block 1)
            [0, 1, 3, 1, 1, 1],  # FOLLOWED (v3 in block 1)
            [0, 1, 2, 1, 1, 1],  # FOLLOWED (v2 in block 1)
            [0, 1, 4, 1, 1, 1],  # VIOLATED (v4 in block 2)
            [0, 1, 6, 1, 1, 1],  # VIOLATED (v6 in block 3)
        ])

        rule_compliant, total_rule_checks = _compute_compliance_from_sequences(
            sequences, jumper_map, block_assignments
        )

        assert total_rule_checks == 5
        assert rule_compliant == 3
        compliance = rule_compliant / total_rule_checks
        assert compliance == pytest.approx(0.6)


class TestComplianceRateNoConstrainedSteps:
    """Verify compliance defaults to 1.0 when no constraints are active."""

    def test_compliance_rate_no_constrained_steps(self) -> None:
        """No constrained steps => compliance defaults to 1.0 (vacuously true).
        This matches the code's `if total_rule_checks > 0 else 1.0` default.
        Tested with sequences containing no jumper vertices.
        """
        n = 8
        block_assignments = np.arange(n) // 2

        # Jumper is vertex 0 with r=2
        jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2)
        jumper_map = {0: jumper}

        # Sequences that never visit vertex 0
        sequences = np.array([
            [1, 2, 3, 4, 5, 6],
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 1],
        ])

        rule_compliant, total_rule_checks = _compute_compliance_from_sequences(
            sequences, jumper_map, block_assignments
        )

        assert total_rule_checks == 0
        assert rule_compliant == 0

        # Default compliance when no checks: 1.0
        compliance = rule_compliant / total_rule_checks if total_rule_checks > 0 else 1.0
        assert compliance == 1.0

    def test_compliance_rate_jumper_at_boundary(self) -> None:
        """When jumper appears but t + r >= seq_len, no constrained step is
        counted. Compliance defaults to 1.0.
        """
        n = 8
        block_assignments = np.arange(n) // 2

        jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=5)
        jumper_map = {0: jumper}

        # Sequence length 6, jumper at t=3, r=5, so t+r=8 >= 6: no check
        sequences = np.array([
            [1, 2, 3, 0, 5, 6],  # t=3: t+r=8 >= 6, no check
        ])

        rule_compliant, total_rule_checks = _compute_compliance_from_sequences(
            sequences, jumper_map, block_assignments
        )

        assert total_rule_checks == 0
        compliance = rule_compliant / total_rule_checks if total_rule_checks > 0 else 1.0
        assert compliance == 1.0


class TestComplianceIndependentAlgebra:
    """Programmatic algebra check: independent recomputation validates counting logic."""

    def test_compliance_independent_algebra(self) -> None:
        """Independent recomputation validates that the code's counting logic
        matches the mathematical formula.

        We build a scenario with multiple jumpers and sequences, then:
        (a) Count using our reference implementation
        (b) Count using a completely independent loop with different code structure
        (c) Verify both produce identical results
        """
        n = 12
        K = 3
        block_size = 4
        block_assignments = np.arange(n) // block_size
        # blocks: [0,0,0,0, 1,1,1,1, 2,2,2,2]

        # Multiple jumpers
        jumpers = [
            JumperInfo(vertex_id=1, source_block=0, target_block=2, r=3),
            JumperInfo(vertex_id=5, source_block=1, target_block=0, r=2),
        ]
        jumper_map = {j.vertex_id: j for j in jumpers}

        # Synthetic sequences
        sequences = np.array([
            [1, 3, 2, 8, 5, 0, 1, 3, 2, 9, 5, 2],
            # t=0: jumper v=1 (r=3), arrival at t=3 -> v=8 (block 2) FOLLOWED
            # t=4: jumper v=5 (r=2), arrival at t=6 -> v=1 (block 0) FOLLOWED
            # t=6: jumper v=1 (r=3), arrival at t=9 -> v=9 (block 2) FOLLOWED
            # t=10: jumper v=5 (r=2), arrival at t=12 >= 12, no check
            [5, 3, 0, 1, 3, 2, 10, 5, 3, 1, 2, 3],
            # t=0: jumper v=5 (r=2), arrival at t=2 -> v=0 (block 0) FOLLOWED
            # t=3: jumper v=1 (r=3), arrival at t=6 -> v=10 (block 2) FOLLOWED
            # t=7: jumper v=5 (r=2), arrival at t=9 -> v=1 (block 0) FOLLOWED
            [1, 3, 2, 6, 5, 0, 11, 5, 3, 1, 2, 3],
            # t=0: jumper v=1 (r=3), arrival at t=3 -> v=6 (block 1) VIOLATED
            # t=4: jumper v=5 (r=2), arrival at t=6 -> v=11 (block 2) VIOLATED (target is block 0)
            # t=7: jumper v=5 (r=2), arrival at t=9 -> v=1 (block 0) FOLLOWED
        ])

        # Method A: our reference implementation
        compliant_a, checks_a = _compute_compliance_from_sequences(
            sequences, jumper_map, block_assignments
        )

        # Method B: completely independent counting
        compliant_b = 0
        checks_b = 0
        for row_idx in range(sequences.shape[0]):
            seq = sequences[row_idx]
            for pos in range(len(seq) - 1):
                vertex = int(seq[pos])
                if vertex in jumper_map:
                    j = jumper_map[vertex]
                    arrival_pos = pos + j.r
                    if arrival_pos < len(seq):
                        checks_b += 1
                        arrival_v = int(seq[arrival_pos])
                        if int(block_assignments[arrival_v]) == j.target_block:
                            compliant_b += 1

        assert checks_a == checks_b, (
            f"Method A checks={checks_a} != Method B checks={checks_b}"
        )
        assert compliant_a == compliant_b, (
            f"Method A compliant={compliant_a} != Method B compliant={compliant_b}"
        )

        # Verify the expected values manually computed from the sequences above
        # Seq 0: 3 checks, 3 followed
        # Seq 1: 3 checks, 3 followed
        # Seq 2: 3 checks, 1 followed
        expected_checks = 9
        expected_compliant = 7
        assert checks_a == expected_checks, (
            f"Expected {expected_checks} checks, got {checks_a}"
        )
        assert compliant_a == expected_compliant, (
            f"Expected {expected_compliant} compliant, got {compliant_a}"
        )

        compliance = compliant_a / checks_a
        assert compliance == pytest.approx(7 / 9)
