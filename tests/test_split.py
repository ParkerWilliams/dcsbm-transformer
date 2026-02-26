"""Tests for held-out evaluation split assignment.

Verifies deterministic, stratified 50/50 split of evaluation walks
into exploratory and confirmatory sets.
"""

import numpy as np
import pytest

from src.evaluation.split import CONFIRMATORY, EXPLORATORY, SPLIT_SEED, assign_split


class TestAssignSplit:
    """Tests for assign_split function."""

    def test_split_deterministic(self):
        """Split assignment is deterministic across runs."""
        failure_index = np.array([5, -1, -1, 3, -1, -1, 2, -1])
        split_a = assign_split(failure_index, split_seed=2026)
        split_b = assign_split(failure_index, split_seed=2026)
        assert np.array_equal(split_a, split_b)

    def test_split_different_seeds(self):
        """Different seeds produce different assignments."""
        failure_index = np.array([5, -1, -1, 3, -1, -1, 2, -1, 7, -1] * 10)
        split_a = assign_split(failure_index, split_seed=2026)
        split_b = assign_split(failure_index, split_seed=9999)
        assert not np.array_equal(split_a, split_b)

    def test_split_all_walks_assigned(self):
        """Every walk gets either 'exploratory' or 'confirmatory'."""
        failure_index = np.array([5, -1, -1, 3, -1, -1, 2, -1])
        splits = assign_split(failure_index)
        for s in splits:
            assert s in (EXPLORATORY, CONFIRMATORY), f"Unexpected split value: {s}"

    def test_split_roughly_equal(self):
        """For 100 walks, each split gets between 45 and 55 walks."""
        failure_index = np.concatenate([
            np.full(20, 5),   # 20 violations
            np.full(80, -1),  # 80 non-violations
        ])
        splits = assign_split(failure_index)
        n_exploratory = (splits == EXPLORATORY).sum()
        n_confirmatory = (splits == CONFIRMATORY).sum()
        assert 45 <= n_exploratory <= 55, f"Exploratory count: {n_exploratory}"
        assert 45 <= n_confirmatory <= 55, f"Confirmatory count: {n_confirmatory}"
        assert n_exploratory + n_confirmatory == 100

    def test_split_stratified_violations(self):
        """Equal proportions of violations in each split (differ by at most 1)."""
        failure_index = np.concatenate([
            np.full(20, 5),   # 20 violations
            np.full(80, -1),  # 80 non-violations
        ])
        splits = assign_split(failure_index)
        violation_mask = failure_index >= 0
        viol_in_exp = (violation_mask & (splits == EXPLORATORY)).sum()
        viol_in_conf = (violation_mask & (splits == CONFIRMATORY)).sum()
        assert abs(viol_in_exp - viol_in_conf) <= 1, (
            f"Violation imbalance: exp={viol_in_exp}, conf={viol_in_conf}"
        )
        assert viol_in_exp + viol_in_conf == 20

    def test_split_stratified_nonviolations(self):
        """Equal proportions of non-violations in each split (differ by at most 1)."""
        failure_index = np.concatenate([
            np.full(20, 5),   # 20 violations
            np.full(80, -1),  # 80 non-violations
        ])
        splits = assign_split(failure_index)
        nonviol_mask = failure_index < 0
        nonviol_in_exp = (nonviol_mask & (splits == EXPLORATORY)).sum()
        nonviol_in_conf = (nonviol_mask & (splits == CONFIRMATORY)).sum()
        assert abs(nonviol_in_exp - nonviol_in_conf) <= 1, (
            f"Non-violation imbalance: exp={nonviol_in_exp}, conf={nonviol_in_conf}"
        )
        assert nonviol_in_exp + nonviol_in_conf == 80

    def test_split_all_violations(self):
        """Edge case: all walks have violations. Split should still produce 50/50."""
        failure_index = np.array([5, 3, 2, 7, 10, 1, 4, 8, 6, 9])
        splits = assign_split(failure_index)
        n_exp = (splits == EXPLORATORY).sum()
        n_conf = (splits == CONFIRMATORY).sum()
        assert n_exp + n_conf == 10
        assert abs(n_exp - n_conf) <= 1

    def test_split_no_violations(self):
        """Edge case: no walks have violations. Split should still produce 50/50."""
        failure_index = np.full(10, -1)
        splits = assign_split(failure_index)
        n_exp = (splits == EXPLORATORY).sum()
        n_conf = (splits == CONFIRMATORY).sum()
        assert n_exp + n_conf == 10
        assert abs(n_exp - n_conf) <= 1

    def test_split_single_walk(self):
        """Edge case: only 1 walk. Should assign to one split without error."""
        failure_index = np.array([-1])
        splits = assign_split(failure_index)
        assert len(splits) == 1
        assert splits[0] in (EXPLORATORY, CONFIRMATORY)

    def test_split_empty(self):
        """Edge case: empty failure_index array. Should return empty array."""
        failure_index = np.array([], dtype=np.int32)
        splits = assign_split(failure_index)
        assert len(splits) == 0

    def test_split_odd_count(self):
        """Odd number of violations: one split gets ceil, other gets floor."""
        failure_index = np.array([5, 3, 2, -1, -1, -1, -1])  # 3 violations, 4 non-viol
        splits = assign_split(failure_index)
        violation_mask = failure_index >= 0
        viol_in_exp = (violation_mask & (splits == EXPLORATORY)).sum()
        viol_in_conf = (violation_mask & (splits == CONFIRMATORY)).sum()
        assert viol_in_exp + viol_in_conf == 3
        assert abs(viol_in_exp - viol_in_conf) <= 1
        # Total should still be all assigned
        assert (splits == EXPLORATORY).sum() + (splits == CONFIRMATORY).sum() == 7
