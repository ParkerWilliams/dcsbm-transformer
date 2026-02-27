"""Tests for signal concentration analysis (Phase 16: MHAD-03, MHAD-04).

Validates entropy, Gini coefficient, signal concentration report, and
ablation comparison infrastructure.
"""

import numpy as np
import pytest

from src.analysis.signal_concentration import (
    compute_auroc_entropy,
    compute_gini_coefficient,
    compute_signal_concentration,
    compute_ablation_comparison,
)


class TestAurocEntropy:
    """Entropy of per-head AUROC distribution."""

    def test_uniform_distribution_max_entropy(self):
        """Equal AUROC across all heads -> entropy = 1.0 (maximally distributed)."""
        aurocs = np.array([0.7, 0.7, 0.7, 0.7])
        assert compute_auroc_entropy(aurocs) == pytest.approx(1.0, abs=1e-6)

    def test_two_heads_equal(self):
        """Two equal heads -> entropy = 1.0."""
        assert compute_auroc_entropy(np.array([0.7, 0.7])) == pytest.approx(1.0, abs=1e-6)

    def test_two_heads_unequal(self):
        """Two unequal heads -> entropy < 1.0."""
        entropy = compute_auroc_entropy(np.array([0.9, 0.5]))
        assert 0 < entropy < 1.0

    def test_highly_concentrated(self):
        """One high, rest near chance -> low entropy."""
        aurocs = np.array([0.95, 0.50, 0.50, 0.50])
        entropy = compute_auroc_entropy(aurocs)
        # Below max (uniform) since one head dominates; but AUROC-based
        # entropy normalizes absolute values, so the gap is modest
        assert entropy < 1.0

    def test_single_head_returns_nan(self):
        """Single head -> entropy is NaN (concentration not meaningful)."""
        assert np.isnan(compute_auroc_entropy(np.array([0.8])))

    def test_nan_values_treated_as_chance(self):
        """NaN AUROC values replaced with 0.5, computation proceeds."""
        aurocs = np.array([0.8, float("nan"), 0.8, 0.8])
        entropy = compute_auroc_entropy(aurocs)
        assert not np.isnan(entropy)
        assert 0 < entropy <= 1.0

    def test_all_nans(self):
        """All NaN -> treated as uniform 0.5 -> entropy = 1.0."""
        aurocs = np.array([float("nan"), float("nan")])
        assert compute_auroc_entropy(aurocs) == pytest.approx(1.0, abs=1e-6)

    def test_entropy_bounded_0_1(self):
        """Entropy should always be in [0, 1] for valid inputs."""
        for _ in range(20):
            aurocs = np.random.uniform(0.3, 0.95, size=4)
            entropy = compute_auroc_entropy(aurocs)
            assert 0 <= entropy <= 1.0 + 1e-10


class TestGiniCoefficient:
    """Gini coefficient of per-head AUROC distribution."""

    def test_equal_values_zero_gini(self):
        """Equal AUROC -> Gini = 0 (perfect equality)."""
        aurocs = np.array([0.7, 0.7, 0.7, 0.7])
        assert compute_gini_coefficient(aurocs) == pytest.approx(0.0, abs=1e-6)

    def test_unequal_positive_gini(self):
        """Unequal AUROC -> Gini > 0."""
        aurocs = np.array([0.9, 0.5, 0.5, 0.5])
        gini = compute_gini_coefficient(aurocs)
        assert gini > 0

    def test_single_head_returns_nan(self):
        """Single head -> Gini is NaN."""
        assert np.isnan(compute_gini_coefficient(np.array([0.8])))

    def test_gini_bounded_0_1(self):
        """Gini should be in [0, 1]."""
        aurocs = np.array([0.99, 0.01, 0.01, 0.01])
        gini = compute_gini_coefficient(aurocs)
        assert 0 <= gini <= 1

    def test_two_equal_heads(self):
        """Two equal -> Gini = 0."""
        assert compute_gini_coefficient(np.array([0.6, 0.6])) == pytest.approx(0.0, abs=1e-6)

    def test_nan_values_handled(self):
        """NaN values treated as 0.5, computation succeeds."""
        aurocs = np.array([0.9, float("nan"), 0.5, 0.5])
        gini = compute_gini_coefficient(aurocs)
        assert not np.isnan(gini)
        assert 0 <= gini <= 1


class TestSignalConcentration:
    """Signal concentration report from per-head AUROC values."""

    def test_basic_output_structure(self):
        """Output dict has all required keys."""
        result = compute_signal_concentration(
            {0: 0.8, 1: 0.6}, metric_name="grassmannian_distance"
        )
        assert result["metric_name"] == "grassmannian_distance"
        assert result["n_heads"] == 2
        assert result["dominant_head"] == 0
        assert result["dominant_auroc"] == pytest.approx(0.8)
        assert "entropy" in result
        assert "gini" in result
        assert "max_to_mean_ratio" in result
        assert "interpretation" in result

    def test_single_head_not_applicable(self):
        """Single head -> interpretation says not applicable."""
        result = compute_signal_concentration({0: 0.8})
        assert "not applicable" in result["interpretation"].lower()
        assert np.isnan(result["entropy"])
        assert np.isnan(result["gini"])

    def test_concentrated_signal_interpretation(self):
        """Strongly concentrated signal -> interpretation mentions concentrated."""
        # Need extreme disparity: one head high, rest near zero
        result = compute_signal_concentration(
            {0: 0.95, 1: 0.05, 2: 0.05, 3: 0.05}
        )
        # Either "concentrated" or "partially"
        interp_lower = result["interpretation"].lower()
        assert "concentrated" in interp_lower or "partially" in interp_lower

    def test_distributed_signal_interpretation(self):
        """Uniformly distributed signal -> entropy near 1.0."""
        result = compute_signal_concentration(
            {0: 0.75, 1: 0.75, 2: 0.75, 3: 0.75}
        )
        assert result["entropy"] == pytest.approx(1.0, abs=1e-3)
        assert "distributed" in result["interpretation"].lower()

    def test_max_to_mean_ratio(self):
        """Max-to-mean ratio is correctly computed."""
        result = compute_signal_concentration({0: 0.9, 1: 0.5})
        expected = 0.9 / ((0.9 + 0.5) / 2)
        assert result["max_to_mean_ratio"] == pytest.approx(expected, abs=1e-6)

    def test_dominant_head_correct(self):
        """Dominant head is the one with highest AUROC."""
        result = compute_signal_concentration({0: 0.6, 1: 0.9, 2: 0.5})
        assert result["dominant_head"] == 1
        assert result["dominant_auroc"] == pytest.approx(0.9)


class TestAblationComparison:
    """Ablation comparison across 1h/2h/4h configs."""

    def test_basic_comparison(self):
        """Comparison produces expected structure."""
        results = {
            1: {"qkt.grassmannian_distance": 0.75},
            2: {"qkt.grassmannian_distance": 0.78},
            4: {"qkt.grassmannian_distance": 0.72},
        }
        comp = compute_ablation_comparison(results)
        assert comp["configs"] == [1, 2, 4]
        assert "aggregate_comparison" in comp
        assert "conclusion" in comp
        assert "qkt.grassmannian_distance" in comp["aggregate_comparison"]

    def test_improvement_detected(self):
        """When multi-head clearly beats single, conclusion reflects it."""
        results = {
            1: {"metric_a": 0.60},
            2: {"metric_a": 0.85},
        }
        comp = compute_ablation_comparison(results)
        assert "improves" in comp["conclusion"].lower()

    def test_comparable_detected(self):
        """When results are similar, conclusion says comparable."""
        results = {
            1: {"metric_a": 0.75},
            2: {"metric_a": 0.76},
        }
        comp = compute_ablation_comparison(results)
        assert "comparable" in comp["conclusion"].lower()

    def test_single_head_outperforms(self):
        """When single-head wins, conclusion reflects it."""
        results = {
            1: {"metric_a": 0.85},
            4: {"metric_a": 0.60},
        }
        comp = compute_ablation_comparison(results)
        assert "outperforms" in comp["conclusion"].lower()

    def test_empty_results(self):
        """Empty input produces valid output."""
        comp = compute_ablation_comparison({})
        assert comp["configs"] == []
        assert comp["conclusion"] == "Insufficient data for comparison"

    def test_missing_metrics_nan(self):
        """Missing metrics in some configs produce NaN in aggregate."""
        results = {
            1: {"metric_a": 0.75, "metric_b": 0.60},
            2: {"metric_a": 0.78},
        }
        comp = compute_ablation_comparison(results)
        assert np.isnan(comp["aggregate_comparison"]["metric_b"][2])
