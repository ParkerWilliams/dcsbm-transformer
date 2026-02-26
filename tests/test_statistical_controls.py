"""Tests for statistical controls: Holm-Bonferroni, BCa bootstrap, Cohen's d,
correlation/redundancy analysis, metric ranking, and headline comparison.

All tests use synthetic data. No model or real evaluation data needed.
"""

import json

import numpy as np
import pytest

from src.analysis.auroc_horizon import PRIMARY_METRICS, auroc_from_groups
from src.analysis.event_extraction import AnalysisEvent
from src.analysis.statistical_controls import (
    apply_statistical_controls,
    auroc_with_bootstrap_ci,
    cohens_d,
    compute_cohens_d_by_lookback,
    compute_correlation_matrix,
    compute_headline_comparison,
    compute_metric_ranking,
    holm_bonferroni,
)
from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(walk_indices, resolution_steps, r_value, outcome):
    """Helper to create AnalysisEvent lists."""
    return [
        AnalysisEvent(
            walk_idx=w,
            encounter_step=rs - r_value,
            resolution_step=rs,
            r_value=r_value,
            outcome=outcome,
            is_first_violation=(outcome == RuleOutcome.VIOLATED),
        )
        for w, rs in zip(walk_indices, resolution_steps)
    ]


# ---------------------------------------------------------------------------
# Holm-Bonferroni tests
# ---------------------------------------------------------------------------

class TestHolmBonferroni:
    """Tests for the Holm-Bonferroni step-down correction."""

    def test_holm_bonferroni_basic(self):
        """p_values=[0.01, 0.04, 0.03, 0.005, 0.8], alpha=0.05.

        Sorted ascending: [0.005, 0.01, 0.03, 0.04, 0.8]
        Indices in original: [3, 0, 2, 1, 4]
        Multiplied by [5, 4, 3, 2, 1]:
          0.005*5=0.025, 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08, 0.8*1=0.8
        Enforce monotonicity:
          0.025, 0.04, 0.09, 0.09, 0.8
        Map back to original order:
          [0.04, 0.09, 0.09, 0.025, 0.8]
        Rejections at alpha=0.05:
          [True, False, False, True, False]
        """
        p_values = np.array([0.01, 0.04, 0.03, 0.005, 0.8])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)

        # Verify adjusted values
        assert adjusted[0] == pytest.approx(0.04)   # p=0.01 -> rank 2 -> 0.01*4=0.04
        assert adjusted[1] == pytest.approx(0.09)   # p=0.04 -> rank 4 -> max(0.04*2, 0.09)=0.09
        assert adjusted[2] == pytest.approx(0.09)   # p=0.03 -> rank 3 -> 0.03*3=0.09
        assert adjusted[3] == pytest.approx(0.025)  # p=0.005 -> rank 1 -> 0.005*5=0.025
        assert adjusted[4] == pytest.approx(0.8)    # p=0.8 -> rank 5 -> 0.8*1=0.8

        # Verify reject flags
        assert reject[0] is True   # 0.04 <= 0.05
        assert reject[1] is False  # 0.09 > 0.05
        assert reject[2] is False  # 0.09 > 0.05
        assert reject[3] is True   # 0.025 <= 0.05
        assert reject[4] is False  # 0.8 > 0.05

    def test_holm_bonferroni_all_significant(self):
        """All p-values very small. All should be rejected."""
        p_values = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)
        assert all(reject), f"Expected all rejected, got {reject}"
        assert all(a <= 0.05 for a in adjusted)

    def test_holm_bonferroni_none_significant(self):
        """All p-values > 0.5. None should be rejected."""
        p_values = np.array([0.6, 0.7, 0.8, 0.9, 0.95])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)
        assert not any(reject), f"Expected none rejected, got {reject}"

    def test_holm_bonferroni_five_primary_metrics(self):
        """Exactly 5 p-values (the pre-registered primary metrics).

        Verify the correction factor is at most 5 (not 21).
        The smallest p-value is multiplied by 5, not 21.
        """
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        adjusted, reject = holm_bonferroni(p_values, alpha=0.05)
        # Smallest p-value (0.01) multiplied by 5 (not 21)
        assert adjusted[0] == pytest.approx(0.05)  # 0.01 * 5 = 0.05
        # Second smallest (0.02) multiplied by 4
        assert adjusted[1] <= 0.08 + 1e-10  # 0.02 * 4 = 0.08, monotonicity


# ---------------------------------------------------------------------------
# BCa Bootstrap CI tests
# ---------------------------------------------------------------------------

class TestBootstrapAurocCi:
    """Tests for BCa bootstrap confidence intervals on AUROC."""

    def test_bootstrap_auroc_ci_separable(self):
        """Perfect separation (AUROC=1.0). CI should be near [0.95, 1.0]."""
        violations = np.array([10.0, 11.0, 12.0, 13.0, 14.0,
                               15.0, 16.0, 17.0, 18.0, 19.0])
        controls = np.array([0.0, 1.0, 2.0, 3.0, 4.0,
                             5.0, 6.0, 7.0, 8.0, 9.0])

        point, ci_low, ci_high = auroc_with_bootstrap_ci(
            violations, controls, n_resamples=1000, rng=42
        )

        assert point == 1.0
        assert ci_low >= 0.9, f"CI low {ci_low} should be >= 0.9"
        assert ci_high <= 1.001, f"CI high {ci_high} should be <= 1.0"

    def test_bootstrap_auroc_ci_no_signal(self):
        """No separation (AUROC ~0.5). CI should contain 0.5."""
        rng = np.random.default_rng(42)
        violations = rng.standard_normal(30)
        controls = rng.standard_normal(30)

        point, ci_low, ci_high = auroc_with_bootstrap_ci(
            violations, controls, n_resamples=1000, rng=42
        )

        assert ci_low < 0.5 < ci_high, (
            f"CI [{ci_low}, {ci_high}] should contain 0.5"
        )

    def test_bootstrap_auroc_ci_deterministic(self):
        """Same inputs, same rng seed. Results must be identical."""
        rng_data = np.random.default_rng(99)
        violations = rng_data.standard_normal(20)
        controls = rng_data.standard_normal(20)

        result1 = auroc_with_bootstrap_ci(violations, controls, n_resamples=500, rng=42)
        result2 = auroc_with_bootstrap_ci(violations, controls, n_resamples=500, rng=42)

        assert result1[0] == result2[0]
        assert result1[1] == pytest.approx(result2[1])
        assert result1[2] == pytest.approx(result2[2])

    def test_bootstrap_auroc_ci_fallback_to_percentile(self):
        """Very small samples (n=3 per group) where BCa may fail.

        Verify CI is returned (not NaN) via percentile fallback.
        """
        violations = np.array([10.0, 11.0, 12.0])
        controls = np.array([1.0, 2.0, 3.0])

        point, ci_low, ci_high = auroc_with_bootstrap_ci(
            violations, controls, n_resamples=500, rng=42
        )

        assert np.isfinite(point)
        assert np.isfinite(ci_low), f"CI low should be finite, got {ci_low}"
        assert np.isfinite(ci_high), f"CI high should be finite, got {ci_high}"
        assert ci_low <= point <= ci_high


# ---------------------------------------------------------------------------
# Cohen's d tests
# ---------------------------------------------------------------------------

class TestCohensD:
    """Tests for Cohen's d effect size computation."""

    def test_cohens_d_known_value(self):
        """group1=[5,6,7,8], group2=[1,2,3,4]. Compute expected d by hand.

        mean1=6.5, mean2=2.5, diff=4.0
        var1 = var([5,6,7,8], ddof=1) = 5/3
        var2 = var([1,2,3,4], ddof=1) = 5/3
        pooled_std = sqrt(((3*5/3) + (3*5/3)) / 6) = sqrt(10/6) = sqrt(5/3)
        d = 4.0 / sqrt(5/3) = 4.0 * sqrt(3/5) = 4.0 * 0.7746 = 3.098
        """
        group1 = np.array([5.0, 6.0, 7.0, 8.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0])
        d = cohens_d(group1, group2)
        expected = 4.0 / np.sqrt(5 / 3)
        assert d == pytest.approx(expected, rel=1e-6)

    def test_cohens_d_identical_groups(self):
        """Same values. d should be 0.0 or NaN (zero pooled_std)."""
        group1 = np.array([5.0, 5.0, 5.0, 5.0])
        group2 = np.array([5.0, 5.0, 5.0, 5.0])
        d = cohens_d(group1, group2)
        # pooled_std is 0, so should return NaN
        assert np.isnan(d) or d == 0.0

    def test_cohens_d_insufficient_samples(self):
        """Less than 2 in a group. Returns NaN."""
        group1 = np.array([5.0])
        group2 = np.array([1.0, 2.0, 3.0])
        assert np.isnan(cohens_d(group1, group2))

        group1 = np.array([5.0, 6.0])
        group2 = np.array([])
        assert np.isnan(cohens_d(group1, group2))

    def test_cohens_d_by_lookback(self):
        """Compute d at each lookback j using violation and control metric values.

        Verify shape matches r.
        """
        r_value = 5
        n_walks = 20
        max_steps = 30

        rng = np.random.default_rng(42)
        metric_array = rng.standard_normal((n_walks, max_steps))

        # Create violation and control events
        violation_events = _make_events(
            walk_indices=list(range(0, 10)),
            resolution_steps=[20] * 10,
            r_value=r_value,
            outcome=RuleOutcome.VIOLATED,
        )
        control_events = _make_events(
            walk_indices=list(range(10, 20)),
            resolution_steps=[20] * 10,
            r_value=r_value,
            outcome=RuleOutcome.FOLLOWED,
        )

        d_array = compute_cohens_d_by_lookback(
            violation_events, control_events, metric_array, r_value
        )
        assert d_array.shape == (r_value,)
        # All values should be finite (10 per group, random data)
        assert all(np.isfinite(d_array))


# ---------------------------------------------------------------------------
# Correlation and redundancy tests
# ---------------------------------------------------------------------------

class TestCorrelationRedundancy:
    """Tests for correlation matrices and redundancy flagging."""

    def test_measurement_correlation_matrix(self):
        """3 metrics: A = B * 2 + noise, C independent.

        Verify shape (3, 3), A-B correlation > 0.9, A-C correlation < 0.3.
        """
        rng = np.random.default_rng(42)
        n_walks = 50
        max_steps = 30

        # Create metric arrays
        base = rng.standard_normal((n_walks, max_steps))
        metric_a = base * 2 + rng.normal(0, 0.1, (n_walks, max_steps))
        metric_b = base
        metric_c = rng.standard_normal((n_walks, max_steps))

        metric_arrays = {
            "qkt.layer_0.metric_a": metric_a,
            "qkt.layer_0.metric_b": metric_b,
            "qkt.layer_0.metric_c": metric_c,
        }

        # Create events that span the metric arrays
        events = _make_events(
            walk_indices=list(range(n_walks)),
            resolution_steps=[20] * n_walks,
            r_value=5,
            outcome=RuleOutcome.VIOLATED,
        )

        result = compute_correlation_matrix(
            metric_arrays, events, mode="measurement"
        )

        assert len(result["metric_names"]) == 3
        matrix = np.array(result["matrix"])
        assert matrix.shape == (3, 3)

        # Find indices
        names = result["metric_names"]
        idx_a = names.index("qkt.layer_0.metric_a")
        idx_b = names.index("qkt.layer_0.metric_b")
        idx_c = names.index("qkt.layer_0.metric_c")

        # A-B correlation should be very high
        assert abs(matrix[idx_a, idx_b]) > 0.9, (
            f"A-B correlation {matrix[idx_a, idx_b]} should be > 0.9"
        )
        # A-C correlation should be low
        assert abs(matrix[idx_a, idx_c]) < 0.3, (
            f"A-C correlation {matrix[idx_a, idx_c]} should be < 0.3"
        )

    def test_predictive_correlation_matrix(self):
        """AUROC curves for 3 metrics. Verify correlation computed across lookback."""
        rng = np.random.default_rng(42)
        n_walks = 40
        max_steps = 30
        r_value = 5

        # Create metric arrays with different signal patterns
        metric_a = rng.standard_normal((n_walks, max_steps))
        metric_b = rng.standard_normal((n_walks, max_steps))
        metric_c = rng.standard_normal((n_walks, max_steps))

        metric_arrays = {
            "qkt.layer_0.metric_a": metric_a,
            "qkt.layer_0.metric_b": metric_b,
            "qkt.layer_0.metric_c": metric_c,
        }

        # Create violation and control events
        all_events = (
            _make_events(
                list(range(0, 20)), [20] * 20, r_value, RuleOutcome.VIOLATED
            )
            + _make_events(
                list(range(20, 40)), [20] * 40, r_value, RuleOutcome.FOLLOWED
            )
        )

        result = compute_correlation_matrix(
            metric_arrays, all_events, mode="predictive"
        )

        assert len(result["metric_names"]) == 3
        matrix = np.array(result["matrix"])
        assert matrix.shape == (3, 3)
        # Diagonal should be 1.0
        for i in range(3):
            assert matrix[i, i] == pytest.approx(1.0, abs=0.01)

    def test_redundancy_flagging(self):
        """Pair with |r| > 0.9 flagged as redundant. Pair with |r| < 0.9 not."""
        rng = np.random.default_rng(42)
        n_walks = 100
        max_steps = 30

        base = rng.standard_normal((n_walks, max_steps))
        metric_a = base
        metric_b = base + rng.normal(0, 0.05, (n_walks, max_steps))  # Near-duplicate
        metric_c = rng.standard_normal((n_walks, max_steps))  # Independent

        metric_arrays = {
            "qkt.layer_0.metric_a": metric_a,
            "qkt.layer_0.metric_b": metric_b,
            "qkt.layer_0.metric_c": metric_c,
        }

        events = _make_events(
            list(range(n_walks)), [20] * n_walks, 5, RuleOutcome.VIOLATED
        )

        result = compute_correlation_matrix(
            metric_arrays, events, mode="measurement"
        )

        # Should flag A-B as redundant
        redundant = result["redundant_pairs"]
        pair_names = set()
        for pair in redundant:
            pair_names.add(frozenset([pair[0], pair[1]]))

        expected_pair = frozenset([
            "qkt.layer_0.metric_a", "qkt.layer_0.metric_b"
        ])
        assert expected_pair in pair_names, (
            f"Expected A-B to be flagged as redundant, got {redundant}"
        )

        # Should NOT flag A-C or B-C
        unexpected_ac = frozenset([
            "qkt.layer_0.metric_a", "qkt.layer_0.metric_c"
        ])
        assert unexpected_ac not in pair_names


# ---------------------------------------------------------------------------
# Metric ranking tests
# ---------------------------------------------------------------------------

class TestMetricRanking:
    """Tests for metric importance ranking by max AUROC."""

    def test_metric_ranking_by_max_auroc(self):
        """5 metrics with different max AUROC values. Ranking descending."""
        auroc_results = {
            "metric_a": {"max_auroc": 0.9, "horizon": 5},
            "metric_b": {"max_auroc": 0.7, "horizon": 3},
            "metric_c": {"max_auroc": 0.95, "horizon": 7},
            "metric_d": {"max_auroc": 0.6, "horizon": 2},
            "metric_e": {"max_auroc": 0.85, "horizon": 4},
        }
        primary_names = ["metric_a", "metric_c"]

        result = compute_metric_ranking(
            auroc_results, primary_names, redundant_pairs=[]
        )

        # All ranking should be in descending max_auroc order
        all_ranking = result["all"]
        max_aurocs = [entry["max_auroc"] for entry in all_ranking]
        assert max_aurocs == sorted(max_aurocs, reverse=True)

        # Top should be metric_c (0.95)
        assert all_ranking[0]["metric"] == "metric_c"

    def test_metric_ranking_annotates_redundancy(self):
        """Top-ranked metric flagged as redundant with another."""
        auroc_results = {
            "metric_a": {"max_auroc": 0.95, "horizon": 7},
            "metric_b": {"max_auroc": 0.90, "horizon": 5},
            "metric_c": {"max_auroc": 0.60, "horizon": 2},
        }
        primary_names = ["metric_a"]
        redundant_pairs = [("metric_a", "metric_b", 0.95)]

        result = compute_metric_ranking(
            auroc_results, primary_names, redundant_pairs
        )

        # metric_a should be annotated as redundant with metric_b
        top_entry = result["all"][0]
        assert top_entry["metric"] == "metric_a"
        assert "metric_b" in top_entry["redundant_with"]

    def test_metric_ranking_per_layer(self):
        """Separate rankings for different layers. Different orderings allowed."""
        # Layer 0: metric_a leads
        auroc_results_l0 = {
            "qkt.layer_0.metric_a": {"max_auroc": 0.9, "horizon": 5},
            "qkt.layer_0.metric_b": {"max_auroc": 0.7, "horizon": 3},
        }
        # Layer 1: metric_b leads
        auroc_results_l1 = {
            "qkt.layer_1.metric_a": {"max_auroc": 0.65, "horizon": 2},
            "qkt.layer_1.metric_b": {"max_auroc": 0.85, "horizon": 4},
        }

        result_l0 = compute_metric_ranking(auroc_results_l0, [], [])
        result_l1 = compute_metric_ranking(auroc_results_l1, [], [])

        # Layer 0 top = metric_a
        assert result_l0["all"][0]["metric"] == "qkt.layer_0.metric_a"
        # Layer 1 top = metric_b
        assert result_l1["all"][0]["metric"] == "qkt.layer_1.metric_b"


# ---------------------------------------------------------------------------
# Headline comparison tests
# ---------------------------------------------------------------------------

class TestHeadlineComparison:
    """Tests for QK^T vs AVWo predictive horizon comparison."""

    def test_headline_comparison_qkt_leads(self):
        """QK^T horizon=12, AVWo horizon=5. qkt_leads=True, gap=7."""
        auroc_results = {
            "by_r_value": {
                8: {
                    "by_metric": {
                        "qkt.layer_0.grassmannian_distance": {
                            "horizon": 12,
                            "max_auroc": 0.85,
                            "is_primary": True,
                        },
                        "qkt.layer_0.spectral_entropy": {
                            "horizon": 10,
                            "max_auroc": 0.80,
                            "is_primary": True,
                        },
                        "avwo.layer_0.stable_rank": {
                            "horizon": 5,
                            "max_auroc": 0.75,
                            "is_primary": True,
                        },
                        "avwo.layer_0.grassmannian_distance": {
                            "horizon": 4,
                            "max_auroc": 0.70,
                            "is_primary": True,
                        },
                    },
                }
            }
        }
        primary_metrics = list(PRIMARY_METRICS)

        result = compute_headline_comparison(auroc_results, primary_metrics)

        r8 = result["by_r_value"][8]
        assert r8["qkt_max_horizon"] == 12
        assert r8["avwo_max_horizon"] == 5
        assert r8["qkt_leads"] is True
        assert r8["gap"] == 7

    def test_headline_comparison_avwo_leads(self):
        """AVWo horizon > QK^T horizon. qkt_leads=False."""
        auroc_results = {
            "by_r_value": {
                4: {
                    "by_metric": {
                        "qkt.layer_0.grassmannian_distance": {
                            "horizon": 2,
                            "max_auroc": 0.60,
                            "is_primary": True,
                        },
                        "avwo.layer_0.stable_rank": {
                            "horizon": 3,
                            "max_auroc": 0.80,
                            "is_primary": True,
                        },
                    },
                }
            }
        }
        primary_metrics = list(PRIMARY_METRICS)

        result = compute_headline_comparison(auroc_results, primary_metrics)

        r4 = result["by_r_value"][4]
        assert r4["qkt_leads"] is False
        assert r4["gap"] == -1  # avwo leads by 1


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestApplyStatisticalControls:
    """Integration test for the full statistical controls pipeline."""

    def test_apply_statistical_controls_to_auroc_results(self):
        """Take mock AUROC results (as from run_auroc_analysis), apply all
        statistical controls. Verify output has required fields and is
        JSON-serializable.
        """
        rng = np.random.default_rng(42)
        n_sequences = 40
        max_steps = 50
        r_value = 8

        jumper = JumperInfo(vertex_id=5, source_block=0, target_block=1, r=r_value)
        jumper_map = {5: jumper}

        # Build eval_data matching run_auroc_analysis expectations
        generated = np.zeros((n_sequences, max_steps), dtype=np.int64)
        generated[:, 20] = 5  # Place jumper at step 20

        rule_outcome = np.full(
            (n_sequences, max_steps - 1), RuleOutcome.NOT_APPLICABLE, dtype=np.int32
        )
        for w in range(15):
            rule_outcome[w, 27] = RuleOutcome.VIOLATED
        for w in range(15, 40):
            rule_outcome[w, 27] = RuleOutcome.FOLLOWED

        failure_index = np.full(n_sequences, -1, dtype=np.int32)
        for w in range(15):
            failure_index[w] = 27

        sequence_lengths = np.full(n_sequences, max_steps, dtype=np.int32)

        # Create one primary QK^T metric and one primary AVWo metric
        qkt_key = "qkt.layer_0.grassmannian_distance"
        avwo_key = "avwo.layer_0.stable_rank"

        metric_qkt = rng.standard_normal((n_sequences, max_steps - 1)).astype(np.float32)
        metric_avwo = rng.standard_normal((n_sequences, max_steps - 1)).astype(np.float32)

        # Inject signal for QK^T at lookback j=1..3
        for w in range(15):
            for j in range(1, 4):
                metric_qkt[w, 28 - j] = 5.0 + rng.normal(0, 0.1)
        for w in range(15, 40):
            for j in range(1, 4):
                metric_qkt[w, 28 - j] = 1.0 + rng.normal(0, 0.1)

        eval_data = {
            "generated": generated,
            "rule_outcome": rule_outcome,
            "failure_index": failure_index,
            "sequence_lengths": sequence_lengths,
            qkt_key: metric_qkt,
            avwo_key: metric_avwo,
        }

        result = apply_statistical_controls(
            auroc_results=None,  # Will compute internally
            eval_data=eval_data,
            jumper_map=jumper_map,
            n_bootstrap=200,  # Small for test speed
            confidence_level=0.95,
            bootstrap_rng=42,
        )

        # Verify required top-level keys
        assert "config" in result
        assert "contamination_audit" in result
        assert "by_r_value" in result
        assert "correlation_matrices" in result
        assert "metric_ranking" in result
        assert "headline_comparison" in result

        # Verify bootstrap CIs are present for metrics with enough events
        r_block = result["by_r_value"][r_value]
        assert "by_metric" in r_block
        for metric_key, metric_data in r_block["by_metric"].items():
            if metric_data.get("event_tier_for_bootstrap") == "full":
                assert "bootstrap_ci" in metric_data
                ci = metric_data["bootstrap_ci"]
                assert len(ci) == 2
                assert ci[0] <= ci[1]

        # Verify Holm-Bonferroni results
        assert "holm_bonferroni" in result
        hb = result["holm_bonferroni"]
        assert "adjusted_p_values" in hb
        assert "reject" in hb

        # Verify JSON-serializable
        json_str = json.dumps(result, default=str)
        assert len(json_str) > 0
