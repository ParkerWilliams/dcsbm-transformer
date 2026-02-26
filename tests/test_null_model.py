"""Tests for null model baseline: null walk generation, drift extraction, and MP reference.

RED phase -- these tests should fail until src/analysis/null_model.py is implemented.
"""

import numpy as np
import pytest
import scipy.sparse
from scipy.integrate import quad

from src.analysis.null_model import (
    compare_null_vs_violation,
    extract_position_matched_drift,
    generate_null_walks,
    marchenko_pastur_cdf,
    marchenko_pastur_pdf,
    run_mp_ks_test,
)
from src.config.experiment import ExperimentConfig, GraphConfig, TrainingConfig
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_graph_with_jumpers():
    """Build a small DCSBM-like graph (n=20, K=2) with 2 jumper vertices.

    Creates a directed adjacency matrix with dense intra-block connectivity
    and sparse inter-block connectivity, then designates one jumper per block.
    """
    n = 20
    K = 2
    block_size = n // K  # 10

    # Block assignments: first 10 in block 0, next 10 in block 1
    block_assignments = np.array([0] * block_size + [1] * block_size, dtype=np.int32)

    # Build adjacency with high intra-block, low inter-block connectivity
    rng = np.random.default_rng(42)
    dense = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if block_assignments[i] == block_assignments[j]:
                # Intra-block: high probability
                if rng.random() < 0.5:
                    dense[i, j] = 1.0
            else:
                # Inter-block: low probability
                if rng.random() < 0.1:
                    dense[i, j] = 1.0

    adjacency = scipy.sparse.csr_matrix(dense)

    # Degree correction (uniform for simplicity)
    theta = np.ones(n, dtype=np.float64)

    graph_data = GraphData(
        adjacency=adjacency,
        block_assignments=block_assignments,
        theta=theta,
        n=n,
        K=K,
        block_size=block_size,
        generation_seed=42,
        attempt=0,
    )

    # Designate 2 jumpers: vertex 3 (block 0 -> block 1) and vertex 13 (block 1 -> block 0)
    jumpers = [
        JumperInfo(vertex_id=3, source_block=0, target_block=1, r=3),
        JumperInfo(vertex_id=13, source_block=1, target_block=0, r=3),
    ]

    # Config with small walk length
    config = ExperimentConfig(
        graph=GraphConfig(n=n, K=K, n_jumpers_per_block=1),
        training=TrainingConfig(
            w=10,
            walk_length=30,
            corpus_size=50_000,
            r=3,
        ),
    )

    return graph_data, jumpers, config


# ---------------------------------------------------------------------------
# Null Walk Generation Tests (1-6)
# ---------------------------------------------------------------------------


class TestGenerateNullWalks:
    """Tests for generate_null_walks()."""

    def test_no_jumpers_visited(self, small_graph_with_jumpers):
        """Test 1: No walk visits any jumper vertex."""
        graph_data, jumpers, config = small_graph_with_jumpers
        jumper_set = {j.vertex_id for j in jumpers}

        null_walks = generate_null_walks(
            graph_data, jumpers, config, n_walks=50, seed=123
        )

        visited = set(null_walks.flatten().tolist())
        assert visited & jumper_set == set(), (
            f"Null walks visited jumper vertices: {visited & jumper_set}"
        )

    def test_correct_count(self, small_graph_with_jumpers):
        """Test 2: Output shape matches requested n_walks."""
        graph_data, jumpers, config = small_graph_with_jumpers

        null_walks = generate_null_walks(
            graph_data, jumpers, config, n_walks=50, seed=456
        )

        assert null_walks.shape == (50, config.training.walk_length)

    def test_edge_validity(self, small_graph_with_jumpers):
        """Test 3: Consecutive vertices in each walk share an edge."""
        graph_data, jumpers, config = small_graph_with_jumpers

        null_walks = generate_null_walks(
            graph_data, jumpers, config, n_walks=20, seed=789
        )

        indptr = graph_data.adjacency.indptr
        indices = graph_data.adjacency.indices

        # Check at least 10 walks
        for wi in range(min(10, null_walks.shape[0])):
            for step in range(null_walks.shape[1] - 1):
                u = null_walks[wi, step]
                v = null_walks[wi, step + 1]
                neighbors = indices[indptr[u]:indptr[u + 1]]
                assert v in neighbors, (
                    f"Walk {wi} step {step}: edge {u}->{v} not in graph"
                )

    def test_deterministic(self, small_graph_with_jumpers):
        """Test 4: Same seed produces identical walks."""
        graph_data, jumpers, config = small_graph_with_jumpers

        walks_a = generate_null_walks(
            graph_data, jumpers, config, n_walks=20, seed=42
        )
        walks_b = generate_null_walks(
            graph_data, jumpers, config, n_walks=20, seed=42
        )

        np.testing.assert_array_equal(walks_a, walks_b)

    def test_different_seeds(self, small_graph_with_jumpers):
        """Test 5: Different seeds produce different walks."""
        graph_data, jumpers, config = small_graph_with_jumpers

        walks_a = generate_null_walks(
            graph_data, jumpers, config, n_walks=20, seed=42
        )
        walks_b = generate_null_walks(
            graph_data, jumpers, config, n_walks=20, seed=99
        )

        assert not np.array_equal(walks_a, walks_b)

    def test_handles_dead_ends(self):
        """Test 6: Handles graph where jumper removal creates dead-end vertices.

        Creates a graph where vertex 2 ONLY connects to jumper vertex 0,
        so removing jumper columns yields a dead-end at vertex 2.
        """
        n = 6
        K = 2
        block_size = 3

        block_assignments = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

        # Build a specific adjacency: vertex 2 only has edge to vertex 0 (jumper)
        # Other vertices have edges among themselves (within blocks)
        dense = np.zeros((n, n), dtype=np.float64)
        # Block 0 internal (excluding jumper-dependent edges)
        dense[0, 1] = 1
        dense[1, 0] = 1
        dense[1, 2] = 1
        dense[2, 0] = 1  # vertex 2 only connects to vertex 0 (jumper)
        # Block 1 internal
        dense[3, 4] = 1
        dense[4, 3] = 1
        dense[4, 5] = 1
        dense[5, 3] = 1
        dense[5, 4] = 1
        dense[3, 5] = 1
        # Some inter-block edges
        dense[1, 3] = 1
        dense[3, 1] = 1

        adjacency = scipy.sparse.csr_matrix(dense)
        theta = np.ones(n, dtype=np.float64)

        graph_data = GraphData(
            adjacency=adjacency,
            block_assignments=block_assignments,
            theta=theta,
            n=n,
            K=K,
            block_size=block_size,
            generation_seed=42,
            attempt=0,
        )

        # Vertex 0 is the jumper
        jumpers = [JumperInfo(vertex_id=0, source_block=0, target_block=1, r=2)]

        config = ExperimentConfig(
            graph=GraphConfig(n=n, K=K, n_jumpers_per_block=1),
            training=TrainingConfig(
                w=4,
                walk_length=10,
                corpus_size=50_000,
                r=2,
            ),
        )

        # Should not hang or crash -- vertex 2 is a dead end after filtering
        null_walks = generate_null_walks(
            graph_data, jumpers, config, n_walks=10, seed=42
        )

        assert null_walks.shape[0] == 10
        assert null_walks.shape[1] == config.training.walk_length

        # No walk should visit jumper vertex 0
        assert 0 not in set(null_walks.flatten().tolist())


# ---------------------------------------------------------------------------
# Position-Matched Drift Extraction Tests (7-9)
# ---------------------------------------------------------------------------


class TestExtractPositionMatchedDrift:
    """Tests for extract_position_matched_drift()."""

    def test_correct_positions(self):
        """Test 7: Extracted values match expected positions."""
        n_seq = 10
        n_steps = 100
        rng = np.random.default_rng(42)
        metric_array = rng.random((n_seq, n_steps)).astype(np.float32)

        event_positions = [30, 50, 70]
        max_lookback = 5

        result = extract_position_matched_drift(
            metric_array, event_positions, max_lookback
        )

        # For each lookback j, verify values match
        for j in range(1, max_lookback + 1):
            assert j in result, f"Missing lookback distance j={j}"
            expected_values = []
            for pos in event_positions:
                idx = pos - j
                if 0 <= idx < n_steps:
                    expected_values.append(metric_array[:, idx])
            expected = np.concatenate(expected_values)
            expected = expected[np.isfinite(expected)]
            np.testing.assert_array_almost_equal(
                np.sort(result[j]),
                np.sort(expected),
                err_msg=f"Mismatch at lookback j={j}",
            )

    def test_nan_filtering(self):
        """Test 8: NaN values are filtered from output."""
        n_seq = 10
        n_steps = 100
        rng = np.random.default_rng(42)
        metric_array = rng.random((n_seq, n_steps)).astype(np.float32)

        # Inject NaN at specific positions
        metric_array[0, 25] = np.nan
        metric_array[3, 25] = np.nan
        metric_array[5, 45] = np.nan

        event_positions = [30, 50]
        max_lookback = 5

        result = extract_position_matched_drift(
            metric_array, event_positions, max_lookback
        )

        for j, values in result.items():
            assert np.all(np.isfinite(values)), (
                f"Non-finite values at lookback j={j}"
            )

    def test_out_of_bounds(self):
        """Test 9: Gracefully skips out-of-bounds indices."""
        n_seq = 5
        n_steps = 20
        rng = np.random.default_rng(42)
        metric_array = rng.random((n_seq, n_steps)).astype(np.float32)

        # Position 3 with lookback j=5 would give idx=-2 (out of bounds)
        event_positions = [3]
        max_lookback = 5

        result = extract_position_matched_drift(
            metric_array, event_positions, max_lookback
        )

        # j=1,2,3 should have values (pos 2, 1, 0 respectively)
        for j in [1, 2, 3]:
            assert j in result
            assert len(result[j]) > 0

        # j=4 gives idx=-1 (out of bounds) -- should be empty or not present
        # j=5 gives idx=-2 (out of bounds) -- should be empty or not present
        for j in [4, 5]:
            if j in result:
                assert len(result[j]) == 0, (
                    f"Lookback j={j} should have no values for position 3"
                )


# ---------------------------------------------------------------------------
# Marchenko-Pastur Tests (10-16)
# ---------------------------------------------------------------------------


class TestMarchenkoPasturPDF:
    """Tests for marchenko_pastur_pdf()."""

    def test_support(self):
        """Test 10: PDF is zero outside support, positive inside."""
        gamma = 0.5
        sigma2 = 1.0
        lam_minus = (1 - np.sqrt(gamma)) ** 2
        lam_plus = (1 + np.sqrt(gamma)) ** 2

        # Outside support: should be 0
        assert marchenko_pastur_pdf(lam_minus - 0.01, gamma, sigma2) == 0.0
        assert marchenko_pastur_pdf(lam_plus + 0.01, gamma, sigma2) == 0.0

        # Inside support: should be > 0
        midpoint = (lam_minus + lam_plus) / 2
        assert marchenko_pastur_pdf(midpoint, gamma, sigma2) > 0.0

    def test_integrates_to_one(self):
        """Test 11: PDF integrates to 1 over its support."""
        gamma = 0.5
        sigma2 = 1.0
        lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
        lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

        integral, _ = quad(marchenko_pastur_pdf, lam_minus, lam_plus, args=(gamma, sigma2))
        assert abs(integral - 1.0) < 0.001, f"MP PDF integral = {integral}, expected 1.0"


class TestMarchenkoPasturCDF:
    """Tests for marchenko_pastur_cdf()."""

    def test_monotone(self):
        """Test 12: CDF is non-decreasing, starts near 0, ends near 1."""
        gamma = 0.5
        sigma2 = 1.0
        lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
        lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

        points = np.linspace(lam_minus + 1e-6, lam_plus - 1e-6, 10)
        cdf_values = [marchenko_pastur_cdf(x, gamma, sigma2) for x in points]

        # Non-decreasing
        for i in range(1, len(cdf_values)):
            assert cdf_values[i] >= cdf_values[i - 1] - 1e-10, (
                f"CDF not monotone at index {i}: {cdf_values[i]} < {cdf_values[i - 1]}"
            )

        # Start near 0, end near 1
        assert cdf_values[0] < 0.15, f"CDF start too high: {cdf_values[0]}"
        assert cdf_values[-1] > 0.85, f"CDF end too low: {cdf_values[-1]}"

    def test_boundaries(self):
        """Test 13: CDF is 0 at lambda_minus and 1 at lambda_plus."""
        gamma = 0.5
        sigma2 = 1.0
        lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
        lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

        assert marchenko_pastur_cdf(lam_minus - 0.01, gamma, sigma2) == 0.0
        assert marchenko_pastur_cdf(lam_minus, gamma, sigma2) == 0.0
        assert marchenko_pastur_cdf(lam_plus, gamma, sigma2) == 1.0
        assert marchenko_pastur_cdf(lam_plus + 0.01, gamma, sigma2) == 1.0


class TestRunMPKSTest:
    """Tests for run_mp_ks_test()."""

    def test_random_matrix(self):
        """Test 14: Random Wishart-like matrix matches MP (high p-value)."""
        rng = np.random.default_rng(42)
        w = 64
        d_k = 128
        gamma = w / d_k  # 0.5

        # Generate random matrix and compute Wishart-like eigenvalues
        X = rng.standard_normal((w, d_k))
        # Eigenvalues of X @ X^T / d_k follow MP
        M = X @ X.T / d_k
        eigenvalues = np.linalg.eigvalsh(M)
        # Eigenvalues are already squared singular values
        # Pass as "singular values" = sqrt(eigenvalues)
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))

        result = run_mp_ks_test(singular_values, gamma)

        assert result["ks_p_value"] > 0.01, (
            f"Random matrix should not be rejected by MP, p={result['ks_p_value']}"
        )

    def test_structured_matrix(self):
        """Test 15: Structured matrix departs from MP (low p-value)."""
        rng = np.random.default_rng(42)
        w = 64
        d_k = 128
        gamma = w / d_k

        # Random base + rank-1 structure
        X = rng.standard_normal((w, d_k))
        # Add dominant rank-1 component
        u = rng.standard_normal(w)
        v = rng.standard_normal(d_k)
        X += 10.0 * np.outer(u, v)

        M = X @ X.T / d_k
        eigenvalues = np.linalg.eigvalsh(M)
        singular_values = np.sqrt(np.maximum(eigenvalues, 0))

        result = run_mp_ks_test(singular_values, gamma)

        assert result["ks_p_value"] < 0.05, (
            f"Structured matrix should be rejected by MP, p={result['ks_p_value']}"
        )

    def test_returns_expected_keys(self):
        """Test 16: Returned dict contains all expected keys."""
        rng = np.random.default_rng(42)
        singular_values = rng.random(30)
        gamma = 0.5

        result = run_mp_ks_test(singular_values, gamma)

        expected_keys = {
            "ks_statistic", "ks_p_value", "gamma", "sigma2",
            "lambda_minus", "lambda_plus",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )


# ---------------------------------------------------------------------------
# Statistical Comparison Tests (17-21)
# ---------------------------------------------------------------------------


class TestCompareNullVsViolation:
    """Tests for compare_null_vs_violation()."""

    def test_basic(self):
        """Test 17: Basic comparison returns correct structure and fields."""
        rng = np.random.default_rng(42)

        # Synthetic null drift: low values, small variance
        null_drift = {
            j: rng.normal(0.1, 0.02, size=50).astype(np.float32)
            for j in range(1, 6)
        }
        # Synthetic violation drift: higher values, larger variance
        violation_drift = {
            j: rng.normal(0.5, 0.1, size=30).astype(np.float32)
            for j in range(1, 6)
        }

        result = compare_null_vs_violation(null_drift, violation_drift)

        # Check top-level keys
        assert "by_lookback" in result
        assert "aggregate" in result

        # Check per-lookback structure
        for j in range(1, 6):
            key = str(j)
            assert key in result["by_lookback"], f"Missing lookback j={j}"
            entry = result["by_lookback"][key]
            expected_fields = {
                "null_mean", "null_std", "violation_mean", "violation_std",
                "mann_whitney_U", "p_value_raw", "p_value_adjusted",
                "cohens_d", "reject", "n_null_valid", "n_violation_valid",
            }
            assert expected_fields.issubset(entry.keys()), (
                f"Missing fields at j={j}: {expected_fields - entry.keys()}"
            )

        # Check aggregate structure
        agg = result["aggregate"]
        agg_fields = {
            "n_lookbacks_tested", "n_lookbacks_rejected",
            "max_cohens_d", "max_cohens_d_lookback", "signal_exceeds_noise",
        }
        assert agg_fields.issubset(agg.keys()), (
            f"Missing aggregate fields: {agg_fields - agg.keys()}"
        )

        # Consistency check: tested count matches number of lookback distances
        assert agg["n_lookbacks_tested"] == 5
        assert agg["n_lookbacks_rejected"] <= agg["n_lookbacks_tested"]

    def test_separate_family(self):
        """Test 18: Holm-Bonferroni applied ONLY across lookback distances in null comparison.

        Verifies the number of adjusted p-values equals the number of tested
        lookback distances (not mixed with any external family).
        """
        rng = np.random.default_rng(42)

        null_drift = {
            j: rng.normal(0.1, 0.02, size=50).astype(np.float32)
            for j in range(1, 8)  # 7 lookback distances
        }
        violation_drift = {
            j: rng.normal(0.5, 0.1, size=30).astype(np.float32)
            for j in range(1, 8)
        }

        result = compare_null_vs_violation(null_drift, violation_drift)

        # Count adjusted p-values that are not NaN
        adjusted_count = sum(
            1 for j in range(1, 8)
            if np.isfinite(result["by_lookback"][str(j)]["p_value_adjusted"])
        )

        # Should equal the number of tested lookbacks (all 7 have enough samples)
        assert adjusted_count == 7, (
            f"Expected 7 adjusted p-values, got {adjusted_count}"
        )
        assert result["aggregate"]["n_lookbacks_tested"] == 7

    def test_insufficient_samples(self):
        """Test 19: Lookback with fewer than 5 samples is excluded from HB correction."""
        rng = np.random.default_rng(42)

        null_drift = {
            1: rng.normal(0.1, 0.02, size=50).astype(np.float32),
            2: rng.normal(0.1, 0.02, size=3).astype(np.float32),   # < 5 samples
            3: rng.normal(0.1, 0.02, size=50).astype(np.float32),
        }
        violation_drift = {
            1: rng.normal(0.5, 0.1, size=30).astype(np.float32),
            2: rng.normal(0.5, 0.1, size=30).astype(np.float32),
            3: rng.normal(0.5, 0.1, size=30).astype(np.float32),
        }

        result = compare_null_vs_violation(null_drift, violation_drift)

        # Lookback j=2 should be marked insufficient
        assert result["by_lookback"]["2"]["insufficient_samples"] is True
        assert np.isnan(result["by_lookback"]["2"]["p_value_adjusted"])
        assert result["by_lookback"]["2"]["reject"] is False

        # Lookbacks j=1 and j=3 should be tested
        assert result["by_lookback"]["1"]["insufficient_samples"] is False
        assert result["by_lookback"]["3"]["insufficient_samples"] is False

        # Only 2 lookbacks should be in the HB family
        assert result["aggregate"]["n_lookbacks_tested"] == 2

    def test_signal_exceeds_noise(self):
        """Test 20: Clear separation triggers signal_exceeds_noise = True."""
        rng = np.random.default_rng(42)

        # Null: tight distribution around 0.1
        null_drift = {
            j: rng.normal(0.1, 0.01, size=100).astype(np.float32)
            for j in range(1, 4)
        }
        # Violation: clearly separated at 0.5 (Cohen's d >> 0.5)
        violation_drift = {
            j: rng.normal(0.5, 0.01, size=100).astype(np.float32)
            for j in range(1, 4)
        }

        result = compare_null_vs_violation(null_drift, violation_drift)

        assert result["aggregate"]["signal_exceeds_noise"] is True
        assert result["aggregate"]["n_lookbacks_rejected"] > 0
        assert result["aggregate"]["max_cohens_d"] > 0.5

    def test_no_signal(self):
        """Test 21: Identical distributions yield signal_exceeds_noise = False."""
        rng = np.random.default_rng(42)

        # Same distribution for both null and violation
        null_drift = {
            j: rng.normal(0.3, 0.05, size=50).astype(np.float32)
            for j in range(1, 4)
        }
        violation_drift = {
            j: rng.normal(0.3, 0.05, size=50).astype(np.float32)
            for j in range(1, 4)
        }

        result = compare_null_vs_violation(null_drift, violation_drift)

        assert result["aggregate"]["signal_exceeds_noise"] is False
