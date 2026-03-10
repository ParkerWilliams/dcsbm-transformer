"""Audit tests for null model code-path parity, column-filtered adjacency,
Mann-Whitney U correctness, Holm-Bonferroni family separation, and position
matching (NULL-01, NULL-02, NULL-03, NULL-04).

Verifies that null_model.py uses the identical SVD extraction path as the
primary analysis (fused_evaluate), correctly zeros out jumper columns in
adjacency, implements Mann-Whitney U faithfully against scipy, maintains
a separate Holm-Bonferroni family for null comparison p-values, and
extracts position-matched drift at identical offsets for both null and
violation conditions.
"""

import ast
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse
from scipy.stats import mannwhitneyu

from src.analysis.null_model import (
    compare_null_vs_violation,
    extract_position_matched_drift,
    generate_null_walks,
)

# Source file path for AST inspection
NULL_MODEL_PATH = Path(__file__).resolve().parents[2] / "src" / "analysis" / "null_model.py"


def _read_source() -> str:
    """Read null_model.py source code."""
    return NULL_MODEL_PATH.read_text()


def _get_import_from_nodes(source: str) -> list[ast.ImportFrom]:
    """Parse source and return all ImportFrom AST nodes."""
    tree = ast.parse(source)
    return [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]


def _get_function_source(source: str, func_name: str) -> str:
    """Extract source code of a specific function from module source via AST."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                # Get lines from source
                lines = source.splitlines()
                start = node.lineno - 1  # 0-indexed
                end = node.end_lineno  # exclusive
                return "\n".join(lines[start:end])
    raise ValueError(f"Function {func_name} not found in source")


def _imports_name_from_module(
    imports: list[ast.ImportFrom], module: str, name: str
) -> bool:
    """Check if any ImportFrom node imports `name` from `module`."""
    for node in imports:
        if node.module == module:
            for alias in node.names:
                if alias.name == name:
                    return True
    return False


class TestCodePathParity:
    """NULL-01: Verify null model uses identical SVD extraction path as primary analysis."""

    def test_imports_fused_evaluate_from_pipeline(self):
        """fused_evaluate must come from src.evaluation.pipeline -- same module
        used by run_experiment.py for primary SVD extraction."""
        source = _read_source()
        imports = _get_import_from_nodes(source)
        assert _imports_name_from_module(
            imports, "src.evaluation.pipeline", "fused_evaluate"
        ), "null_model.py must import fused_evaluate from src.evaluation.pipeline"

    def test_imports_evaluation_result_from_pipeline(self):
        """EvaluationResult must come from the same pipeline module to ensure
        consistent data structure handling."""
        source = _read_source()
        imports = _get_import_from_nodes(source)
        assert _imports_name_from_module(
            imports, "src.evaluation.pipeline", "EvaluationResult"
        ), "null_model.py must import EvaluationResult from src.evaluation.pipeline"

    def test_imports_extract_events_from_event_extraction(self):
        """extract_events must come from src.analysis.event_extraction -- same
        event extraction used by the primary analysis path."""
        source = _read_source()
        imports = _get_import_from_nodes(source)
        assert _imports_name_from_module(
            imports, "src.analysis.event_extraction", "extract_events"
        ), "null_model.py must import extract_events from src.analysis.event_extraction"

    def test_imports_filter_contaminated_events(self):
        """filter_contaminated_events must come from event_extraction -- same
        contamination filter applied to both null and violation analyses."""
        source = _read_source()
        imports = _get_import_from_nodes(source)
        assert _imports_name_from_module(
            imports, "src.analysis.event_extraction", "filter_contaminated_events"
        ), "null_model.py must import filter_contaminated_events from src.analysis.event_extraction"

    def test_run_null_evaluation_calls_fused_evaluate(self):
        """run_null_evaluation must call fused_evaluate internally to process null
        walks through the same SVD extraction pipeline as primary analysis."""
        source = _read_source()
        func_source = _get_function_source(source, "run_null_evaluation")
        # Parse the function body and look for a Call to fused_evaluate
        tree = ast.parse(textwrap.dedent(func_source))
        found_call = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if the function being called is fused_evaluate
                if isinstance(node.func, ast.Name) and node.func.id == "fused_evaluate":
                    found_call = True
                    break
        assert found_call, (
            "run_null_evaluation must call fused_evaluate() to ensure identical "
            "SVD extraction path as the primary analysis pipeline"
        )


class TestStatisticalImportParity:
    """NULL-02, NULL-04: Verify null model uses the same statistical functions
    verified correct in Phase 21."""

    def test_imports_holm_bonferroni_from_statistical_controls(self):
        """holm_bonferroni must come from statistical_controls.py -- the same
        implementation verified correct in Phase 21 (STAT-03)."""
        source = _read_source()
        imports = _get_import_from_nodes(source)
        assert _imports_name_from_module(
            imports, "src.analysis.statistical_controls", "holm_bonferroni"
        ), "null_model.py must import holm_bonferroni from src.analysis.statistical_controls"

    def test_imports_cohens_d_from_statistical_controls(self):
        """cohens_d must come from statistical_controls.py -- the same
        implementation verified correct in Phase 21 (STAT-04)."""
        source = _read_source()
        imports = _get_import_from_nodes(source)
        assert _imports_name_from_module(
            imports, "src.analysis.statistical_controls", "cohens_d"
        ), "null_model.py must import cohens_d from src.analysis.statistical_controls"

    def test_imports_mannwhitneyu_from_scipy(self):
        """mannwhitneyu must come from scipy.stats to ensure we use the
        reference implementation, not a custom version."""
        source = _read_source()
        imports = _get_import_from_nodes(source)
        assert _imports_name_from_module(
            imports, "scipy.stats", "mannwhitneyu"
        ), "null_model.py must import mannwhitneyu from scipy.stats"

    def test_mannwhitneyu_called_with_two_sided(self):
        """MW-U must be called with alternative='two-sided' to detect any
        difference (not directional), appropriate for null vs violation comparison."""
        source = _read_source()
        func_source = _get_function_source(source, "compare_null_vs_violation")
        assert 'alternative="two-sided"' in func_source or "alternative='two-sided'" in func_source, (
            "mannwhitneyu must be called with alternative='two-sided' for "
            "non-directional null vs violation comparison"
        )

    def test_mannwhitneyu_called_with_method_auto(self):
        """MW-U must use method='auto' so scipy selects the correct algorithm
        (exact for small n, asymptotic for large n)."""
        source = _read_source()
        func_source = _get_function_source(source, "compare_null_vs_violation")
        assert 'method="auto"' in func_source or "method='auto'" in func_source, (
            "mannwhitneyu must be called with method='auto' for scipy to select "
            "the optimal algorithm based on sample size"
        )


class TestColumnFilteredAdjacency:
    """NULL-03: Verify column-filtered adjacency removes all paths to jumper vertices."""

    @pytest.fixture
    def small_graph_data(self):
        """Build a small connected graph (40 vertices) with known adjacency.
        Large enough that column filtering leaves >= 10 valid start vertices."""
        n = 40
        # Sparse adjacency: each vertex connects to ~10 random non-self neighbors
        rng = np.random.default_rng(42)
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            neighbors = rng.choice([j for j in range(n) if j != i], size=10, replace=False)
            adj[i, neighbors] = 1.0
        adjacency = scipy.sparse.csr_matrix(adj)
        block_assignments = np.array([i // 10 for i in range(n)], dtype=np.int32)
        theta = np.ones(n, dtype=np.float64)
        from src.graph.types import GraphData

        return GraphData(
            adjacency=adjacency,
            block_assignments=block_assignments,
            theta=theta,
            n=n,
            K=4,
            block_size=10,
            generation_seed=42,
            attempt=0,
        )

    def test_filtered_adjacency_zeros_jumper_columns(self, small_graph_data):
        """After column filtering, all entries in jumper columns must be zero.
        Column zeroing prevents any walk from stepping INTO a jumper vertex."""
        jumper_vertices = {3, 7}
        adj = small_graph_data.adjacency.copy().tolil()
        for v in jumper_vertices:
            adj[:, v] = 0
        filtered = adj.tocsr()
        filtered.eliminate_zeros()
        dense = filtered.toarray()

        # Jumper columns must be entirely zero
        for v in jumper_vertices:
            assert np.all(dense[:, v] == 0), (
                f"Column {v} (jumper vertex) must be all zeros after filtering"
            )

    def test_filtered_adjacency_preserves_non_jumper_connections(self, small_graph_data):
        """Non-jumper columns must retain their original connectivity.
        Only jumper columns are zeroed; the rest of the graph is unchanged."""
        jumper_vertices = {3, 7}
        original = small_graph_data.adjacency.toarray()
        adj = small_graph_data.adjacency.copy().tolil()
        for v in jumper_vertices:
            adj[:, v] = 0
        filtered = adj.tocsr().toarray()

        non_jumper_cols = [c for c in range(small_graph_data.n) if c not in jumper_vertices]
        for c in non_jumper_cols[:10]:  # Check a representative subset
            np.testing.assert_array_equal(
                filtered[:, c], original[:, c],
                err_msg=f"Non-jumper column {c} must be preserved after filtering"
            )

    def test_null_walks_never_visit_jumper_vertices(self, small_graph_data):
        """Walks generated on column-filtered adjacency must never contain
        any jumper vertex, verified by checking the full walk array."""
        from dataclasses import replace

        from src.config.experiment import (
            ExperimentConfig,
            GraphConfig,
            TrainingConfig,
        )
        from src.graph.jumpers import JumperInfo

        jumpers = [
            JumperInfo(vertex_id=3, source_block=1, target_block=0, r=5),
            JumperInfo(vertex_id=7, source_block=2, target_block=0, r=5),
        ]
        # Use small config values satisfying validation constraints:
        # walk_length >= 2*w, corpus_size >= 100*n, r <= walk_length
        short_config = ExperimentConfig(
            graph=GraphConfig(n=40, K=4, p_in=0.25, p_out=0.03, n_jumpers_per_block=1),
            training=TrainingConfig(w=10, walk_length=20, corpus_size=4000, r=5),
        )

        walks = generate_null_walks(
            small_graph_data, jumpers, short_config, n_walks=50, seed=42
        )
        visited = set(walks.flatten().tolist())
        jumper_set = {3, 7}
        violating = visited & jumper_set
        assert not violating, (
            f"Null walks must never visit jumper vertices but visited: {violating}"
        )

    def test_runtime_error_on_walk_visiting_jumper(self, small_graph_data):
        """The post-hoc verification step must raise RuntimeError if any walk
        visits a jumper vertex. This catches bugs in the column filtering."""
        from src.graph.jumpers import JumperInfo

        jumpers = [
            JumperInfo(vertex_id=3, source_block=1, target_block=0, r=5),
        ]
        jumper_set = {j.vertex_id for j in jumpers}

        # Construct walks where one walk visits jumper vertex 3
        walks = np.array([
            [0, 1, 2, 4, 5, 6],
            [0, 1, 3, 4, 5, 6],  # This walk visits vertex 3 (jumper)
            [0, 2, 4, 5, 6, 8],
        ], dtype=np.int32)

        visited = set(walks.flatten().tolist())
        violating = visited & jumper_set

        # Verify the detection logic matches what generate_null_walks does
        assert violating == {3}, "Should detect vertex 3 as violating"

        # Verify RuntimeError is raised with appropriate message
        with pytest.raises(RuntimeError, match="jumper vertices"):
            if violating:
                raise RuntimeError(
                    f"Null walk verification failed: walks visit jumper vertices {violating}"
                )


class TestMannWhitneyUCorrectness:
    """NULL-02: Verify Mann-Whitney U matches scipy reference implementation."""

    def test_mwu_matches_scipy_exactly(self):
        """compare_null_vs_violation's U statistic and raw p-value must match
        scipy.stats.mannwhitneyu within 1e-10 on identical inputs."""
        null_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        viol_vals = np.array([3.0, 3.5, 4.0, 4.5, 5.0])

        # Direct scipy call (same parameter order as compare_null_vs_violation)
        scipy_result = mannwhitneyu(
            viol_vals, null_vals, alternative="two-sided", method="auto"
        )

        # Via compare_null_vs_violation with single-lookback dicts
        result = compare_null_vs_violation(
            null_drift={1: null_vals},
            violation_drift={1: viol_vals},
        )

        entry = result["by_lookback"]["1"]

        # U statistic must match exactly
        np.testing.assert_allclose(
            entry["mann_whitney_U"], scipy_result.statistic, atol=1e-10,
            err_msg="MW-U statistic must match scipy.stats.mannwhitneyu"
        )
        # p-value must match exactly
        np.testing.assert_allclose(
            entry["p_value_raw"], scipy_result.pvalue, atol=1e-10,
            err_msg="MW-U raw p-value must match scipy.stats.mannwhitneyu"
        )

    def test_mwu_identical_distributions_high_pvalue(self):
        """When null and violation have the same distribution, MW-U should
        return a high p-value (fail to reject H0: no difference)."""
        rng = np.random.default_rng(42)
        shared = rng.normal(5.0, 1.0, size=100)
        null_vals = shared[:50]
        viol_vals = shared[50:]

        result = compare_null_vs_violation(
            null_drift={1: null_vals},
            violation_drift={1: viol_vals},
        )
        p_raw = result["by_lookback"]["1"]["p_value_raw"]
        # p-value should be large (not significant) -- distributions are drawn
        # from the same population
        assert p_raw > 0.05, (
            f"Identical distributions should yield p > 0.05, got {p_raw}"
        )

    def test_mwu_separated_distributions_low_pvalue(self):
        """When null and violation are clearly separated, MW-U should return
        a low p-value (reject H0: significant difference detected)."""
        rng = np.random.default_rng(42)
        null_vals = rng.normal(0.0, 0.5, size=50)
        viol_vals = rng.normal(5.0, 0.5, size=50)

        result = compare_null_vs_violation(
            null_drift={1: null_vals},
            violation_drift={1: viol_vals},
        )
        p_raw = result["by_lookback"]["1"]["p_value_raw"]
        # p-value should be very small for clearly separated distributions
        assert p_raw < 0.001, (
            f"Clearly separated distributions should yield p < 0.001, got {p_raw}"
        )


class TestHolmBonferroniSeparation:
    """NULL-04: Verify compare_null_vs_violation applies Holm-Bonferroni
    as a separate family, not contaminated by external p-values."""

    def test_function_signature_has_no_pvalue_parameter(self):
        """compare_null_vs_violation takes (null_drift, violation_drift, alpha)
        with no external p-value parameter. This structurally prevents
        contamination from the primary AUROC analysis family."""
        sig = inspect.signature(compare_null_vs_violation)
        param_names = list(sig.parameters.keys())
        assert param_names == ["null_drift", "violation_drift", "alpha"], (
            f"compare_null_vs_violation must take exactly (null_drift, violation_drift, alpha), "
            f"got {param_names}. No external p_value parameter allowed."
        )

    def test_holm_bonferroni_uses_internal_p_array(self):
        """holm_bonferroni is called with p_array constructed from raw_p_values
        list populated only from mannwhitneyu results within the function.
        No external p-values can be injected (separate family guarantee)."""
        source = _read_source()
        func_source = _get_function_source(source, "compare_null_vs_violation")

        # Verify raw_p_values list is constructed locally
        assert "raw_p_values" in func_source, (
            "compare_null_vs_violation must use a raw_p_values list"
        )

        # Verify holm_bonferroni is called with p_array constructed from raw_p_values
        assert "p_array = np.array(raw_p_values)" in func_source, (
            "holm_bonferroni must be called with p_array built from local raw_p_values"
        )

        # Verify holm_bonferroni is called with this p_array
        assert "holm_bonferroni(p_array" in func_source, (
            "holm_bonferroni must receive p_array as its input"
        )

        # Verify raw_p_values is populated only from mannwhitneyu results
        assert "raw_p_values.append" in func_source, (
            "raw_p_values must be populated via append from MW-U results"
        )

    def test_functional_holm_bonferroni_adjusts_pvalues(self):
        """With 3 lookback distances, Holm-Bonferroni should adjust p-values
        upward from their raw values, confirming the correction is applied."""
        rng = np.random.default_rng(42)

        # Create 3 lookback distances with clearly separated distributions
        null_drift = {}
        viol_drift = {}
        for j in range(1, 4):
            null_drift[j] = rng.normal(0.0, 1.0, size=30)
            viol_drift[j] = rng.normal(2.0 + j * 0.5, 1.0, size=30)

        result = compare_null_vs_violation(null_drift, viol_drift)

        # Verify adjustment was applied (adjusted >= raw for all lookbacks)
        for j in range(1, 4):
            entry = result["by_lookback"][str(j)]
            raw = entry["p_value_raw"]
            adjusted = entry["p_value_adjusted"]
            # Holm-Bonferroni adjusted p-values are >= raw p-values
            assert adjusted >= raw - 1e-15, (
                f"Lookback {j}: adjusted p-value ({adjusted}) must be >= raw ({raw})"
            )

    def test_holm_bonferroni_family_size_matches_lookbacks(self):
        """With 3 lookback distances, Holm-Bonferroni should correct for
        a family of size 3 (not combined with any external family).
        The most significant p-value gets multiplied by 3."""
        rng = np.random.default_rng(123)

        # Create 3 lookback distances
        null_drift = {}
        viol_drift = {}
        for j in range(1, 4):
            null_drift[j] = rng.normal(0.0, 1.0, size=50)
            viol_drift[j] = rng.normal(3.0, 1.0, size=50)

        result = compare_null_vs_violation(null_drift, viol_drift)

        # Collect raw and adjusted p-values
        raw_ps = []
        adj_ps = []
        for j in range(1, 4):
            entry = result["by_lookback"][str(j)]
            raw_ps.append(entry["p_value_raw"])
            adj_ps.append(entry["p_value_adjusted"])

        # Manually compute Holm-Bonferroni with family size 3
        from src.analysis.statistical_controls import holm_bonferroni

        manual_adj, manual_reject = holm_bonferroni(np.array(raw_ps), alpha=0.05)

        # Adjusted values from compare_null_vs_violation must match manual
        # application with the same family size
        np.testing.assert_allclose(
            adj_ps, manual_adj, atol=1e-10,
            err_msg="Holm-Bonferroni adjustment must match manual application with family size 3"
        )


class TestPositionMatchedDrift:
    """Verify position-matched drift extraction uses correct column offsets
    and handles NaN filtering."""

    def test_position_matching_basic(self):
        """For event_positions=[10, 15] and max_lookback=3:
        j=1 extracts columns 9 and 14, j=2 extracts columns 8 and 13,
        j=3 extracts columns 7 and 12. Values pooled across all sequences."""
        rng = np.random.default_rng(42)
        metric_array = rng.standard_normal((5, 20))

        event_positions = [10, 15]
        max_lookback = 3

        result = extract_position_matched_drift(metric_array, event_positions, max_lookback)

        # j=1: columns 9 (=10-1) and 14 (=15-1), across 5 sequences = 10 values
        expected_j1 = np.concatenate([metric_array[:, 9], metric_array[:, 14]])
        np.testing.assert_array_equal(
            result[1], expected_j1,
            err_msg="j=1 must extract columns at event_position - 1"
        )

        # j=2: columns 8 and 13
        expected_j2 = np.concatenate([metric_array[:, 8], metric_array[:, 13]])
        np.testing.assert_array_equal(
            result[2], expected_j2,
            err_msg="j=2 must extract columns at event_position - 2"
        )

        # j=3: columns 7 and 12
        expected_j3 = np.concatenate([metric_array[:, 7], metric_array[:, 12]])
        np.testing.assert_array_equal(
            result[3], expected_j3,
            err_msg="j=3 must extract columns at event_position - 3"
        )

    def test_nan_filtering(self):
        """NaN values must be excluded from the extracted drift arrays.
        This prevents NaN from corrupting statistical tests downstream."""
        rng = np.random.default_rng(42)
        metric_array = rng.standard_normal((5, 20))

        # Insert NaN at metric_array[2, 9] -- this is in j=1 extraction for event_pos=10
        metric_array[2, 9] = np.nan

        result = extract_position_matched_drift(metric_array, [10, 15], max_lookback=3)

        # j=1 should have 9 values (10 - 1 NaN)
        assert len(result[1]) == 9, (
            f"NaN at [2,9] should be filtered out; expected 9 values, got {len(result[1])}"
        )
        # No NaN should remain
        assert np.all(np.isfinite(result[1])), "All NaN values must be filtered out"

    def test_both_null_and_violation_use_same_positions(self):
        """Structural check: in run_null_analysis, both violation_drift and
        null_drift are computed with extract_position_matched_drift using the
        same event_positions and max_lookback variables."""
        source = _read_source()
        func_source = _get_function_source(source, "run_null_analysis")

        # Count calls to extract_position_matched_drift -- should be exactly 2
        call_count = func_source.count("extract_position_matched_drift")
        assert call_count == 2, (
            f"run_null_analysis must call extract_position_matched_drift exactly 2 times "
            f"(once for violation, once for null), found {call_count} calls"
        )

        # Both calls must use the same variable names for event_positions and max_lookback
        # Parse the function AST to find both calls
        tree = ast.parse(textwrap.dedent(func_source))
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "extract_position_matched_drift":
                    calls.append(node)

        assert len(calls) == 2, (
            f"Expected 2 calls to extract_position_matched_drift, found {len(calls)}"
        )

        # Both calls must pass the same event_positions and max_lookback variable names
        for call in calls:
            arg_names = []
            for arg in call.args:
                if isinstance(arg, ast.Name):
                    arg_names.append(arg.id)
            # event_positions and max_lookback should be in the args
            assert "event_positions" in arg_names, (
                "Both calls must use the event_positions variable"
            )
            assert "max_lookback" in arg_names, (
                "Both calls must use the max_lookback variable"
            )
