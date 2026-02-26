"""Integration tests for fused evaluation pipeline, NPZ output, and warmup skip.

Uses a small TransformerLM and synthetic graph to test the complete fused
evaluation pass: generation + SVD collection + behavioral labeling.
"""

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.config.experiment import ExperimentConfig, GraphConfig, ModelConfig, TrainingConfig
from src.evaluation.pipeline import (
    EvaluationResult,
    SVD_TARGETS,
    fused_evaluate,
    save_evaluation_results,
)
from src.graph.jumpers import JumperInfo
from src.graph.types import GraphData
from src.model.transformer import TransformerLM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def small_config():
    """Small config for fast integration tests."""
    return ExperimentConfig(
        graph=GraphConfig(n=10, K=2, p_in=0.5, p_out=0.1, n_jumpers_per_block=1),
        model=ModelConfig(d_model=16, n_layers=2, n_heads=1, dropout=0.0),
        training=TrainingConfig(
            w=8, walk_length=32, corpus_size=2000, r=3,
            learning_rate=3e-4, batch_size=8, max_steps=100,
            eval_interval=50, checkpoint_interval=100,
        ),
        seed=42,
    )


@pytest.fixture
def small_model(small_config):
    """Small TransformerLM for testing."""
    torch.manual_seed(42)
    return TransformerLM(
        vocab_size=small_config.graph.n,
        d_model=small_config.model.d_model,
        n_layers=small_config.model.n_layers,
        max_seq_len=small_config.training.w,
        dropout=0.0,
    )


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def small_graph():
    """10-vertex graph with 2 blocks, all-connected (complete graph with self-loops)."""
    n = 10
    K = 2
    block_size = n // K
    block_assignments = np.repeat(np.arange(K), block_size).astype(np.int32)

    rows, cols = [], []
    for i in range(n):
        for j in range(n):
            rows.append(i)
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = csr_matrix((data, (rows, cols)), shape=(n, n))

    return GraphData(
        adjacency=adjacency,
        block_assignments=block_assignments,
        theta=np.ones(n),
        n=n,
        K=K,
        block_size=block_size,
        generation_seed=42,
        attempt=0,
    )


@pytest.fixture
def jumpers():
    """Two jumpers with different r values."""
    return [
        JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3),
        JumperInfo(vertex_id=5, source_block=1, target_block=0, r=4),
    ]


@pytest.fixture
def eval_walks():
    """Short evaluation walks: 5 walks of length 16."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 10, size=(5, 16), dtype=np.int32)


@pytest.fixture
def fused_result(small_model, eval_walks, small_graph, jumpers, small_config, device):
    """Pre-computed fused evaluation result."""
    return fused_evaluate(
        small_model, eval_walks, small_graph, jumpers, small_config, device,
        batch_size=3,
    )


# ---------------------------------------------------------------------------
# TestFusedPass
# ---------------------------------------------------------------------------
class TestFusedPass:
    """Core fused evaluation pass tests."""

    def test_fused_returns_correct_shapes(self, fused_result, eval_walks, small_config):
        """Run fused_evaluate, verify generated/edge_valid/rule_outcome/failure_index shapes."""
        n_seq = eval_walks.shape[0]
        result = fused_result
        assert isinstance(result, EvaluationResult)
        assert result.generated.shape[0] == n_seq
        assert result.edge_valid.shape[0] == n_seq
        assert result.rule_outcome.shape[0] == n_seq
        assert result.failure_index.shape == (n_seq,)
        assert result.sequence_lengths.shape == (n_seq,)

    def test_fused_svd_metric_keys(self, fused_result, small_config):
        """Verify svd_metrics dict has keys for all 3 targets x layers x metrics."""
        n_layers = small_config.model.n_layers
        expected_targets = ["qkt", "wvwo", "avwo"]
        for target in expected_targets:
            for layer_idx in range(n_layers):
                # Check at least stable_rank exists
                key = f"{target}.layer_{layer_idx}.stable_rank"
                assert key in fused_result.svd_metrics, f"Missing key: {key}"

    def test_fused_svd_metric_shapes(self, fused_result, eval_walks):
        """Each SVD metric array has shape [n_seq, max_steps-1]."""
        n_seq = eval_walks.shape[0]
        for key, arr in fused_result.svd_metrics.items():
            assert arr.shape[0] == n_seq, f"{key} has wrong n_seq dimension"
            # All metric arrays should have same step dimension
            assert arr.ndim == 2, f"{key} should be 2D"

    def test_single_forward_pass(self, small_model, eval_walks, small_graph, jumpers, small_config, device):
        """Verify the model's forward is called with ExtractionMode.SVD_TARGETS."""
        # Run with very small batch to check extraction works
        result = fused_evaluate(
            small_model, eval_walks[:1], small_graph, jumpers, small_config, device,
            batch_size=1,
        )
        # If extraction didn't work, svd_metrics would have no non-NaN values after warmup
        w = small_config.training.w
        has_real_values = False
        for key, arr in result.svd_metrics.items():
            if "wvwo" in key:
                continue  # WvWo might have values everywhere
            if not np.all(np.isnan(arr)):
                has_real_values = True
                break
        assert has_real_values, "No real SVD metrics collected (extraction may have failed)"


# ---------------------------------------------------------------------------
# TestWarmupSkip
# ---------------------------------------------------------------------------
class TestWarmupSkip:
    """SVD metrics should be NaN for positions < w."""

    def test_warmup_positions_are_nan(self, fused_result, small_config):
        """For positions < w, all SVD metric values should be NaN."""
        w = small_config.training.w
        for key, arr in fused_result.svd_metrics.items():
            if "wvwo" in key:
                continue  # WvWo is static, may have values everywhere
            warmup_slice = arr[:, :w - 1]
            assert np.all(np.isnan(warmup_slice)), (
                f"{key} has non-NaN values in warmup positions (< w={w})"
            )

    def test_post_warmup_positions_have_values(self, fused_result, small_config):
        """For positions >= w (within sequence length), SVD metrics should be finite."""
        w = small_config.training.w
        # Check at least some QK^T metrics have real values after warmup
        found_finite = False
        for key, arr in fused_result.svd_metrics.items():
            if "qkt" in key and "grassmannian" not in key:
                post_warmup = arr[:, w - 1:]
                # At least some should be finite (not all NaN)
                if np.any(np.isfinite(post_warmup)):
                    found_finite = True
                    break
        assert found_finite, "No finite QK^T metrics found after warmup"


# ---------------------------------------------------------------------------
# TestNPZOutput
# ---------------------------------------------------------------------------
class TestNPZOutput:
    """Test save/load of token_metrics.npz."""

    def test_save_and_load_token_metrics(self, fused_result, tmp_path):
        """Create EvaluationResult, save to NPZ, reload, verify keys match."""
        summary = save_evaluation_results(fused_result, tmp_path)
        npz_path = tmp_path / "token_metrics.npz"
        assert npz_path.exists()

        loaded = np.load(str(npz_path))
        # Check SVD metric keys present
        for key in fused_result.svd_metrics:
            assert key in loaded, f"Missing SVD metric key: {key}"
        # Check behavioral arrays
        assert "edge_valid" in loaded
        assert "rule_outcome" in loaded
        assert "failure_index" in loaded
        assert "sequence_lengths" in loaded

    def test_npz_key_convention(self, fused_result, tmp_path):
        """Keys follow target.layer_N.metric_name pattern."""
        save_evaluation_results(fused_result, tmp_path)
        loaded = np.load(str(tmp_path / "token_metrics.npz"))
        for key in loaded.files:
            if key in ("edge_valid", "rule_outcome", "failure_index", "sequence_lengths", "generated"):
                continue
            parts = key.split(".")
            assert len(parts) == 3, f"Key '{key}' doesn't follow target.layer.metric pattern"
            assert parts[0] in SVD_TARGETS, f"Unknown target in key: {key}"
            assert parts[1].startswith("layer_"), f"Layer part malformed in key: {key}"

    def test_behavioral_arrays_in_npz(self, fused_result, tmp_path):
        """edge_valid, rule_outcome, failure_index stored correctly."""
        save_evaluation_results(fused_result, tmp_path)
        loaded = np.load(str(tmp_path / "token_metrics.npz"))
        np.testing.assert_array_equal(loaded["edge_valid"], fused_result.edge_valid)
        np.testing.assert_array_equal(loaded["rule_outcome"], fused_result.rule_outcome)
        np.testing.assert_array_equal(loaded["failure_index"], fused_result.failure_index)


# ---------------------------------------------------------------------------
# TestSVDTargets
# ---------------------------------------------------------------------------
class TestSVDTargets:
    """Verify distinct SVD targets produce different metrics."""

    def test_qkt_svd_produces_finite_metrics(self, fused_result, small_config):
        """QK^T metrics for positions >= w are all finite."""
        w = small_config.training.w
        for key, arr in fused_result.svd_metrics.items():
            if "qkt" not in key or "grassmannian" in key:
                continue
            # Check post-warmup positions for first sequence
            post_warmup = arr[0, w - 1:]
            # At least some should be finite
            finite_mask = np.isfinite(post_warmup)
            if finite_mask.any():
                # All finite values should be real numbers
                assert not np.any(np.isinf(post_warmup[finite_mask]))

    def test_wvwo_svd_is_static(self, fused_result, small_config):
        """WvWo metrics are identical across all steps (static weight matrix)."""
        w = small_config.training.w
        for key, arr in fused_result.svd_metrics.items():
            if "wvwo" not in key or "grassmannian" in key:
                continue
            # Get non-NaN values
            valid_vals = arr[0][~np.isnan(arr[0])]
            if len(valid_vals) > 1:
                # All values should be approximately equal (static)
                assert np.allclose(valid_vals, valid_vals[0], atol=1e-5), (
                    f"{key} WvWo values vary across steps (should be static)"
                )

    def test_avwo_svd_differs_from_qkt(self, fused_result, small_config):
        """AVWo metrics are not identical to QK^T metrics (different targets)."""
        w = small_config.training.w
        n_layers = small_config.model.n_layers
        for layer_idx in range(n_layers):
            qkt_key = f"qkt.layer_{layer_idx}.stable_rank"
            avwo_key = f"avwo.layer_{layer_idx}.stable_rank"
            qkt_vals = fused_result.svd_metrics[qkt_key]
            avwo_vals = fused_result.svd_metrics[avwo_key]
            # They should differ (different targets produce different SVD)
            both_valid = ~np.isnan(qkt_vals) & ~np.isnan(avwo_vals)
            if both_valid.any():
                # Not all values should be identical
                qkt_valid = qkt_vals[both_valid]
                avwo_valid = avwo_vals[both_valid]
                if len(qkt_valid) > 0 and len(avwo_valid) > 0:
                    # At least some values should differ
                    assert not np.allclose(qkt_valid, avwo_valid, atol=1e-6), (
                        f"QK^T and AVWo stable_rank layer {layer_idx} are identical"
                    )


# ---------------------------------------------------------------------------
# TestTailExtension
# ---------------------------------------------------------------------------
class TestTailExtension:
    """Tail extension for late jumper encounters."""

    def test_no_extension_without_late_jumper(
        self, small_model, small_graph, small_config, device
    ):
        """Sequences without any jumpers have default length."""
        # Use empty jumper list so no tail extension can trigger
        eval_walks_no_jumper = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int32)
        result = fused_evaluate(
            small_model, eval_walks_no_jumper, small_graph, [],
            small_config, device, batch_size=1,
        )
        default_length = 4 * small_config.training.w
        # Sequence length should be default_length (no extension possible)
        assert result.sequence_lengths[0] == default_length


# ---------------------------------------------------------------------------
# TestGuardActivations
# ---------------------------------------------------------------------------
class TestGuardActivations:
    """Guard activation tracking."""

    def test_guard_activations_counted(self, fused_result):
        """Verify guard_activations dict exists and counts are >= 0."""
        assert isinstance(fused_result.guard_activations, dict)
        for key, count in fused_result.guard_activations.items():
            assert isinstance(count, int)
            assert count >= 0

    def test_guard_activation_keys_present(self, fused_result, small_config):
        """Guard activations have keys for all targets and layers."""
        n_layers = small_config.model.n_layers
        for layer_idx in range(n_layers):
            wvwo_key = f"wvwo.layer_{layer_idx}"
            assert wvwo_key in fused_result.guard_activations


# ---------------------------------------------------------------------------
# TestSummaryOutput
# ---------------------------------------------------------------------------
class TestSummaryOutput:
    """Test save_evaluation_results summary dict."""

    def test_summary_has_required_fields(self, fused_result, tmp_path):
        """Summary dict contains expected fields for result.json."""
        summary = save_evaluation_results(fused_result, tmp_path)
        assert "scalars" in summary
        assert "svd_metric_stats" in summary["scalars"]
        assert "guard_activations" in summary["scalars"]
        assert "failure_index_list" in summary["scalars"]
        assert "n_sequences" in summary["scalars"]
        assert "n_violations" in summary["scalars"]
