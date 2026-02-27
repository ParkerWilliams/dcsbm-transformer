"""Comprehensive tests for the transformer model package.

Covers all three requirements:
  MODL-01: Configurable NanoGPT-scale transformer (d_model, n_layers, 1 head)
  MODL-02: SVD extraction (QK^T, attention weights, values, WvWo, residual stream)
  MODL-03: Vocabulary equals graph vertices (no special tokens)
"""

import torch
import pytest

from src.config.defaults import ANCHOR_CONFIG
from src.config.experiment import (
    ExperimentConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
)
from src.model import TransformerLM, ExtractionMode, ForwardOutput, create_model


# ---------------------------------------------------------------------------
# Helper: create a minimal valid config with custom overrides
# ---------------------------------------------------------------------------
def _make_config(
    n: int = 500,
    d_model: int = 128,
    n_layers: int = 4,
    w: int = 64,
    dropout: float = 0.0,
) -> ExperimentConfig:
    """Build an ExperimentConfig with custom model/graph/training params."""
    return ExperimentConfig(
        graph=GraphConfig(n=n),
        model=ModelConfig(d_model=d_model, n_layers=n_layers, dropout=dropout),
        training=TrainingConfig(
            w=w,
            walk_length=max(2 * w, 256),
            corpus_size=max(100 * n, 200_000),
            r=int(0.9 * w),
        ),
    )


# ===========================================================================
# MODL-01: NanoGPT-scale transformer with configurable d_model, n_layers
# ===========================================================================


class TestMODL01ConfigurableArchitecture:
    """MODL-01: Configurable d_model (64, 128, 256), n_layers (2, 4, 6), 1 head."""

    def test_anchor_config_creation(self):
        """create_model with ANCHOR_CONFIG produces correct architecture."""
        model = create_model(ANCHOR_CONFIG)
        assert model.d_model == 128
        assert model.n_layers == 4
        assert model.vocab_size == 500
        assert model.max_seq_len == 64

    @pytest.mark.parametrize("d_model", [64, 128, 256])
    def test_configurable_d_model(self, d_model: int):
        """Models with d_model=64, 128, 256 produce correct logit shapes."""
        config = _make_config(d_model=d_model)
        model = create_model(config)
        model.eval()

        x = torch.randint(0, config.graph.n, (2, config.training.w))
        with torch.no_grad():
            out = model(x)
        assert out.logits.shape == (2, config.training.w, config.graph.n)

    @pytest.mark.parametrize("n_layers", [2, 4, 6])
    def test_configurable_n_layers(self, n_layers: int):
        """Models with n_layers=2, 4, 6 have the correct number of blocks."""
        config = _make_config(n_layers=n_layers)
        model = create_model(config)
        assert len(model.blocks) == n_layers

    def test_n_heads_validation(self):
        """ExperimentConfig rejects invalid n_heads; accepts 1, 2, 4."""
        with pytest.raises(ValueError, match="n_heads must be 1, 2, or 4"):
            ExperimentConfig(model=ModelConfig(n_heads=3, d_model=384))

        # n_heads=2 with matching d_model is accepted
        config = ExperimentConfig(model=ModelConfig(n_heads=2, d_model=256))
        assert config.model.n_heads == 2

        # Verify single-head Q projection output is [B, T, D]
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randn(2, 16, 128)
        block = model.blocks[0]
        q = block.attention.W_q(x)
        assert q.shape == (2, 16, 128), f"Q should be [B, T, D], got {q.shape}"


# ===========================================================================
# MODL-02: SVD extraction (QK^T, attention weights, values, WvWo, residual)
# ===========================================================================


class TestMODL02ExtractionNone:
    """ExtractionMode.NONE returns logits only with no overhead."""

    def test_extraction_none_returns_logits_only(self):
        """NONE mode returns ForwardOutput with only logits populated."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (2, 64))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.NONE)
        assert isinstance(out, ForwardOutput)
        assert out.logits is not None
        assert out.qkt is None
        assert out.attention_weights is None
        assert out.values is None
        assert out.residual_stream is None
        assert out.residual_norms is None


class TestMODL02SVDTargets:
    """ExtractionMode.SVD_TARGETS returns QK^T, attention weights, values."""

    def test_extraction_svd_targets_shapes(self):
        """SVD_TARGETS returns correctly shaped tensors for all layers."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (2, 64))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.SVD_TARGETS)

        assert out.qkt.shape == (2, 4, 1, 64, 64)  # [B, n_layers, n_heads=1, T, T]
        assert out.attention_weights.shape == (2, 4, 1, 64, 64)
        assert out.values.shape == (2, 4, 1, 64, 128)  # [B, n_layers, n_heads=1, T, d_head]
        # SVD_TARGETS should NOT include residual stream
        assert out.residual_stream is None
        assert out.residual_norms is None

    def test_qkt_zero_filled_causal_mask(self):
        """Upper triangle of QK^T is all zeros (zero fill, not -inf)."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (1, 16))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.SVD_TARGETS)

        for layer_idx in range(4):
            qkt_layer = out.qkt[0, layer_idx, 0]  # [T, T] (head 0 for single-head)
            # Upper triangle (strictly above diagonal) should be zero
            upper = torch.triu(qkt_layer, diagonal=1)
            assert torch.all(upper == 0.0), (
                f"Layer {layer_idx}: upper triangle of QK^T should be zero-filled"
            )

    def test_qkt_lower_triangle_nonzero(self):
        """Lower triangle and diagonal of QK^T contain actual values."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (1, 16))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.SVD_TARGETS)

        # Check at least one layer has non-zero values in lower triangle
        any_nonzero = False
        for layer_idx in range(4):
            qkt_layer = out.qkt[0, layer_idx, 0]  # [T, T] (head 0 for single-head)
            lower = torch.tril(qkt_layer)
            if torch.any(lower != 0.0):
                any_nonzero = True
                break
        assert any_nonzero, "Lower triangle of QK^T should contain non-zero values"

    def test_attention_weights_sum_to_one(self):
        """Each row of attention weights sums to ~1.0."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (2, 32))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.SVD_TARGETS)

        # Sum across last dimension (key positions) should be ~1.0
        # Shape is [B, n_layers, n_heads, T, T]; sum across dim=-1 gives [B, n_layers, n_heads, T]
        row_sums = out.attention_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Attention weights should sum to 1, got range [{row_sums.min():.6f}, {row_sums.max():.6f}]"
        )

    def test_attention_weights_causal(self):
        """Attention weights have zeros above diagonal (causal mask enforced)."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (1, 16))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.SVD_TARGETS)

        for layer_idx in range(4):
            attn = out.attention_weights[0, layer_idx, 0]  # [T, T] (head 0 for single-head)
            upper = torch.triu(attn, diagonal=1)
            assert torch.all(upper == 0.0), (
                f"Layer {layer_idx}: attention weights should be zero above diagonal"
            )

    def test_extracted_tensors_detached(self):
        """QK^T, attention_weights, values all have requires_grad=False."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (2, 32))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.SVD_TARGETS)

        assert not out.qkt.requires_grad, "QK^T should be detached"
        assert not out.attention_weights.requires_grad, "Attention weights should be detached"
        assert not out.values.requires_grad, "Values should be detached"


class TestMODL02WvWo:
    """WvWo weight product extraction."""

    def test_get_wvwo_shape(self):
        """get_wvwo() returns [n_layers, n_heads, d_model, d_model] detached tensor."""
        model = create_model(ANCHOR_CONFIG)
        wvwo = model.get_wvwo()
        assert wvwo.shape == (4, 1, 128, 128)  # [n_layers, n_heads=1, d_model, d_model]
        assert not wvwo.requires_grad, "WvWo should be detached"

    def test_get_wvwo_input_agnostic(self):
        """get_wvwo() returns identical results on repeated calls (no input dependence)."""
        model = create_model(ANCHOR_CONFIG)
        wvwo1 = model.get_wvwo()
        wvwo2 = model.get_wvwo()
        assert torch.equal(wvwo1, wvwo2), "WvWo should be identical across calls"


class TestMODL02Residual:
    """Residual stream extraction modes."""

    def test_residual_mode_returns_stream(self):
        """RESIDUAL mode returns residual_stream and residual_norms."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (2, 32))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.RESIDUAL)

        # n_layers+1 residual states (includes pre-block embedding output)
        assert out.residual_stream.shape == (2, 32, 5, 128)  # [B, T, n_layers+1, D]
        assert out.residual_norms.shape == (2, 32, 5)  # [B, T, n_layers+1]
        # SVD targets should also be populated
        assert out.qkt is not None
        assert out.attention_weights is not None
        assert out.values is not None

    def test_full_mode_returns_everything(self):
        """FULL mode returns all fields populated."""
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (2, 32))
        with torch.no_grad():
            out = model(x, mode=ExtractionMode.FULL)

        assert out.logits is not None
        assert out.qkt is not None
        assert out.attention_weights is not None
        assert out.values is not None
        assert out.residual_stream is not None
        assert out.residual_norms is not None


# ===========================================================================
# MODL-03: Vocabulary equals graph vertices (no special tokens)
# ===========================================================================


class TestMODL03VocabularyMapping:
    """MODL-03: Tokens are vertex IDs, vocabulary = graph.n exactly."""

    @pytest.mark.parametrize("n", [100, 500])
    def test_vocab_equals_graph_n(self, n: int):
        """create_model with different graph.n values produces matching embedding/head sizes."""
        config = _make_config(n=n)
        model = create_model(config)
        assert model.token_embedding.num_embeddings == n
        assert model.lm_head.out_features == n
        assert model.vocab_size == n

    def test_no_special_tokens(self):
        """vocab_size equals config.graph.n exactly (no +1 for BOS/PAD/EOS)."""
        config = _make_config(n=300)
        model = create_model(config)
        assert model.vocab_size == 300  # Not 301 or 302

    def test_valid_token_range(self):
        """Forward pass with token IDs 0 to n-1 succeeds."""
        config = _make_config(n=100)
        model = create_model(config)
        model.eval()

        # Use all valid token IDs
        x = torch.arange(100).unsqueeze(0)[:, :config.training.w]  # [1, w]
        with torch.no_grad():
            out = model(x)
        assert out.logits.shape[2] == 100


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    """Integration tests: initialization, determinism, parameter count."""

    def test_weight_initialization(self):
        """Embedding std is ~0.02; W_o has smaller std (residual scaling)."""
        model = create_model(ANCHOR_CONFIG)

        # Embedding should be ~Normal(0, 0.02)
        emb_std = model.token_embedding.weight.std().item()
        assert 0.01 < emb_std < 0.03, f"Embedding std {emb_std} not near 0.02"

        # W_o should have smaller std due to residual scaling: 0.02 / sqrt(2 * 4) ~ 0.00707
        wo_std = model.blocks[0].attention.W_o.weight.std().item()
        assert wo_std < emb_std, (
            f"W_o std ({wo_std}) should be smaller than embedding std ({emb_std})"
        )

    def test_no_weight_tying(self):
        """token_embedding.weight is NOT the same tensor as lm_head.weight."""
        model = create_model(ANCHOR_CONFIG)
        assert model.token_embedding.weight is not model.lm_head.weight, (
            "Embedding and lm_head weights should be separate (no weight tying)"
        )

    def test_deterministic_output(self):
        """With same seed, two forward passes produce identical logits."""
        torch.manual_seed(42)
        model = create_model(ANCHOR_CONFIG)
        model.eval()
        x = torch.randint(0, 500, (2, 64))
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.equal(out1.logits, out2.logits), "Same input should give same output"

    def test_parameter_count(self):
        """Anchor config parameter count is reasonable (sanity check)."""
        model = create_model(ANCHOR_CONFIG)
        total_params = sum(p.numel() for p in model.parameters())
        # d_model=128, n_layers=4, vocab=500
        # Rough estimate: embeddings ~128k, 4 blocks ~1M each, lm_head ~64k
        # Should be between 500K and 10M parameters
        assert 500_000 < total_params < 10_000_000, (
            f"Parameter count {total_params} outside expected range [500K, 10M]"
        )
