"""Tests for multi-head attention support (Phase 16: MHAD-01).

Validates that the transformer correctly supports n_heads = 1, 2, 4 with
per-head QK^T extraction, proper scaling, and backward compatibility.
"""

import pytest
import torch

from src.config.experiment import ExperimentConfig, ModelConfig
from src.model.attention import CausalSelfAttention
from src.model.block import TransformerBlock
from src.model.transformer import TransformerLM, create_model
from src.model.types import ExtractionMode


# -- Config validation tests --


class TestConfigValidation:
    """Config accepts n_heads 1, 2, 4 and rejects others."""

    def test_n_heads_1_accepted(self):
        config = ExperimentConfig(model=ModelConfig(n_heads=1, d_model=128))
        assert config.model.n_heads == 1

    def test_n_heads_2_accepted(self):
        config = ExperimentConfig(model=ModelConfig(n_heads=2, d_model=256))
        assert config.model.n_heads == 2

    def test_n_heads_4_accepted(self):
        config = ExperimentConfig(model=ModelConfig(n_heads=4, d_model=512))
        assert config.model.n_heads == 4

    def test_n_heads_3_rejected(self):
        with pytest.raises(ValueError, match="n_heads must be 1, 2, or 4"):
            ExperimentConfig(model=ModelConfig(n_heads=3, d_model=384))

    def test_n_heads_8_rejected(self):
        with pytest.raises(ValueError, match="n_heads must be 1, 2, or 4"):
            ExperimentConfig(model=ModelConfig(n_heads=8, d_model=1024))

    def test_n_heads_0_rejected(self):
        with pytest.raises(ValueError, match="n_heads must be 1, 2, or 4"):
            ExperimentConfig(model=ModelConfig(n_heads=0, d_model=128))

    def test_d_model_not_divisible_rejected(self):
        with pytest.raises(ValueError, match="divisible"):
            ExperimentConfig(model=ModelConfig(n_heads=2, d_model=129))


# -- CausalSelfAttention tests --


class TestCausalSelfAttentionMultiHead:
    """CausalSelfAttention with n_heads > 1 produces correct shapes and behavior."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.mark.parametrize("n_heads,d_model", [(1, 128), (2, 256), (4, 512)])
    def test_output_shape(self, n_heads, d_model, device):
        B, T = 2, 16
        attn = CausalSelfAttention(d_model, n_heads, max_seq_len=T).to(device)
        x = torch.randn(B, T, d_model, device=device)
        y, _ = attn(x, extract=False)
        assert y.shape == (B, T, d_model)

    @pytest.mark.parametrize("n_heads,d_model", [(1, 128), (2, 256), (4, 512)])
    def test_extraction_shapes(self, n_heads, d_model, device):
        B, T = 2, 16
        d_head = d_model // n_heads
        attn = CausalSelfAttention(d_model, n_heads, max_seq_len=T).to(device)
        x = torch.randn(B, T, d_model, device=device)
        y, internals = attn(x, extract=True)
        assert internals is not None
        assert internals.qkt.shape == (B, n_heads, T, T)
        assert internals.attention_weights.shape == (B, n_heads, T, T)
        assert internals.values.shape == (B, n_heads, T, d_head)

    def test_causal_mask_per_head(self, device):
        """Each head respects causal mask independently."""
        n_heads, d_model, T = 2, 256, 8
        attn = CausalSelfAttention(d_model, n_heads, max_seq_len=T).to(device)
        x = torch.randn(1, T, d_model, device=device)
        _, internals = attn(x, extract=True)
        # Upper triangle of QK^T should be zero (zero-filled for SVD target)
        for h in range(n_heads):
            qkt_h = internals.qkt[0, h]  # [T, T]
            upper = torch.triu(qkt_h, diagonal=1)
            assert torch.all(upper == 0), f"Head {h} has non-zero upper triangle"

    def test_attention_weights_sum_to_one(self, device):
        """Attention weights sum to 1 along last dim per head."""
        n_heads, d_model, T = 4, 512, 8
        attn = CausalSelfAttention(d_model, n_heads, max_seq_len=T).to(device)
        x = torch.randn(1, T, d_model, device=device)
        _, internals = attn(x, extract=True)
        for h in range(n_heads):
            row_sums = internals.attention_weights[0, h].sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_single_head_d_head_equals_d_model(self, device):
        """Single-head uses d_head = d_model (backward compat scale factor)."""
        d_model = 128
        attn = CausalSelfAttention(d_model, n_heads=1, max_seq_len=8).to(device)
        assert attn.d_head == d_model

    def test_multi_head_d_head_constant(self, device):
        """Multi-head configs keep d_k = 128 when d_model scales."""
        # 2 heads with d_model=256: d_head = 128
        attn2 = CausalSelfAttention(256, n_heads=2, max_seq_len=8).to(device)
        assert attn2.d_head == 128
        # 4 heads with d_model=512: d_head = 128
        attn4 = CausalSelfAttention(512, n_heads=4, max_seq_len=8).to(device)
        assert attn4.d_head == 128

    def test_heads_produce_different_qkt(self, device):
        """Different heads should generally produce different QK^T matrices."""
        n_heads, d_model, T = 2, 256, 16
        attn = CausalSelfAttention(d_model, n_heads, max_seq_len=T).to(device)
        x = torch.randn(1, T, d_model, device=device)
        _, internals = attn(x, extract=True)
        qkt_0 = internals.qkt[0, 0]
        qkt_1 = internals.qkt[0, 1]
        # After random init, heads should differ
        assert not torch.allclose(qkt_0, qkt_1, atol=1e-3)

    def test_internals_are_detached(self, device):
        """Extracted internals should be detached from computation graph."""
        attn = CausalSelfAttention(128, 1, max_seq_len=8).to(device)
        x = torch.randn(1, 8, 128, device=device, requires_grad=True)
        _, internals = attn(x, extract=True)
        assert not internals.qkt.requires_grad
        assert not internals.attention_weights.requires_grad
        assert not internals.values.requires_grad

    def test_no_extraction_returns_none(self, device):
        """extract=False returns None internals."""
        attn = CausalSelfAttention(128, 1, max_seq_len=8).to(device)
        x = torch.randn(1, 8, 128, device=device)
        y, internals = attn(x, extract=False)
        assert internals is None
        assert y.shape == (1, 8, 128)


# -- TransformerBlock tests --


class TestTransformerBlockMultiHead:
    """TransformerBlock passes n_heads to attention."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.mark.parametrize("n_heads,d_model", [(1, 128), (2, 256)])
    def test_block_output_shape(self, n_heads, d_model, device):
        B, T = 2, 16
        block = TransformerBlock(d_model, n_heads, max_seq_len=T).to(device)
        x = torch.randn(B, T, d_model, device=device)
        out, internals = block(x, extract=True)
        assert out.shape == (B, T, d_model)
        assert internals is not None
        assert internals.qkt.shape == (B, n_heads, T, T)


# -- TransformerLM tests --


class TestTransformerLMMultiHead:
    """TransformerLM with multi-head support."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.mark.parametrize("n_heads,d_model", [(1, 128), (2, 256), (4, 512)])
    def test_forward_output_shapes(self, n_heads, d_model, device):
        B, T, n_layers, vocab = 2, 16, 2, 100
        d_head = d_model // n_heads
        model = TransformerLM(vocab, d_model, n_layers, n_heads, T).to(device)
        idx = torch.randint(0, vocab, (B, T), device=device)
        out = model(idx, mode=ExtractionMode.SVD_TARGETS)
        assert out.logits.shape == (B, T, vocab)
        assert out.qkt.shape == (B, n_layers, n_heads, T, T)
        assert out.attention_weights.shape == (B, n_layers, n_heads, T, T)
        assert out.values.shape == (B, n_layers, n_heads, T, d_head)

    @pytest.mark.parametrize("n_heads,d_model", [(1, 128), (2, 256), (4, 512)])
    def test_get_wvwo_shape(self, n_heads, d_model, device):
        n_layers = 2
        model = TransformerLM(100, d_model, n_layers, n_heads, 16).to(device)
        wvwo = model.get_wvwo()
        assert wvwo.shape == (n_layers, n_heads, d_model, d_model)

    def test_create_model_passes_n_heads(self):
        config = ExperimentConfig(
            model=ModelConfig(n_heads=2, d_model=256),
        )
        model = create_model(config)
        assert model.n_heads == 2
        assert model.d_head == 128

    def test_create_model_default_single_head(self):
        config = ExperimentConfig()
        model = create_model(config)
        assert model.n_heads == 1
        assert model.d_head == 128

    def test_no_extraction_mode_skips_internals(self, device):
        model = TransformerLM(100, 256, 2, 2, 16).to(device)
        idx = torch.randint(0, 100, (1, 16), device=device)
        out = model(idx, mode=ExtractionMode.NONE)
        assert out.qkt is None
        assert out.attention_weights is None
        assert out.values is None

    def test_residual_mode_works_with_multihead(self, device):
        B, T, n_layers, d_model = 1, 8, 2, 256
        model = TransformerLM(100, d_model, n_layers, 2, T).to(device)
        idx = torch.randint(0, 100, (B, T), device=device)
        out = model(idx, mode=ExtractionMode.RESIDUAL)
        assert out.residual_stream.shape == (B, T, n_layers + 1, d_model)
        assert out.residual_norms.shape == (B, T, n_layers + 1)
        # Also check SVD targets are populated in RESIDUAL mode
        assert out.qkt is not None

    def test_n_heads_stored_on_model(self, device):
        model = TransformerLM(100, 256, 2, 2, 16).to(device)
        assert model.n_heads == 2
        assert model.d_head == 128

    def test_get_wvwo_detached(self, device):
        model = TransformerLM(100, 128, 2, 1, 16).to(device)
        wvwo = model.get_wvwo()
        assert not wvwo.requires_grad


# -- Backward compatibility tests --


class TestBackwardCompatibility:
    """Single-head behavior is correct and backward compatible."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_single_head_wvwo_includes_head_dim(self, device):
        """Single-head get_wvwo has n_heads=1 dimension."""
        model = TransformerLM(100, 128, 2, 1, 16).to(device)
        wvwo = model.get_wvwo()
        # Shape is [n_layers, 1, d_model, d_model]
        assert wvwo.shape == (2, 1, 128, 128)

    def test_single_head_forward_includes_head_dim(self, device):
        """Single-head forward output includes n_heads=1 dimension."""
        model = TransformerLM(100, 128, 2, 1, 16).to(device)
        idx = torch.randint(0, 100, (1, 16), device=device)
        out = model(idx, mode=ExtractionMode.SVD_TARGETS)
        assert out.qkt.shape == (1, 2, 1, 16, 16)  # [B, n_layers, n_heads=1, T, T]

    def test_single_head_logits_reasonable(self, device):
        """Single-head model produces reasonable logits (not NaN/Inf)."""
        model = TransformerLM(100, 128, 2, 1, 16).to(device)
        idx = torch.randint(0, 100, (2, 16), device=device)
        out = model(idx, mode=ExtractionMode.NONE)
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()

    def test_multi_head_logits_reasonable(self, device):
        """Multi-head model produces reasonable logits."""
        model = TransformerLM(100, 256, 2, 2, 16).to(device)
        idx = torch.randint(0, 100, (2, 16), device=device)
        out = model(idx, mode=ExtractionMode.NONE)
        assert not torch.isnan(out.logits).any()
        assert not torch.isinf(out.logits).any()

    def test_gradient_flows_through_multihead(self, device):
        """Gradients flow through multi-head attention."""
        model = TransformerLM(100, 256, 2, 2, 16).to(device)
        idx = torch.randint(0, 100, (2, 16), device=device)
        out = model(idx, mode=ExtractionMode.NONE)
        loss = out.logits.sum()
        loss.backward()
        # Check that W_q has gradients
        for block in model.blocks:
            assert block.attention.W_q.weight.grad is not None
            assert block.attention.W_o.weight.grad is not None
