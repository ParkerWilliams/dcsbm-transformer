"""Audit tests for QK^T construction and dual mask behavior (SVD-01).

Verifies the mathematical correctness of CausalSelfAttention's QK^T
computation, causal masking (zero-fill for SVD target vs -inf for
softmax path), multi-head per-head projections, and scale factor
1/sqrt(d_head).
"""

import math

import torch
import torch.nn.functional as F

from src.model.attention import CausalSelfAttention


class TestQKTFormulaCorrectness:
    """Verify QK^T = (x @ Wq.T) @ (x @ Wk.T).T / sqrt(d_head) for single-head."""

    def test_qkt_formula_single_head(self) -> None:
        """QK^T must equal the manually computed scaled dot product.
        For nn.Linear(bias=False), output = input @ weight.T, so:
          Q = x @ W_q.weight.T
          K = x @ W_k.weight.T
          QK^T = Q @ K.T / sqrt(d_head)
        With n_heads=1, d_head=d_model=8 and scale = 1/sqrt(8).
        """
        torch.manual_seed(42)
        d_model = 8
        T = 4

        attn = CausalSelfAttention(d_model=d_model, n_heads=1, max_seq_len=T, dropout=0.0)

        # Set known weights: use simple deterministic values
        Wq = torch.eye(d_model) * 0.5
        Wk = torch.eye(d_model) * 0.3
        with torch.no_grad():
            attn.W_q.weight.copy_(Wq)
            attn.W_k.weight.copy_(Wk)

        # Known input
        x = torch.randn(1, T, d_model)

        # Manual computation: Q = x @ Wq.T, K = x @ Wk.T
        Q_manual = x @ Wq.T  # [1, 4, 8]
        K_manual = x @ Wk.T  # [1, 4, 8]
        scale = 1.0 / math.sqrt(d_model)  # d_head = d_model for n_heads=1
        qkt_manual = (Q_manual @ K_manual.transpose(-2, -1)) * scale  # [1, 4, 4]

        # Extract from model
        _, internals = attn(x, extract=True)
        qkt_extracted = internals.qkt  # [1, 1, 4, 4] (B, n_heads, T, T)

        # The extracted qkt has zero-fill mask applied; compare only causal region
        # (lower triangular including diagonal)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        torch.testing.assert_close(
            qkt_extracted[0, 0][mask],
            qkt_manual[0][mask],
            atol=1e-5,
            rtol=1e-5,
        )

    def test_qkt_shape_single_head(self) -> None:
        """QK^T shape must be [B=1, n_heads=1, T=4, T=4]."""
        d_model = 8
        T = 4
        attn = CausalSelfAttention(d_model=d_model, n_heads=1, max_seq_len=T, dropout=0.0)
        x = torch.randn(1, T, d_model)

        _, internals = attn(x, extract=True)
        assert internals.qkt.shape == (1, 1, T, T), (
            f"Expected shape (1, 1, {T}, {T}), got {internals.qkt.shape}"
        )


class TestQKTDualMask:
    """Verify SVD target gets zero-fill mask, softmax path gets -inf mask."""

    def test_svd_target_zero_fill(self) -> None:
        """For the SVD target (qkt), future positions (j > i) must be exactly 0.0.
        Zero-filling preserves clean input for singular value decomposition.
        """
        torch.manual_seed(42)
        d_model = 8
        T = 4
        attn = CausalSelfAttention(d_model=d_model, n_heads=1, max_seq_len=T, dropout=0.0)
        x = torch.randn(1, T, d_model)

        _, internals = attn(x, extract=True)
        qkt = internals.qkt[0, 0]  # [T, T]

        # Upper triangular (strictly) must be zero
        for i in range(T):
            for j in range(i + 1, T):
                assert qkt[i, j].item() == 0.0, (
                    f"qkt[{i},{j}] = {qkt[i,j].item()}, expected 0.0 for SVD target"
                )

    def test_attention_weights_softmax_of_neg_inf(self) -> None:
        """Attention weights for future positions (j > i) must be 0.0.
        softmax(-inf) = 0, so the -inf mask produces zero attention to future.
        """
        torch.manual_seed(42)
        d_model = 8
        T = 4
        attn = CausalSelfAttention(d_model=d_model, n_heads=1, max_seq_len=T, dropout=0.0)
        x = torch.randn(1, T, d_model)

        _, internals = attn(x, extract=True)
        att_weights = internals.attention_weights[0, 0]  # [T, T]

        # Future positions must have zero attention weight
        for i in range(T):
            for j in range(i + 1, T):
                assert att_weights[i, j].item() == 0.0, (
                    f"att_weights[{i},{j}] = {att_weights[i,j].item()}, expected 0.0"
                )

    def test_attention_weights_sum_to_one(self) -> None:
        """Attention weights must sum to 1.0 along the last dim (valid probability).
        This verifies softmax is applied correctly on the -inf masked scores.
        """
        torch.manual_seed(42)
        d_model = 8
        T = 4
        attn = CausalSelfAttention(d_model=d_model, n_heads=1, max_seq_len=T, dropout=0.0)
        x = torch.randn(1, T, d_model)

        _, internals = attn(x, extract=True)
        att_weights = internals.attention_weights[0, 0]  # [T, T]

        # Each row must sum to 1.0 (probability distribution)
        row_sums = att_weights.sum(dim=-1)
        torch.testing.assert_close(
            row_sums,
            torch.ones(T),
            atol=1e-5,
            rtol=1e-5,
        )


class TestQKTMultiHead:
    """Verify per-head QK^T computation for n_heads=4."""

    def test_qkt_multi_head_formula(self) -> None:
        """For multi-head attention with d_model=16, n_heads=4:
        d_head = 4, scale = 1/sqrt(4) = 0.5.
        For each head h, Q_h = x @ Wq[h*4:(h+1)*4, :].T and K_h similarly.
        QK^T_h = Q_h @ K_h.T * scale must match extracted per-head qkt.
        """
        torch.manual_seed(42)
        d_model = 16
        n_heads = 4
        d_head = d_model // n_heads  # 4
        T = 4

        attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=T, dropout=0.0)

        # Set known weights
        Wq = torch.randn(d_model, d_model) * 0.1
        Wk = torch.randn(d_model, d_model) * 0.1
        with torch.no_grad():
            attn.W_q.weight.copy_(Wq)
            attn.W_k.weight.copy_(Wk)

        x = torch.randn(1, T, d_model)

        # Manually compute per-head QK^T
        # Q_full = x @ Wq.T -> [1, T, d_model], then reshape to [1, n_heads, T, d_head]
        Q_full = x @ Wq.T
        K_full = x @ Wk.T
        Q_heads = Q_full.view(1, T, n_heads, d_head).transpose(1, 2)  # [1, 4, T, 4]
        K_heads = K_full.view(1, T, n_heads, d_head).transpose(1, 2)  # [1, 4, T, 4]
        scale = 1.0 / math.sqrt(d_head)
        qkt_manual = (Q_heads @ K_heads.transpose(-2, -1)) * scale  # [1, 4, T, T]

        # Extract from model
        _, internals = attn(x, extract=True)
        qkt_extracted = internals.qkt  # [1, 4, T, T]

        # Compare causal region for each head
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        for h in range(n_heads):
            torch.testing.assert_close(
                qkt_extracted[0, h][mask],
                qkt_manual[0, h][mask],
                atol=1e-5,
                rtol=1e-5,
            )

    def test_qkt_multi_head_shape(self) -> None:
        """QK^T shape must be [B=1, n_heads=4, T=4, T=4] for 4-head config."""
        d_model = 16
        n_heads = 4
        T = 4
        attn = CausalSelfAttention(d_model=d_model, n_heads=n_heads, max_seq_len=T, dropout=0.0)
        x = torch.randn(1, T, d_model)

        _, internals = attn(x, extract=True)
        assert internals.qkt.shape == (1, n_heads, T, T), (
            f"Expected shape (1, {n_heads}, {T}, {T}), got {internals.qkt.shape}"
        )


class TestQKTScaleFactor:
    """Verify scale factor is 1/sqrt(d_head), not 1/sqrt(d_model), for multi-head."""

    def test_scale_factor_differs_by_head_count(self) -> None:
        """With d_model=16:
        - n_heads=4: d_head=4, scale=1/sqrt(4)=0.5
        - n_heads=1: d_head=16, scale=1/sqrt(16)=0.25
        Using identity weights and same input, the raw QK^T values should
        differ by the ratio of scale factors: 0.5/0.25 = 2.0.
        """
        torch.manual_seed(42)
        d_model = 16
        T = 4

        # Config 1: n_heads=4, d_head=4, scale=0.5
        attn_4h = CausalSelfAttention(d_model=d_model, n_heads=4, max_seq_len=T, dropout=0.0)
        # Config 2: n_heads=1, d_head=16, scale=0.25
        attn_1h = CausalSelfAttention(d_model=d_model, n_heads=1, max_seq_len=T, dropout=0.0)

        # Set both to identity weights so Q=x, K=x -> QK^T = x @ x.T * scale
        Wq_identity = torch.eye(d_model)
        Wk_identity = torch.eye(d_model)
        with torch.no_grad():
            attn_4h.W_q.weight.copy_(Wq_identity)
            attn_4h.W_k.weight.copy_(Wk_identity)
            attn_1h.W_q.weight.copy_(Wq_identity)
            attn_1h.W_k.weight.copy_(Wk_identity)

        x = torch.randn(1, T, d_model)

        _, internals_4h = attn_4h(x, extract=True)
        _, internals_1h = attn_1h(x, extract=True)

        # For 4-head: each head sees d_head=4 dimensional slice, but with identity
        # weights the raw dot products are the same as the 1-head case (just partitioned).
        # However, the SCALE differs: 0.5 vs 0.25.
        #
        # For a concrete check: compute unscaled x @ x.T for the first head's slice
        # x_h0 = x[:, :, 0:4]  (first 4 dims)
        # unscaled = x_h0 @ x_h0.T  (without scale)
        # 4h result = unscaled * 0.5
        # 1h result at same positions = (full x @ x.T)[:, :4_related] * 0.25
        #
        # Simpler: verify the scale values directly
        scale_4h = 1.0 / math.sqrt(4)   # 0.5
        scale_1h = 1.0 / math.sqrt(16)  # 0.25

        assert abs(scale_4h - 0.5) < 1e-10, f"4h scale should be 0.5, got {scale_4h}"
        assert abs(scale_1h - 0.25) < 1e-10, f"1h scale should be 0.25, got {scale_1h}"

        # Verify through the model: compute unscaled QK^T manually for head 0 of 4h config
        x_h0 = x[:, :, 0:4]  # first d_head=4 dimensions
        unscaled_h0 = x_h0 @ x_h0.transpose(-2, -1)  # [1, T, T]

        # The 4h model should produce qkt_h0 = unscaled_h0 * 0.5
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        expected_4h = unscaled_h0 * scale_4h
        torch.testing.assert_close(
            internals_4h.qkt[0, 0][mask],
            expected_4h[0][mask],
            atol=1e-5,
            rtol=1e-5,
        )

        # The 1h model should produce qkt = (x @ x.T) * 0.25
        unscaled_full = x @ x.transpose(-2, -1)  # [1, T, T]
        expected_1h = unscaled_full * scale_1h
        torch.testing.assert_close(
            internals_1h.qkt[0, 0][mask],
            expected_1h[0][mask],
            atol=1e-5,
            rtol=1e-5,
        )
