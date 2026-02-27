"""Multi-head causal self-attention with transparent per-head QK^T extraction.

Supports n_heads = 1 (backward compatible with v1.0), 2, or 4.
Manual scaled dot-product attention enables extraction of per-head QK^T matrix,
attention weights, and value matrices for SVD stability analysis.
No Flash Attention -- transparency over speed for research analysis.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.types import AttentionInternals


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with per-head QK^T extraction.

    Supports n_heads = 1 (backward compatible), 2, or 4. Q, K, V are projected
    to [B, T, d_model] then reshaped to [B, n_heads, T, d_head] for per-head
    scaled dot-product attention. Per-head QK^T is [B, n_heads, T, T].

    When n_heads=1, d_head=d_model and behavior is numerically identical to
    the v1.0 single-head implementation.

    Args:
        d_model: Model dimension (must be divisible by n_heads).
        n_heads: Number of attention heads (1, 2, or 4).
        max_seq_len: Maximum sequence length for causal mask buffer.
        dropout: Dropout rate for attention weights and residual path.
    """

    def __init__(
        self, d_model: int, n_heads: int = 1, max_seq_len: int = 64, dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Separate projection matrices for transparency
        # Full d_model -> d_model projections, reshaped to per-head internally
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Pre-computed causal mask: moves with model device automatically
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
        )

    def forward(
        self, x: torch.Tensor, extract: bool = False
    ) -> tuple[torch.Tensor, AttentionInternals | None]:
        """Forward pass with optional per-head extraction of attention internals.

        Args:
            x: Input tensor of shape [B, T, D].
            extract: If True, return detached per-head QK^T, attention weights, and values.

        Returns:
            Tuple of (output [B, T, D], AttentionInternals or None).
            When extracting, AttentionInternals contains per-head tensors:
                qkt: [B, n_heads, T, T]
                attention_weights: [B, n_heads, T, T]
                values: [B, n_heads, T, d_head]
        """
        B, T, D = x.shape

        # Project to Q, K, V: [B, T, d_model]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape to multi-head: [B, T, d_model] -> [B, T, n_heads, d_head] -> [B, n_heads, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product per head: QK^T / sqrt(d_head)
        # NOTE: When n_heads=1, d_head=d_model, so scale = 1/sqrt(d_model) (same as v1.0)
        scale = 1.0 / math.sqrt(self.d_head)
        qkt_raw = (q @ k.transpose(-2, -1)) * scale  # [B, n_heads, T, T]

        # Slice causal mask to current sequence length
        mask = self.causal_mask[:T, :T]  # [T, T]

        # Softmax path: -inf fill for future positions
        att_scores = qkt_raw.masked_fill(~mask, float("-inf"))
        att_weights = F.softmax(att_scores, dim=-1)  # [B, n_heads, T, T]
        att_weights_dropped = self.attn_dropout(att_weights)

        # Compute attention output per head: [B, n_heads, T, d_head]
        attn_out = att_weights_dropped @ v

        # Concatenate heads: [B, n_heads, T, d_head] -> [B, T, n_heads, d_head] -> [B, T, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection + dropout
        y = self.resid_dropout(self.W_o(attn_out))  # [B, T, D]

        if extract:
            # SVD target: zero-filled causal mask (NOT -inf)
            # CRITICAL: Two different masks on the same raw QK^T:
            #   - Zero fill for SVD target (clean input for singular value decomposition)
            #   - -inf fill for softmax (proper probability distribution)
            qkt_target = qkt_raw.masked_fill(~mask, 0.0).detach()  # [B, n_heads, T, T]
            return y, AttentionInternals(
                qkt=qkt_target,
                attention_weights=att_weights.detach(),  # [B, n_heads, T, T]
                values=v.detach(),  # [B, n_heads, T, d_head]
            )

        return y, None
