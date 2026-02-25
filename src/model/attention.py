"""Single-head causal self-attention with transparent QK^T extraction.

No Flash Attention, no multi-head reshaping. Q, K, V are each [B, T, D].
Manual scaled dot-product attention enables extraction of raw QK^T matrix,
attention weights, and value matrices for SVD stability analysis.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.types import AttentionInternals


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention with optional extraction.

    Implements manual scaled dot-product attention to expose internal
    components for three-target SVD analysis (QK^T routing, WvWo OV circuit,
    AVWo net residual update).

    Args:
        d_model: Model dimension. Q, K, V are each [B, T, d_model].
        max_seq_len: Maximum sequence length for causal mask buffer.
        dropout: Dropout rate for attention weights and residual path.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Separate projection matrices for transparency
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
        """Forward pass with optional extraction of attention internals.

        Args:
            x: Input tensor of shape [B, T, D].
            extract: If True, return detached QK^T, attention weights, and values.

        Returns:
            Tuple of (output [B, T, D], AttentionInternals or None).
        """
        B, T, D = x.shape

        # Project to Q, K, V -- single head, no reshape needed
        q = self.W_q(x)  # [B, T, D]
        k = self.W_k(x)  # [B, T, D]
        v = self.W_v(x)  # [B, T, D]

        # Scaled dot-product: QK^T / sqrt(d_model)
        scale = 1.0 / math.sqrt(self.d_model)
        qkt_raw = (q @ k.transpose(-2, -1)) * scale  # [B, T, T]

        # Slice causal mask to current sequence length
        mask = self.causal_mask[:T, :T]

        # Softmax path: -inf fill for future positions
        att_scores = qkt_raw.masked_fill(~mask, float("-inf"))
        att_weights = F.softmax(att_scores, dim=-1)  # [B, T, T]
        att_weights_dropped = self.attn_dropout(att_weights)

        # Compute output: attention-weighted values through output projection
        y = self.resid_dropout(self.W_o(att_weights_dropped @ v))  # [B, T, D]

        if extract:
            # SVD target: zero-filled causal mask (NOT -inf)
            # CRITICAL: Two different masks on the same raw QK^T:
            #   - Zero fill for SVD target (clean input for singular value decomposition)
            #   - -inf fill for softmax (proper probability distribution)
            qkt_target = qkt_raw.masked_fill(~mask, 0.0).detach()
            return y, AttentionInternals(
                qkt=qkt_target,
                attention_weights=att_weights.detach(),
                values=v.detach(),
            )

        return y, None
