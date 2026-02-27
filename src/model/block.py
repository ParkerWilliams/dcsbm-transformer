"""Pre-norm transformer block with attention + MLP + residual connections.

Follows the NanoGPT / GPT-2 architecture: LayerNorm before each sublayer,
GELU activation in the MLP, 4x expansion factor.
"""

import torch
import torch.nn as nn

from src.model.attention import CausalSelfAttention
from src.model.types import AttentionInternals


class TransformerBlock(nn.Module):
    """Pre-norm transformer block.

    Architecture:
        x -> LN -> Attention -> + residual -> LN -> MLP -> + residual

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads (1, 2, or 4).
        max_seq_len: Maximum sequence length (passed to attention).
        dropout: Dropout rate for attention and MLP.
    """

    def __init__(self, d_model: int, n_heads: int = 1, max_seq_len: int = 64, dropout: float = 0.0):
        super().__init__()

        # Pre-norm layer norms
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

        # Attention sublayer
        self.attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)

        # MLP sublayer: 4x expansion with GELU activation (NanoGPT standard)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, extract: bool = False
    ) -> tuple[torch.Tensor, AttentionInternals | None]:
        """Forward pass with optional extraction.

        Args:
            x: Input tensor of shape [B, T, D].
            extract: If True, return attention internals from this block.

        Returns:
            Tuple of (output [B, T, D], AttentionInternals or None).
        """
        # Pre-norm attention with residual connection
        attn_out, internals = self.attention(self.ln_1(x), extract=extract)
        x = x + attn_out

        # Pre-norm MLP with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x, internals
