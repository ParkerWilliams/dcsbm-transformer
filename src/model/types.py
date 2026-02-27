"""Data types for the transformer model package.

Defines extraction modes, attention internals, and forward output structures
for the NanoGPT-scale causal transformer with multi-head attention support.
"""

from dataclasses import dataclass, field
from enum import StrEnum

import torch


class ExtractionMode(StrEnum):
    """Controls what internal components the forward pass returns.

    NONE: Lean training mode -- logits only, no extraction overhead.
    SVD_TARGETS: QK^T (zero-filled causal), attention weights A, value matrix V
                 per layer. Wv/Wo accessible as model parameters.
    RESIDUAL: SVD targets + full residual stream at every position/layer.
    FULL: All of the above.
    """

    NONE = "none"
    SVD_TARGETS = "svd_targets"
    RESIDUAL = "residual"
    FULL = "full"


@dataclass
class AttentionInternals:
    """Per-layer attention extraction results. All tensors are detached.

    Attributes:
        qkt: Raw QK^T with zero-filled causal mask. Shape [B, n_heads, T, T].
        attention_weights: Softmax attention weights A. Shape [B, n_heads, T, T].
        values: Value matrix V before output projection. Shape [B, n_heads, T, d_head].
    """

    qkt: torch.Tensor  # [B, n_heads, T, T] -- zero-filled (not -inf) for SVD target
    attention_weights: torch.Tensor  # [B, n_heads, T, T] -- valid probability distribution
    values: torch.Tensor  # [B, n_heads, T, d_head]


@dataclass
class ForwardOutput:
    """Structured output from TransformerLM.forward().

    Always contains logits. Other fields populated based on ExtractionMode.

    Attributes:
        logits: Next-token prediction logits. Shape [B, T, vocab_size].
        qkt: Stacked QK^T across layers and heads, zero-filled.
            Shape [B, n_layers, n_heads, T, T].
        attention_weights: Stacked attention weights.
            Shape [B, n_layers, n_heads, T, T].
        values: Stacked value matrices.
            Shape [B, n_layers, n_heads, T, d_head].
        residual_stream: Per-layer residual states. Shape [B, T, n_layers+1, D].
            Includes pre-block embedding output as first state.
        residual_norms: L2 norms of residual states. Shape [B, T, n_layers+1].
    """

    logits: torch.Tensor  # [B, T, vocab_size]
    qkt: torch.Tensor | None = None  # [B, n_layers, n_heads, T, T]
    attention_weights: torch.Tensor | None = None  # [B, n_layers, n_heads, T, T]
    values: torch.Tensor | None = None  # [B, n_layers, n_heads, T, d_head]
    residual_stream: torch.Tensor | None = None  # [B, T, n_layers+1, D]
    residual_norms: torch.Tensor | None = None  # [B, T, n_layers+1]
