"""Public API for the transformer model package.

Exports:
    ExtractionMode: String enum controlling forward pass extraction level.
    ForwardOutput: Structured output containing logits and optional internals.
    AttentionInternals: Per-layer attention extraction results.
"""

from src.model.types import AttentionInternals, ExtractionMode, ForwardOutput
