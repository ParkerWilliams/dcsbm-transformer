"""Public API for the transformer model package.

Exports:
    TransformerLM: NanoGPT-scale single-head causal transformer.
    ExtractionMode: String enum controlling forward pass extraction level.
    ForwardOutput: Structured output containing logits and optional internals.
    AttentionInternals: Per-layer attention extraction results.
    create_model: Factory function to create TransformerLM from ExperimentConfig.
"""

from src.model.transformer import TransformerLM, create_model
from src.model.types import AttentionInternals, ExtractionMode, ForwardOutput
