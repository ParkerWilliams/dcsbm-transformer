"""TransformerLM: NanoGPT-scale single-head causal language model.

Full model with learned token/positional embeddings, transformer blocks,
output projection head, and configurable extraction modes for SVD analysis.
No weight tying, no Flash Attention, no multi-head reshaping.
"""

import math

import torch
import torch.nn as nn

from src.model.block import TransformerBlock
from src.model.types import ExtractionMode, ForwardOutput


class TransformerLM(nn.Module):
    """NanoGPT-scale single-head causal transformer language model.

    Processes vertex-ID token sequences and exposes internal attention
    components for three-target SVD stability analysis.

    Args:
        vocab_size: Number of tokens (= number of graph vertices, no special tokens).
        d_model: Model dimension.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length (context window w).
        dropout: Dropout rate applied to embeddings, attention, and MLP.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store config for inspection
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token and position embeddings (GPT-2 style learned positional)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, max_seq_len, dropout) for _ in range(n_layers)]
        )

        # Final layer norm (pre-norm convention: LN before output head)
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection -- SEPARATE from token_embedding (no weight tying)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight initialization: Normal(0, 0.02) for all, then residual scaling
        self.apply(self._init_weights)

        # Special residual projection scaling: 1/sqrt(2 * n_layers)
        # Applied to W_o in attention and second linear in MLP (index 2 in Sequential)
        for block in self.blocks:
            torch.nn.init.normal_(
                block.attention.W_o.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layers),
            )
            torch.nn.init.normal_(
                block.mlp[2].weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layers),
            )

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        mode: ExtractionMode = ExtractionMode.NONE,
    ) -> ForwardOutput:
        """Forward pass with configurable extraction.

        Args:
            idx: Token indices of shape [B, T] where T <= max_seq_len.
            mode: Extraction mode controlling what internals are returned.

        Returns:
            ForwardOutput with logits always populated, other fields based on mode.
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )

        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)  # [B, T, D]
        pos_ids = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos_ids)  # [T, D]
        x = self.embed_dropout(tok_emb + pos_emb)

        # Determine extraction level
        extract = mode != ExtractionMode.NONE
        collect_residual = mode in (ExtractionMode.RESIDUAL, ExtractionMode.FULL)

        # Per-layer extraction collectors
        all_qkt: list[torch.Tensor] = []
        all_attn: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []

        # Residual stream collector (includes pre-block embedding state)
        residual_states: list[torch.Tensor] = []
        if collect_residual:
            residual_states.append(x.detach())

        # Process through transformer blocks
        for block in self.blocks:
            x, internals = block(x, extract=extract)
            if extract and internals is not None:
                all_qkt.append(internals.qkt)
                all_attn.append(internals.attention_weights)
                all_values.append(internals.values)
            if collect_residual:
                residual_states.append(x.detach())

        # Final layer norm + output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B, T, vocab_size]

        # Build ForwardOutput
        qkt = None
        attention_weights = None
        values = None
        residual_stream = None
        residual_norms = None

        if extract and all_qkt:
            qkt = torch.stack(all_qkt, dim=1)  # [B, n_layers, T, T]
            attention_weights = torch.stack(all_attn, dim=1)  # [B, n_layers, T, T]
            values = torch.stack(all_values, dim=1)  # [B, n_layers, T, D]

        if collect_residual and residual_states:
            # Stack: [B, T, n_layers+1, D]
            # residual_states is a list of [B, T, D] tensors (n_layers+1 of them)
            residual_stream = torch.stack(residual_states, dim=2)
            # L2 norms: [B, T, n_layers+1]
            residual_norms = torch.linalg.norm(residual_stream, dim=-1)

        return ForwardOutput(
            logits=logits,
            qkt=qkt,
            attention_weights=attention_weights,
            values=values,
            residual_stream=residual_stream,
            residual_norms=residual_norms,
        )

    def get_wvwo(self) -> torch.Tensor:
        """Return stacked WvWo weight product for all layers.

        WvWo represents the OV circuit: the composition of value and output
        projections. This is input-agnostic (depends only on model weights).

        nn.Linear stores weights as [out_features, in_features].
        W_v.weight is [d_model, d_model], so W_v.weight.T is the actual
        value projection matrix. The product W_v.weight.T @ W_o.weight
        gives [d_model, d_model] mapping input space through value to output.

        Returns:
            Tensor of shape [n_layers, d_model, d_model], detached.
        """
        return torch.stack(
            [
                block.attention.W_v.weight.T @ block.attention.W_o.weight
                for block in self.blocks
            ]
        ).detach()


def create_model(config: "ExperimentConfig") -> TransformerLM:
    """Factory function to create TransformerLM from ExperimentConfig.

    Derives all constructor arguments from the experiment configuration:
        - vocab_size from config.graph.n (MODL-03: tokens are vertex IDs)
        - d_model from config.model.d_model (MODL-01: configurable)
        - n_layers from config.model.n_layers (MODL-01: configurable)
        - max_seq_len from config.training.w (context window)
        - dropout from config.model.dropout

    Args:
        config: Experiment configuration with graph, model, and training sub-configs.

    Returns:
        Initialized TransformerLM ready for training.
    """
    from src.config.experiment import ExperimentConfig

    return TransformerLM(
        vocab_size=config.graph.n,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        max_seq_len=config.training.w,
        dropout=config.model.dropout,
    )
