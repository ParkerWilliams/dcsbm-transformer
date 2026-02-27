"""TransformerLM: NanoGPT-scale causal language model with multi-head attention.

Full model with learned token/positional embeddings, transformer blocks,
output projection head, and configurable extraction modes for SVD analysis.
Supports n_heads = 1 (backward compatible with v1.0), 2, or 4.
No weight tying, no Flash Attention.
"""

import math

import torch
import torch.nn as nn

from src.model.block import TransformerBlock
from src.model.types import ExtractionMode, ForwardOutput


class TransformerLM(nn.Module):
    """NanoGPT-scale causal transformer language model with multi-head attention.

    Processes vertex-ID token sequences and exposes internal attention
    components for three-target SVD stability analysis. Supports configurable
    n_heads for multi-head ablation studies.

    Args:
        vocab_size: Number of tokens (= number of graph vertices, no special tokens).
        d_model: Model dimension (must be divisible by n_heads).
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads (1, 2, or 4).
        max_seq_len: Maximum sequence length (context window w).
        dropout: Dropout rate applied to embeddings, attention, and MLP.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int = 1,
        max_seq_len: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Store config for inspection
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Token and position embeddings (GPT-2 style learned positional)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
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
            When extracting:
                qkt: [B, n_layers, n_heads, T, T]
                attention_weights: [B, n_layers, n_heads, T, T]
                values: [B, n_layers, n_heads, T, d_head]
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
                all_qkt.append(internals.qkt)        # [B, n_heads, T, T]
                all_attn.append(internals.attention_weights)  # [B, n_heads, T, T]
                all_values.append(internals.values)   # [B, n_heads, T, d_head]
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
            # Each entry is [B, n_heads, T, T]; stack on dim=1 gives [B, n_layers, n_heads, T, T]
            qkt = torch.stack(all_qkt, dim=1)
            attention_weights = torch.stack(all_attn, dim=1)
            values = torch.stack(all_values, dim=1)

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
        """Return stacked per-head WvWo weight product for all layers.

        WvWo represents the OV circuit: the composition of value and output
        projections. This is input-agnostic (depends only on model weights).

        For each layer and each head, computes the per-head OV circuit:
            Wv_h = W_v.weight[h*d_head:(h+1)*d_head, :]  -- [d_head, d_model]
            Wo_h = W_o.weight[:, h*d_head:(h+1)*d_head]  -- [d_model, d_head]
            WvWo_h = Wv_h.T @ Wo_h.T = [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]

        This maps input -> value head h -> output projection head h -> residual stream.

        Returns:
            Tensor of shape [n_layers, n_heads, d_model, d_model], detached.
        """
        layers = []
        for block in self.blocks:
            heads = []
            Wv = block.attention.W_v.weight  # [d_model, d_model]
            Wo = block.attention.W_o.weight  # [d_model, d_model]
            for h in range(self.n_heads):
                start = h * self.d_head
                end = (h + 1) * self.d_head
                # Per-head value projection: Wv[start:end, :] is [d_head, d_model]
                # nn.Linear: output = input @ weight.T, so V_full = x @ Wv.T
                # Head h gets V_full[:, :, start:end] = x @ Wv[start:end, :].T
                Wv_h = Wv[start:end, :]  # [d_head, d_model]
                # Per-head output projection: Wo[:, start:end] is [d_model, d_head]
                # After concat, output = concat @ Wo.T, head h contributes at [start:end]
                Wo_h = Wo[:, start:end]  # [d_model, d_head]
                # OV circuit: Wv_h.T @ Wo_h.T = [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
                wvwo_h = Wv_h.T @ Wo_h.T
                heads.append(wvwo_h)
            layers.append(torch.stack(heads))  # [n_heads, d_model, d_model]
        return torch.stack(layers).detach()  # [n_layers, n_heads, d_model, d_model]


def create_model(config: "ExperimentConfig") -> TransformerLM:
    """Factory function to create TransformerLM from ExperimentConfig.

    Derives all constructor arguments from the experiment configuration:
        - vocab_size from config.graph.n (MODL-03: tokens are vertex IDs)
        - d_model from config.model.d_model (MODL-01: configurable)
        - n_layers from config.model.n_layers (MODL-01: configurable)
        - n_heads from config.model.n_heads (MHAD-01: 1, 2, or 4)
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
        n_heads=config.model.n_heads,
        max_seq_len=config.training.w,
        dropout=config.model.dropout,
    )
