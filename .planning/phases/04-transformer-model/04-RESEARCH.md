# Phase 4: Transformer Model - Research

**Researched:** 2026-02-25
**Domain:** NanoGPT-scale single-head transformer with internal attention extraction for SVD analysis
**Confidence:** HIGH

## Summary

Phase 4 requires implementing a minimal, fully transparent GPT-2-style decoder-only transformer that processes vertex-ID token sequences and exposes its internal attention components (QK^T, A, V, Wv, Wo) for three-target SVD stability analysis. The model is purely an instrument -- training and SVD metric computation are separate phases (5 and 6 respectively).

The critical constraint is that `torch.nn.functional.scaled_dot_product_attention` (Flash Attention) does NOT return attention weights or intermediate QK^T values. This means we MUST implement manual scaled dot-product attention to extract the raw QK^T matrix, attention weights A, and value matrix V. This is the fundamental architectural decision driving the entire implementation. NanoGPT's reference implementation already shows the manual attention path as a fallback, and this project requires it exclusively.

The implementation follows the well-established NanoGPT pattern: pre-norm transformer blocks with learned positional embeddings, GELU MLP, and a separate output projection head. The single-head constraint (already enforced by ExperimentConfig validation) simplifies the architecture since there is no head dimension reshaping needed -- Q, K, V are each [batch, seq, d_model] directly.

**Primary recommendation:** Implement a from-scratch single-head causal transformer following NanoGPT's architecture, with manual attention computation and a StrEnum-controlled extraction mode parameter on `forward()`.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- Three SVD targets per attention layer: QK^T (routing), WvWo (OV circuit), AVWo (net residual update)
- All targets extracted from all layers as stacked tensors with layer dimension
- All extracted tensors detached from computation graph
- Extraction during evaluation only (not training forward passes)
- WvWo tracked in both training and evaluation contexts
- Model runs in float32; float64 upcast is Phase 6 responsibility
- SVD extraction is per-sequence, batch-computed
- Single forward() method with string enum parameter (4 modes): no extraction, SVD targets, SVD targets + residual stream, full extraction
- Residual stream: full tensor within jumper event windows, L2 norm elsewhere
- Pure vertex IDs only (no BOS/PAD/EOS tokens), randomly permuted
- Vocabulary size from graph config n
- Sliding window with stride 1, context window w
- Learned positional embeddings (GPT-2 style), shape [w, d_model]
- Token embedding dimension equals d_model directly (no projection layer)
- Random initialization for token embeddings (Xavier/Kaiming)
- Separate input/output weight matrices (no weight tying)
- Attention components only -- no MLP exposure
- Numerical precision: model in float32, no float16 forcing

### Claude's Discretion
- Layer normalization placement (pre-norm vs post-norm)
- Dropout implementation details
- Exact weight initialization scheme (Xavier vs Kaiming vs custom)
- Internal tensor layout for efficient extraction
- GELU vs ReLU activation in MLP blocks
- Exact string enum values for the forward pass extraction parameter

### Deferred Ideas (OUT OF SCOPE)
- Cross-layer instability propagation analysis (Phase 7)
- Baseline SVD values on untrained model (Phase 6)
- MLP contribution exposure (explicitly excluded)

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODL-01 | NanoGPT-scale transformer with configurable d_model (64, 128, 256), n_layers (2, 4, 6), exactly 1 attention head | NanoGPT architecture pattern with single-head simplification; config already validates n_heads=1; ModelConfig dataclass already defines d_model, n_layers, n_heads, dropout |
| MODL-02 | Model exposes internal components for SVD: raw QK^T (causal-masked, zero-filled), attention weights A, value matrix V, Wv/Wo parameters -- enabling three SVD targets | Manual attention implementation (no Flash Attention); detached tensor extraction; StrEnum forward pass modes; WvWo as weight parameter product; AVWo computed from A, V, Wo |
| MODL-03 | Vocabulary size equals number of graph vertices (tokens are vertex IDs) | nn.Embedding with vocab_size=config.graph.n; no special tokens; output projection nn.Linear(d_model, n) |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10.0 | Model definition, tensor ops, autograd | Already installed in project venv; all existing code uses PyTorch |
| torch.nn | (bundled) | nn.Module, nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout, nn.GELU | Standard PyTorch neural network building blocks |
| torch.nn.functional | (bundled) | softmax, cross_entropy | Functional ops for attention computation |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| enum (stdlib) | Python 3.12+ | StrEnum for forward pass extraction mode | Defining the 4-mode extraction parameter |
| dataclasses (stdlib) | Python 3.12+ | Return type for forward pass internals | Structuring extraction output |
| math (stdlib) | Python 3.12+ | sqrt for attention scaling | 1/sqrt(d_model) scaling factor |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual attention | torch.nn.MultiheadAttention(need_weights=True) | MHA wraps too much; QK^T before softmax not accessible; flash attention may silently skip weight return |
| Manual attention | torch.nn.functional.scaled_dot_product_attention | Does NOT return attention weights at all; unsuitable for this project |
| From-scratch model | HuggingFace Transformers | Explicitly out of scope per REQUIREMENTS.md; obscures QK^T extraction |

**Installation:**
```bash
# No new packages needed -- torch 2.10.0 already in venv
```

## Architecture Patterns

### Recommended Project Structure
```
src/
  model/
    __init__.py          # Public API: TransformerLM, ExtractionMode, ForwardOutput
    attention.py         # CausalSelfAttention with manual QK^T extraction
    block.py             # TransformerBlock (pre-norm, attention + MLP + residual)
    transformer.py       # TransformerLM (full model: embeddings, blocks, output head)
    types.py             # ExtractionMode enum, ForwardOutput dataclass, AttentionInternals
```

### Pattern 1: Manual Single-Head Causal Self-Attention
**What:** Compute Q, K, V projections via separate nn.Linear layers (or a fused 3*d_model linear), then explicitly compute QK^T, apply causal mask with zero fill (not -inf), compute softmax on properly masked scores, and return all intermediates.
**When to use:** Always -- this is the core of the model.
**Critical detail:** The CONTEXT.md specifies "causal-masked with zero fill" for the raw QK^T matrix. This means: compute Q @ K^T, then set future positions to 0.0 (not -inf). The -inf masking is applied separately before softmax to compute A. The returned QK^T has zeros in masked positions for clean SVD input.

```python
# Attention computation with extraction
# Q, K, V each [batch, seq, d_model] (single head, no reshape needed)
q = self.W_q(x)  # [B, T, D]
k = self.W_k(x)  # [B, T, D]
v = self.W_v(x)  # [B, T, D]

# Raw QK^T: [B, T, T]
scale = 1.0 / math.sqrt(self.d_model)
qkt_raw = (q @ k.transpose(-2, -1)) * scale

# Causal mask: zero fill for extraction target
causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
qkt_masked = qkt_raw.masked_fill(~causal_mask, 0.0)  # SVD target: zero-filled

# For softmax: -inf masked scores
att_scores = qkt_raw.masked_fill(~causal_mask, float('-inf'))
att_weights = F.softmax(att_scores, dim=-1)  # A: [B, T, T]
att_weights = self.attn_dropout(att_weights)

# Output
y = att_weights @ v  # [B, T, D]
y = self.W_o(y)      # Output projection
```

### Pattern 2: Pre-Norm Transformer Block
**What:** Apply LayerNorm before attention and before MLP, with residual connections wrapping both sublayers.
**When to use:** Every transformer block.
**Why pre-norm:** Pre-norm is definitively better for training stability at small scale. GPT-2, NanoGPT, and all modern LLMs use it. It prevents gradient explosion without careful warmup schedules. The user left this as Claude's discretion; pre-norm is the clear recommendation.

```python
class TransformerBlock(nn.Module):
    def forward(self, x, extraction_mode):
        # Pre-norm attention with residual
        attn_out, internals = self.attention(self.ln_1(x), extraction_mode)
        x = x + attn_out
        # Pre-norm MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x, internals
```

### Pattern 3: StrEnum-Controlled Extraction Mode
**What:** A Python 3.11+ StrEnum with 4 values controlling what the forward pass returns.
**When to use:** The single forward() method parameter.

```python
from enum import StrEnum

class ExtractionMode(StrEnum):
    NONE = "none"                # Lean training: logits only
    SVD_TARGETS = "svd_targets"  # QK^T, A, V per layer; Wv/Wo accessible as params
    RESIDUAL = "residual"        # SVD targets + residual stream
    FULL = "full"                # All of the above
```

### Pattern 4: ForwardOutput Dataclass
**What:** A structured return type that contains logits and optional extraction data.
**When to use:** Return value of forward().

```python
@dataclass
class ForwardOutput:
    logits: torch.Tensor                              # [B, T, vocab_size]
    qkt: torch.Tensor | None = None                   # [B, n_layers, T, T] - zero-filled
    attention_weights: torch.Tensor | None = None      # [B, n_layers, T, T]
    values: torch.Tensor | None = None                 # [B, n_layers, T, D]
    residual_stream: torch.Tensor | None = None        # varies by mode
    residual_norms: torch.Tensor | None = None         # [B, T, n_layers]
```

### Pattern 5: GPT-2 Style Weight Initialization
**What:** Normal(0, 0.02) for all weights, zeros for biases, with residual projection scaling by 1/sqrt(2*n_layers).
**When to use:** Model __init__.
**Why:** This is the NanoGPT standard. Matches GPT-2 paper. The user left initialization as discretion -- GPT-2 style normal(0, 0.02) is the NanoGPT convention and works well for this scale.

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

Special scaling for output projections (W_o) in attention:
```python
# After apply(self._init_weights), scale residual projections
for block in self.blocks:
    torch.nn.init.normal_(
        block.attention.W_o.weight,
        mean=0.0,
        std=0.02 / math.sqrt(2 * self.n_layers)
    )
```

### Pattern 6: Single-Head Simplification
**What:** With n_heads=1, Q/K/V are each [B, T, d_model] -- no head dimension reshape needed.
**Why it matters:** NanoGPT typically reshapes to [B, n_heads, T, head_dim]. With n_heads=1, head_dim=d_model and the reshape is identity. The code should skip the reshape entirely for clarity and to match the "transparent, readable" goal from CONTEXT.md.

### Anti-Patterns to Avoid
- **Using Flash Attention / scaled_dot_product_attention:** Does NOT return QK^T or attention weights. The entire extraction mechanism depends on manual attention.
- **Weight tying (embedding = lm_head):** CONTEXT.md explicitly requires separate input/output weight matrices. Do not set `lm_head.weight = wte.weight`.
- **Adding special tokens (BOS/PAD/EOS):** CONTEXT.md explicitly states pure vertex IDs only. vocab_size = config.graph.n exactly.
- **Multi-head reshaping with n_heads=1:** Unnecessary complexity. Keep tensors [B, T, D] throughout attention.
- **Returning attached tensors for extraction:** All extracted tensors MUST be `.detach()` -- no gradient flow through the extraction path.
- **Extracting during training mode:** Extraction is evaluation-only. The enum mode can be used during training but extraction tensors should only be populated in eval (torch.no_grad context).
- **Post-norm architecture:** Less stable for training; pre-norm is the clear choice for this scale.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Layer normalization | Custom LN | `nn.LayerNorm(d_model)` | Numerically stable, GPU-optimized, handles epsilon correctly |
| GELU activation | Manual GELU approx | `nn.GELU()` | PyTorch's exact GELU implementation; no approximation needed |
| Embedding lookup | Manual indexing | `nn.Embedding(vocab_size, d_model)` | Handles sparse gradients efficiently, GPU-accelerated |
| Softmax | Manual exp/sum | `F.softmax(x, dim=-1)` | Numerically stable log-sum-exp trick built in |
| Cross entropy loss | Manual log-softmax + NLL | `F.cross_entropy(logits, targets)` | Fused, numerically stable, handles class dimension correctly |
| Causal mask | Dynamic per-call | `register_buffer("causal_mask", torch.tril(...))` | Created once, moved with model device, no gradient tracking |

**Key insight:** The model itself is simple -- NanoGPT is ~300 lines. The complexity is in the extraction interface design, not the transformer mechanics. Do not over-engineer the transformer; focus engineering effort on the extraction mode API and tensor organization.

## Common Pitfalls

### Pitfall 1: Flash Attention Silently Dropping Weights
**What goes wrong:** Using `torch.nn.MultiheadAttention` with `need_weights=True` may still not return attention weights when Flash Attention is active (PyTorch 2.0+). The weights come back as None or averaged.
**Why it happens:** Flash Attention is an optimization that never materializes the full attention matrix.
**How to avoid:** Implement attention manually. Do NOT use `nn.MultiheadAttention` or `F.scaled_dot_product_attention`.
**Warning signs:** Attention weights are None, or have unexpected shape.

### Pitfall 2: QK^T Mask Convention Mismatch
**What goes wrong:** Using -inf for the QK^T SVD target (should be 0.0) or using 0.0 for the softmax input (should be -inf).
**Why it happens:** Two different masking conventions needed for two different purposes.
**How to avoid:** Compute QK^T once, then create two views: zero-filled for SVD target, -inf-filled for softmax. The raw QK^T before any masking is the computation; apply masks afterward.
**Warning signs:** SVD on QK^T produces extreme singular values (from -inf entries), or softmax gives uniform attention (from 0.0 fill instead of -inf).

### Pitfall 3: Gradient Flow Through Extracted Tensors
**What goes wrong:** Extracted QK^T/A/V tensors are still attached to the computation graph, causing memory leaks or unintended gradient updates.
**Why it happens:** Forgetting to call `.detach()` on extracted tensors.
**How to avoid:** Every tensor stored in ForwardOutput (except logits) must be `.detach()`. Use a helper method that always detaches.
**Warning signs:** GPU memory grows unexpectedly during evaluation, or training loss changes when extraction is enabled.

### Pitfall 4: Vocabulary Size Mismatch
**What goes wrong:** Model uses wrong vocab_size (e.g., hardcoded 500 instead of reading from config.graph.n).
**Why it happens:** Not threading graph config through to model construction.
**How to avoid:** Model constructor takes vocab_size as parameter, factory function derives it from ExperimentConfig.graph.n.
**Warning signs:** Index out of bounds errors when feeding walk tokens, or unused embedding rows.

### Pitfall 5: Positional Embedding Size vs Context Window
**What goes wrong:** Positional embeddings sized to walk_length instead of w (context window).
**Why it happens:** Confusing walk_length (full walk) with w (sliding window input size).
**How to avoid:** Position embeddings shape [w, d_model]. The sliding window produces input sequences of exactly length w. Position IDs are always 0..w-1.
**Warning signs:** Positional embedding index errors, or model receiving sequences longer than embedding table.

### Pitfall 6: Residual Projection Scaling Omitted
**What goes wrong:** Output projections (W_o in attention, second linear in MLP) not scaled by 1/sqrt(2*n_layers), causing training instability with deeper configs (n_layers=6).
**Why it happens:** Forgetting the GPT-2 initialization convention for residual projections.
**How to avoid:** After `apply(_init_weights)`, apply special scaling to W_o and MLP output projection weights.
**Warning signs:** Loss diverges with n_layers=6 but works with n_layers=2.

### Pitfall 7: Deterministic Algorithms Incompatibility
**What goes wrong:** Model operations fail with `RuntimeError: ... does not have a deterministic implementation` when `torch.use_deterministic_algorithms(True)` is set (which this project uses).
**Why it happens:** Some CUDA ops lack deterministic kernels. Scatter/gather operations in embedding backward pass can trigger this.
**How to avoid:** The project already sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Embedding backward is deterministic in PyTorch 2.10. Test early with the seed module.
**Warning signs:** RuntimeError on backward pass mentioning deterministic algorithms.

### Pitfall 8: WvWo Extraction Confusion
**What goes wrong:** Computing WvWo at runtime during forward pass instead of exposing it as a weight parameter product.
**Why it happens:** Confusing the three SVD targets' computation paths.
**How to avoid:** WvWo is input-agnostic (weight matrices only). It is `W_v.weight @ W_o.weight` (or transposed, depending on convention). Phase 6 computes this from model parameters; Phase 4 just needs to ensure W_v and W_o are accessible as named parameters.
**Warning signs:** WvWo changes within a batch (it shouldn't -- it's the same for all inputs at a given checkpoint).

## Code Examples

### Complete Attention Module Pattern

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Single-head causal self-attention with transparent QK^T extraction.

    No Flash Attention, no multi-head reshaping. Q, K, V are each [B, T, D].
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Pre-compute causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor, extract: bool = False):
        B, T, D = x.shape
        q = self.W_q(x)  # [B, T, D]
        k = self.W_k(x)  # [B, T, D]
        v = self.W_v(x)  # [B, T, D]

        # Scaled dot-product: QK^T / sqrt(d)
        scale = 1.0 / math.sqrt(D)
        qkt_raw = (q @ k.transpose(-2, -1)) * scale  # [B, T, T]

        # Causal mask slice for current sequence length
        mask = self.causal_mask[:T, :T]

        # For softmax: -inf in future positions
        att_scores = qkt_raw.masked_fill(~mask, float("-inf"))
        att_weights = F.softmax(att_scores, dim=-1)  # A: [B, T, T]
        att_weights_dropped = self.attn_dropout(att_weights)

        # Compute output
        y = att_weights_dropped @ v  # [B, T, D]
        y = self.resid_dropout(self.W_o(y))

        if extract:
            # QK^T target: zero-filled (not -inf)
            qkt_target = qkt_raw.masked_fill(~mask, 0.0).detach()
            return y, qkt_target, att_weights.detach(), v.detach()

        return y, None, None, None
```

### Complete Model Forward Pass Pattern

```python
def forward(self, idx: torch.Tensor, mode: ExtractionMode = ExtractionMode.NONE):
    B, T = idx.shape
    assert T <= self.max_seq_len

    # Embeddings
    tok_emb = self.token_embedding(idx)            # [B, T, D]
    pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # [T, D]
    x = self.embed_dropout(tok_emb + pos_emb)

    # Extraction collectors
    extract = mode != ExtractionMode.NONE
    all_qkt, all_attn, all_values = [], [], []

    # Transformer blocks
    for block in self.blocks:
        x, internals = block(x, extract=extract)
        if extract and internals is not None:
            all_qkt.append(internals.qkt)
            all_attn.append(internals.attention_weights)
            all_values.append(internals.values)

    # Final layer norm + output projection
    x = self.ln_f(x)
    logits = self.lm_head(x)  # [B, T, vocab_size]

    # Build output
    output = ForwardOutput(logits=logits)
    if extract:
        output.qkt = torch.stack(all_qkt, dim=1)              # [B, n_layers, T, T]
        output.attention_weights = torch.stack(all_attn, dim=1)  # [B, n_layers, T, T]
        output.values = torch.stack(all_values, dim=1)          # [B, n_layers, T, D]

    return output
```

### Model Factory from ExperimentConfig

```python
def create_model(config: ExperimentConfig) -> TransformerLM:
    """Create TransformerLM from experiment configuration."""
    return TransformerLM(
        vocab_size=config.graph.n,      # MODL-03: tokens are vertex IDs
        d_model=config.model.d_model,   # MODL-01: configurable 64/128/256
        n_layers=config.model.n_layers, # MODL-01: configurable 2/4/6
        max_seq_len=config.training.w,  # Context window size
        dropout=config.model.dropout,
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual attention (pre-2.0) | Flash Attention via F.sdpa | PyTorch 2.0 (2023) | Faster, but no weight access -- unsuitable here |
| Post-norm (original Transformer) | Pre-norm (GPT-2+) | GPT-2 (2019) | Training stability; now universal standard |
| Weight tying (GPT-2) | Separate matrices (project choice) | N/A | Required by CONTEXT.md for this project |
| Separate Q/K/V linears | Fused c_attn (3*d_model) | NanoGPT convention | Either works; separate is clearer for extraction |
| TorchScript | torch.compile (or neither) | PyTorch 2.0+ | TorchScript deprecated in 2.10; not needed at this scale |

**Deprecated/outdated:**
- `torch.nn.functional.multi_head_attention_forward`: Low-level, not recommended
- TorchScript: Deprecated in PyTorch 2.10; use torch.export instead (not needed here)
- `attn_mask` as float tensor: Boolean masks preferred in modern PyTorch

## Discretion Recommendations

These are areas marked as Claude's discretion in CONTEXT.md. Research-informed recommendations:

| Area | Recommendation | Rationale |
|------|---------------|-----------|
| LayerNorm placement | **Pre-norm** | Universal standard for GPT-style models; better training stability; NanoGPT uses it |
| Dropout | **nn.Dropout on attention weights and residual paths** | NanoGPT pattern; config.model.dropout=0.0 by default so it's a no-op unless changed |
| Weight initialization | **Normal(0, 0.02) with 1/sqrt(2*n_layers) residual scaling** | Exact GPT-2/NanoGPT convention; proven at this scale |
| Internal tensor layout | **Separate W_q/W_k/W_v/W_o linears** | Clearer than fused 3*d_model for extraction; slight speed cost irrelevant at this scale |
| MLP activation | **GELU** | GPT-2 standard; nn.GELU() is numerically exact in PyTorch 2.10 |
| Extraction enum values | **"none", "svd_targets", "residual", "full"** | Descriptive, lowercase per StrEnum convention |

## Open Questions

1. **Residual stream window specification**
   - What we know: Full tensor stored within jumper event windows [jumper_step, jumper_step + r + buffer(5-10)], L2 norm elsewhere
   - What's unclear: The model itself does not know about jumper events at forward pass time. The window selection must happen at evaluation time (Phase 6), not in the model.
   - Recommendation: The model should support returning the full residual stream when mode is RESIDUAL or FULL. Phase 6 handles the windowing logic. The model just exposes residual states at every position and every layer.

2. **WvWo computation responsibility**
   - What we know: WvWo = product of W_v and W_o weight matrices (input-agnostic)
   - What's unclear: Whether Phase 4 model should provide a helper method like `get_wvwo()` or Phase 6 computes it from model.named_parameters()
   - Recommendation: Add a convenience method `get_wvwo() -> Tensor` that returns `W_v.weight.T @ W_o.weight` (properly detached) per layer, stacked [n_layers, d_model, d_model]. This keeps extraction logic co-located with the model that owns the weights.

3. **AVWo construction**
   - What we know: AVWo = A @ V @ Wo per layer, per sequence
   - What's unclear: Phase 4 provides A, V, and access to Wo. Does Phase 4 compute AVWo or does Phase 6?
   - Recommendation: Phase 4 returns A, V (detached). Phase 6 constructs AVWo = A @ V @ Wo since it is a derived metric target, not a model internal. The model provides the ingredients.

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10 documentation: torch.nn.functional.scaled_dot_product_attention does NOT return attention weights (verified)
- PyTorch 2.10 documentation: nn.MultiheadAttention need_weights may not return weights with Flash Attention (verified)
- NanoGPT model.py (github.com/karpathy/nanoGPT): Pre-norm architecture, GPT-2 initialization, manual attention fallback pattern
- PyTorch 2.10 documentation: torch.nn.init (normal_, zeros_), nn.LayerNorm, nn.Embedding, nn.GELU
- Python 3.12 stdlib: enum.StrEnum (available since 3.11)

### Secondary (MEDIUM confidence)
- GPT-2 weight initialization: Normal(0, 0.02) with 1/sqrt(2*n_layers) scaling for residual projections (multiple sources agree: NanoGPT code, HuggingFace GPT-2, Annotated GPT-2)
- Pre-norm vs Post-norm: Pre-norm universally preferred for training stability at small scale (multiple 2024-2025 analyses agree)

### Tertiary (LOW confidence)
- None. All findings verified with primary or multiple secondary sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - PyTorch 2.10 already installed and used throughout project; no new deps
- Architecture: HIGH - NanoGPT is a well-documented, widely-replicated architecture; single-head simplifies it further
- Pitfalls: HIGH - Flash Attention limitation verified in official PyTorch docs; mask conventions well-documented
- Extraction API: MEDIUM - The 4-mode enum is a project-specific design; patterns are sound but untested in this specific context

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (stable domain; PyTorch 2.10 is current)
