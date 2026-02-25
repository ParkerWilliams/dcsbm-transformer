# Phase 4: Transformer Model - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

A minimal NanoGPT-scale single-head transformer that processes vertex-ID token sequences and exposes internal attention components for three-target SVD stability analysis (QK^T routing, WvWo OV circuit, AVWo net residual update). The model is the instrument; training and SVD metric computation are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Three SVD Targets — Model Extraction Requirements
- The model must expose internal components for three distinct SVD targets per attention layer:
  1. **QK^T** — Routing stability: raw QK^T matrix, causal-masked with zero fill, shape [w, w]
  2. **WvWo** — OV circuit stability (input-agnostic): product of value and output projection weight matrices
  3. **AVWo** — Net residual stream update stability: raw head contribution A @ V @ Wo (no residual connection added)
- All three targets extracted from **all layers**, returned as stacked tensors with a layer dimension (e.g., qkt shape [n_layers, w, w])
- All extracted tensors **detached** from the computation graph (no gradient flow through extraction path)
- Extraction happens during **evaluation only** (not during training forward passes)
- **Attention components only** — MLP contributions are not exposed
- WvWo is tracked in **both training and evaluation contexts**: once per evaluation pass (frozen weights), and per-checkpoint during training (to track OV circuit evolution)

### Numerical Precision
- Model runs in **float32** (standard for training)
- Extracted tensors upcast to **float64** before SVD computation (Phase 6 responsibility, but model should not prevent this by e.g., forcing float16)

### Batch Handling
- SVD extraction is **per-sequence, batch-computed**: use batched tensor operations for GPU efficiency, but store results per-sequence for per-event analysis granularity

### Forward Pass API
- Single `forward()` method with a **string enum** parameter controlling extraction level (4 modes):
  1. No extraction (lean training mode — logits only)
  2. SVD target tensors (QK^T, A, V for AVWo construction; Wv/Wo accessible as model parameters)
  3. SVD targets + residual stream
  4. Full extraction (all of the above)
- No separate methods — one code path with conditional returns

### Residual Stream Exposure
- Residual stream exposed **selectively**, not at every token step
- **Full tensor** [seq_pos, n_layers, d_model] stored within jumper event windows: [jumper_step, jumper_step + r + buffer(5-10 steps)]
- **L2 norm per layer** stored at all other positions
- No pre-jumper residual history needed (SVD time series covers that)
- Window is asymmetric and forward-looking from the jumper encounter through the expected violation step plus a small aftermath buffer

### Vocabulary and Special Tokens
- **Pure vertex IDs** only: vocabulary = {0, 1, ..., n-1}, no BOS/PAD/EOS tokens
- Vertex IDs assigned via **random permutation** so block membership is not encoded in the ID ordering
- Vocabulary size parameter derived from graph config (n)

### Sequence Handling
- Walks (length 2w, 4w, 8w) fed to model via **sliding window with stride 1**
- Context window w determines the input sequence length at each step
- Each position in the walk gets a next-token prediction

### Embedding Design
- **Learned positional embeddings** (GPT-2 style), shape [w, d_model]
- Token embedding dimension **equals d_model** directly (no projection layer)
- **Random initialization** for token embeddings (Xavier/Kaiming — no graph-informed init)
- **Separate input/output weight matrices** (no weight tying between embedding and output projection)

### Downstream Metric Computation (Phase 6 concern, not Phase 4)
- Scalar metrics per SVD target (7 metrics): stable rank, spectral entropy, spectral gap (+ generalized k=2,4), condition number, rank-1 residual norm, read-write subspace alignment (WvWo only)
- Step-over-step deltas computed by Phase 6/7, not by the model

### Claude's Discretion
- Layer normalization placement (pre-norm vs post-norm)
- Dropout implementation details
- Exact weight initialization scheme (Xavier vs Kaiming vs custom)
- Internal tensor layout for efficient extraction
- GELU vs ReLU activation in MLP blocks
- Exact string enum values for the forward pass extraction parameter

</decisions>

<specifics>
## Specific Ideas

- The three SVD targets decouple because a head can change WHERE it attends (QK^T unstable) while keeping WHAT it does constant (WvWo stable), or vice versa. Tracking all three simultaneously enables attribution of instability to its source.
- A rank-1 WvWo is a signature of a clean copying or feature-shifting head. The rank-1 residual norm metric directly measures proximity to this.
- The residual stream is specifically interesting between jumper encounter and violation step — that's where information is either maintained or decays as context shifts.
- "NanoGPT-scale" means this should be a clean, readable implementation — not a framework wrapper. The QK^T extraction needs to be transparent, not bolted onto an opaque library.

</specifics>

<deferred>
## Deferred Ideas

- Cross-layer instability propagation analysis (whether instability in early layers cascades to later layers) — Phase 7 analysis concern
- Baseline SVD values on untrained (random init) model as reference — Phase 6 can compute this before training
- MLP contribution exposure — explicitly excluded; attention-only SVD analysis

</deferred>

---

*Phase: 04-transformer-model*
*Context gathered: 2026-02-25*
