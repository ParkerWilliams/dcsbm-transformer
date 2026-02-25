# Phase 6: Behavioral Evaluation and SVD Collection - Research

**Researched:** 2026-02-25
**Domain:** Autoregressive generation with fused SVD extraction, numerical linear algebra, behavioral classification
**Confidence:** HIGH

## Summary

Phase 6 implements a fused evaluation pass that performs autoregressive free-generation from seed tokens and simultaneously collects two kinds of data at every generation step: (1) 4-class behavioral labels (edge valid/invalid x rule followed/violated/not-applicable) and (2) SVD metrics across three targets (QK^T routing, WvWo OV circuit, AVWo net residual update) for all layers. This is the largest phase by requirement count (12 requirements) and the core data-collection stage that Phase 7 (predictive horizon analysis) consumes.

The existing codebase provides all necessary building blocks. The model's `ExtractionMode.SVD_TARGETS` mode already returns per-layer QK^T, attention weights, and values in a single forward pass. The `greedy_generate` function in `src/training/evaluate.py` provides the autoregressive loop pattern. The `GraphData` and `JumperInfo` types provide the graph structure needed for behavioral classification. The `write_result` function already supports `token_metrics` NPZ output. The primary implementation work is: (a) modifying the generation loop to extract internals at each step, (b) implementing the 7+1 SVD metric functions with numerical guards, (c) implementing the 4-class behavioral classifier, (d) handling tail extension for late jumper encounters, and (e) organizing output into the `token_metrics.npz` and `result.json` formats.

**Primary recommendation:** Implement as three modules: `src/evaluation/svd_metrics.py` (7+1 metric functions with numerical guards, unit-testable against known matrices), `src/evaluation/behavioral.py` (4-class label assignment, failure_index computation), and `src/evaluation/pipeline.py` (fused generation loop that calls both). Keep SVD metrics as pure functions on tensors for testability per SVD-07.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Autoregressive free-generation: model generates each token from its own previous output
- Seed: single start token per eval walk (the walk's first vertex)
- Every step gets a 4-class label: edge valid/invalid x rule followed/violated/not-applicable
- Edge validity checked against the DCSBM adjacency matrix
- Rule compliance checked against jumper metadata (is the generated token in the target block at step encounter+r?)
- "Not-applicable" assigned to steps where no jumper rule is active
- Generation continues after rule violations (no early stopping at failure_index)
- All eval walks used as seeds (one autoregressive generation per eval walk)
- Default sequence length: 4w tokens
- Tail extension: if a jumper with rule length r is encountered at position > 4w-r, extend generation to encounter+r+1 (resolve the jump, then one more step)
- Jumper encounters detected from the generated path itself (model lands on a jumper vertex during free generation)
- SVD metrics collected for all layers (not just final layer)
- WvWo (OV circuit) computed once per model state (static weight matrices, does not change during eval)
- QK^T and AVWo computed per-step as usual
- Grassmannian distance stored as 8th metric per target (subspace rotation tracking)
- Numerical guard activations logged and counted -- summary stats stored in result.json
- token_metrics.npz: all per-step data as 2D arrays [n_sequences, n_steps]
  - SVD metrics keyed by target.layer.metric (e.g., qkt.layer_0.stable_rank)
  - Behavioral labels as int arrays (e.g., edge_valid, rule_outcome)
  - failure_index array [n_sequences] for AUROC alignment
  - NaN padding for sequences shorter than max length (due to tail extension variance)
  - Shape convention documented in schema for Phase 7 consumption
- result.json: aggregate summaries only
  - Mean/std of each metric across sequences
  - Guard activation counts per metric
  - Per-sequence failure_index list
  - Experiment-level metadata

### Claude's Discretion
- Exact batching strategy for autoregressive generation on GPU
- Internal SVD computation batching and memory management
- Specific NaN sentinel value and masking convention
- Guard activation threshold tuning (epsilon values, condition number cap at 1e6 per spec)

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | 4-class outcome per step: edge valid/invalid x rule followed/violated/not-applicable | Behavioral classifier using GraphData.adjacency CSR lookup for edges, JumperInfo map + block_assignments for rules. Three-valued rule outcome (followed/violated/not-applicable) based on active jumper tracking. |
| EVAL-02 | Edge validity check per step | CSR indptr/indices lookup (same pattern as Phase 5 evaluate.py). O(degree) per check. |
| EVAL-03 | Rule compliance check per step | Track active jumper constraints during generation: when model lands on a jumper vertex, record (encounter_step, r, target_block). At step encounter+r, check block_assignments[generated_token] == target_block. |
| EVAL-04 | failure_index annotation per sequence | First step where rule_outcome == VIOLATED. None/sentinel for fully-correct sequences. Stored as 1D array in NPZ. |
| EVAL-05 | Fused forward pass (behavioral + SVD in single inference) | Modified autoregressive loop using ExtractionMode.SVD_TARGETS: each forward pass returns logits (for generation) + internals (for SVD). Behavioral labels computed from generated tokens + graph structure. No separate inference run. |
| SVD-01 | SVD on 3 targets (QK^T, WvWo, AVWo) per layer at every step | QK^T from ForwardOutput.qkt [B, n_layers, T, T]; WvWo from model.get_wvwo() [n_layers, D, D] (static); AVWo = attention_weights @ values @ W_o.weight per layer [B, T, D]. torch.linalg.svd with full_matrices=False. |
| SVD-02 | torch.linalg.svd with full_matrices=False, batched | Verified: torch 2.10 supports batched SVD. Shape [B, T, T] -> U [B, T, T], S [B, T], Vh [B, T, T]. Also torch.linalg.svdvals for metrics needing only singular values. |
| SVD-03 | 7 scalar metrics per target: stable_rank, spectral_entropy, spectral_gap (sigma1-sigma2, sigma_k-sigma_{k+1} for k=2,4), condition_number, rank1_residual, read-write alignment (WvWo only) | Pure functions on singular values (or U, S, Vh). Each metric implemented as a standalone function. Epsilon guards on all denominators. |
| SVD-04 | Token-level time series in NPZ keyed by target.layer.metric | NPZ keys: `qkt.layer_0.stable_rank`, `wvwo.layer_0.spectral_entropy`, `avwo.layer_2.condition_number`, etc. Each value is [n_sequences, n_steps] float32 array. |
| SVD-05 | Numerical guards: NaN/Inf clamping, epsilon in entropy, condition cap at 1e6, Grassmannian distance | Pre-SVD: clamp input to finite range, check for NaN. Post-SVD: epsilon=1e-12 in all denominators, cap condition_number at 1e6, clamp entropy probabilities. Count guard activations. |
| SVD-06 | Collect SVD only for positions >= w | First w steps have incomplete context. Store NaN for positions 0..w-1, real metrics for w..max_steps. Consistent with tail-extension NaN padding at end. |
| SVD-07 | Unit tests against analytically known matrices | Identity matrix (all singular values = 1), rank-1 matrix (one nonzero singular value), diagonal matrix (known singular values), known condition numbers. Test each metric function independently. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.10+ | SVD computation, tensor operations, model forward pass | Already in project; `torch.linalg.svd` is the standard GPU-accelerated SVD |
| torch.linalg.svd | built-in | Full SVD decomposition (U, S, Vh) | Needed for rank-1 residual, Grassmannian, read-write alignment |
| torch.linalg.svdvals | built-in | Singular values only (more efficient) | Sufficient for stable_rank, entropy, gaps, condition_number |
| numpy | 2.0+ | NPZ output, NaN handling, array manipulation | Already in project; `np.savez_compressed` for token_metrics.npz |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.sparse | 1.14+ | CSR adjacency matrix operations | Already in project; edge validity lookup |
| math | stdlib | Constants (pi, log) | Entropy computation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torch.linalg.svd | torch.svd (old API) | Old API deprecated; linalg.svd is current standard |
| Full SVD for all metrics | svdvals for value-only metrics | Use svdvals where possible to avoid computing U, Vh unnecessarily; need full SVD for rank-1 residual, alignment, Grassmannian |
| Per-step SVD in Python loop | Batched SVD across steps | Batching across steps requires storing all matrices; per-step compute-and-discard is more memory efficient |

**Installation:**
```bash
# No new dependencies needed -- all libraries already in project
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── evaluation/
│   ├── __init__.py           # Public API exports
│   ├── svd_metrics.py        # 7+1 SVD metric functions (pure, testable)
│   ├── behavioral.py         # 4-class labeling, failure_index
│   └── pipeline.py           # Fused generation + collection orchestrator
```

### Pattern 1: Pure SVD Metric Functions
**What:** Each SVD metric is a standalone pure function taking singular values (or U, S, Vh) and returning a scalar tensor. Guards are built into each function.
**When to use:** All SVD metric computation (SVD-03, SVD-07).
**Example:**
```python
def stable_rank(singular_values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Stable rank: ||M||^2_F / ||M||^2_2 = sum(s_i^2) / s_1^2."""
    s_sq = singular_values ** 2
    return s_sq.sum(dim=-1) / (s_sq[..., 0] + eps)

def spectral_entropy(singular_values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Spectral entropy: -sum(p_i * log(p_i)) where p_i = sigma_i / sum(sigma)."""
    s = singular_values
    p = s / (s.sum(dim=-1, keepdim=True) + eps)
    return -(p * torch.log(p + eps)).sum(dim=-1)

def condition_number(singular_values: torch.Tensor, eps: float = 1e-12, cap: float = 1e6) -> torch.Tensor:
    """Condition number: sigma_1 / sigma_n, capped at cap."""
    raw = singular_values[..., 0] / (singular_values[..., -1] + eps)
    return torch.clamp(raw, max=cap)
```

### Pattern 2: Fused Generation Loop with SVD Extraction
**What:** Modified autoregressive loop that uses `ExtractionMode.SVD_TARGETS`, computes metrics at each step, and discards large tensors immediately.
**When to use:** The main evaluation pass (EVAL-05).
**Example:**
```python
def fused_generate_and_collect(model, start_tokens, length, graph_data, jumpers, ...):
    """Generate tokens autoregressively while collecting SVD metrics and behavioral labels."""
    model.eval()
    generated = start_tokens.to(device)  # [B, 1]
    wvwo = model.get_wvwo()  # [n_layers, D, D] -- static, compute once

    # Pre-compute WvWo SVD metrics (static across all steps)
    wvwo_metrics = compute_all_metrics(wvwo)  # dict of {metric_name: [n_layers]}

    all_step_metrics = []  # list of per-step metric dicts
    all_labels = []        # list of per-step behavioral label dicts

    with torch.no_grad():
        for step in range(length - 1):
            context = generated[:, -max_seq_len:]
            output = model(context, mode=ExtractionMode.SVD_TARGETS)

            # Next token via argmax
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Behavioral labels
            labels = classify_step(generated, step, graph_data, jumpers)
            all_labels.append(labels)

            # SVD metrics (skip warmup positions < w)
            if step >= w:
                # QK^T SVD
                qkt_metrics = compute_qkt_metrics(output.qkt)  # per layer
                # AVWo SVD
                avwo = compute_avwo(output.attention_weights, output.values, model)
                avwo_metrics = compute_avwo_metrics(avwo)
                all_step_metrics.append({
                    'qkt': qkt_metrics,
                    'avwo': avwo_metrics,
                    'wvwo': wvwo_metrics,  # broadcast static values
                })
            else:
                all_step_metrics.append(None)  # NaN placeholder

    return generated, all_step_metrics, all_labels
```

### Pattern 3: Active Jumper Constraint Tracking
**What:** Track which jumper rules are active during generation by monitoring when the model lands on jumper vertices.
**When to use:** Rule compliance classification (EVAL-01, EVAL-03).
**Example:**
```python
def track_jumper_encounters(generated_seq, jumper_map):
    """Track active constraints from jumper encounters in generated path."""
    active_constraints = []  # list of (encounter_step, deadline_step, target_block)

    for t in range(len(generated_seq)):
        vertex = int(generated_seq[t])
        if vertex in jumper_map:
            j = jumper_map[vertex]
            active_constraints.append((t, t + j.r, j.target_block))

    return active_constraints

def classify_step(seq, step, graph_data, active_constraints):
    """Classify step into 4 classes: edge_valid x rule_outcome."""
    # Edge validity
    u, v = int(seq[step]), int(seq[step + 1])
    neighbors = graph_data.adjacency.indices[
        graph_data.adjacency.indptr[u]:graph_data.adjacency.indptr[u + 1]
    ]
    edge_valid = v in neighbors

    # Rule outcome: check if any constraint has deadline at step+1
    rule_outcome = RuleOutcome.NOT_APPLICABLE
    for enc_step, deadline, target_block in active_constraints:
        if step + 1 == deadline:
            actual_block = int(graph_data.block_assignments[v])
            rule_outcome = (RuleOutcome.FOLLOWED if actual_block == target_block
                           else RuleOutcome.VIOLATED)
            break

    return edge_valid, rule_outcome
```

### Pattern 4: Compute-and-Discard Memory Strategy
**What:** At each generation step, compute SVD metrics from extracted tensors and immediately store only the scalar results, discarding the large intermediate tensors.
**When to use:** All SVD collection steps to manage GPU memory.
**Example:**
```python
# At each step, output.qkt is [B, n_layers, T, T] -- could be [32, 4, 64, 64] = 2MB
# Compute metrics immediately:
for layer in range(n_layers):
    qkt_layer = output.qkt[:, layer]  # [B, T, T]
    U, S, Vh = torch.linalg.svd(qkt_layer, full_matrices=False)
    metrics = compute_metrics_from_svd(U, S, Vh)
    # Store only scalar metrics (8 values per layer per target)
    step_metrics[f'qkt.layer_{layer}'] = metrics
    # U, S, Vh, qkt_layer go out of scope and are freed
```

### Anti-Patterns to Avoid
- **Storing all QK^T matrices across steps:** With 256 steps x 4 layers x [B, 64, 64], this would consume ~2GB per batch element. Compute metrics per-step and discard.
- **Separate generation and SVD passes:** Violates EVAL-05. Must extract internals during the same forward passes used for generation.
- **Computing WvWo SVD at every step:** WvWo is static (weight matrices don't change during eval). Compute once, broadcast metrics.
- **Ignoring NaN from torch.linalg.svd:** SVD throws LinAlgError on NaN input. Must clamp/check inputs before SVD, not after.
- **Using torch.svd instead of torch.linalg.svd:** Old API is deprecated and has different return conventions (V vs Vh).
- **Forgetting to detach before SVD:** The model already detaches in ExtractionMode.SVD_TARGETS, but AVWo computation from attention_weights @ values needs explicit no_grad context.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SVD computation | Custom eigendecomposition | `torch.linalg.svd(full_matrices=False)` | GPU-accelerated, numerically stable, handles batching |
| Singular values only | Full SVD then discard U, Vh | `torch.linalg.svdvals()` | ~2x faster, less memory |
| NaN checking | Manual element-wise comparison | `torch.isnan().any()` / `torch.isfinite()` | Vectorized, handles edge cases |
| Matrix Frobenius norm | Manual sum of squares | `torch.linalg.norm(M, 'fro')` | Numerically stable accumulation |
| CSR edge lookup | Dense adjacency matrix | `scipy.sparse.csr_matrix` indptr/indices | Already in codebase, O(degree) lookup |
| Compressed NPZ output | Custom binary format | `np.savez_compressed()` | Standard, portable, already used in schema.py |

**Key insight:** SVD metric functions should be pure functions on tensors (singular values or U/S/Vh). This makes them unit-testable against analytically known matrices (SVD-07) and composable across different SVD targets.

## Common Pitfalls

### Pitfall 1: torch.linalg.svd Throws on NaN Input
**What goes wrong:** If any element of the input matrix is NaN or Inf, `torch.linalg.svd` raises `LinAlgError` instead of returning NaN singular values.
**Why it happens:** LAPACK/cuSOLVER SVD algorithms cannot handle non-finite inputs.
**How to avoid:** Pre-SVD guard: `torch.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)`. Count and log guard activations. This is the critical numerical guard from SVD-05.
**Warning signs:** `LinAlgError: The algorithm failed to converge because the input matrix contained non-finite values.`

### Pitfall 2: Spectral Entropy Log-Domain Underflow
**What goes wrong:** When a singular value is 0, `p * log(p)` becomes `0 * -inf = NaN`. When using `log(p + eps)` with very small eps, the entropy can become slightly negative.
**Why it happens:** The entropy formula requires `p_i > 0` for all i, but near-zero singular values produce near-zero probabilities.
**How to avoid:** Use `eps=1e-12` in both the normalization denominator and the log argument: `p = s / (s.sum() + eps)`, `entropy = -(p * torch.log(p + eps)).sum()`. Clamp the final result to be >= 0.
**Warning signs:** Negative entropy values or NaN in entropy arrays.

### Pitfall 3: Condition Number Explosion
**What goes wrong:** When the smallest singular value is near zero, `sigma_1 / sigma_n` can be 1e15+ or Inf. Downstream analysis may not handle extreme values.
**Why it happens:** Rank-deficient or near-singular matrices are common in attention (especially early positions with limited context).
**How to avoid:** Cap at 1e6 per spec: `torch.clamp(raw_cond, max=1e6)`. Log guard activations when capping occurs.
**Warning signs:** Condition numbers at exactly 1e6 in the data (indicates capping).

### Pitfall 4: Context Window Alignment During Generation
**What goes wrong:** During autoregressive generation, the model only sees the last `max_seq_len` (=w) tokens. If the SVD metrics are indexed by generation step but the QK^T matrix shifts as the context window slides, the metrics from step t don't correspond to position t in the sequence.
**Why it happens:** Confusion between sequence position and context-window-relative position.
**How to avoid:** Always index metrics by generation step (absolute position in the generated sequence). The QK^T matrix at step t reflects the model's state when predicting token t+1, regardless of which tokens are in the context window. SVD-06's warmup skip (positions < w) handles the initial period where the context isn't full.
**Warning signs:** Metrics that show a discontinuity exactly at position w (where the window starts sliding).

### Pitfall 5: Tail Extension Creates Ragged Sequences
**What goes wrong:** When different sequences have different lengths (due to tail extension), stacking them into a 2D array requires padding.
**Why it happens:** Jumper encounters at positions > 4w-r trigger extension to encounter+r+1. Different sequences may encounter jumpers at different positions (or not at all).
**How to avoid:** Pre-compute the maximum possible length (4w + max_r + 1 where max_r = round(2.0 * w)). Allocate 2D arrays at this maximum size. Fill with NaN for positions beyond each sequence's actual length. Document the shape convention in the NPZ schema.
**Warning signs:** IndexError when trying to write beyond array bounds; misaligned failure_index lookups in Phase 7.

### Pitfall 6: AVWo Computation Uses Wrong Weight Convention
**What goes wrong:** Computing `A @ V @ W_o` using the wrong weight matrix orientation produces incorrect SVD targets.
**Why it happens:** `nn.Linear` stores weights as `[out_features, in_features]`, but the mathematical operation is `x @ W^T`. The existing codebase computes `W_v.weight.T @ W_o.weight` for WvWo.
**How to avoid:** AVWo = `attention_weights @ values @ W_o.weight` where `W_o.weight` is `[d_model, d_model]`. The values tensor is already the output of `W_v(x)`, so it's in the correct space. The W_o linear layer computes `values @ W_o.weight.T` internally, but for the SVD target we want the matrix `A @ V` (in value space) projected through `W_o`. Use `(A @ V) @ block.attention.W_o.weight` to get `[B, T, D]`.
**Warning signs:** AVWo metrics that are identical to or strongly correlated with QK^T metrics (they should differ).

### Pitfall 7: Grassmannian Distance Requires Consecutive Step Tracking
**What goes wrong:** Grassmannian distance measures subspace rotation between consecutive steps, but if implemented naively, the "previous step" subspace isn't available.
**Why it happens:** Compute-and-discard pattern frees tensors at each step.
**How to avoid:** Retain only the top-k singular vectors (U[:, :k]) from the previous step for Grassmannian computation. This is a small tensor ([B, T, k] or [B, D, k]). After computing Grassmannian, update the cached subspace to the current step's.
**Warning signs:** Grassmannian distance of 0 at all steps (forgot to update the cached subspace) or NaN at step 0 (no previous subspace exists).

### Pitfall 8: Batched Generation Memory Growth
**What goes wrong:** With large batch sizes, the generated tensor grows linearly with sequence length, and storing all extraction outputs overwhelms GPU memory.
**Why it happens:** `torch.cat` for the generated tensor keeps all history on GPU.
**How to avoid:** Keep `generated` on GPU for the forward pass context window, but periodically move completed portions to CPU. For SVD extraction, compute metrics immediately and only store scalars. Process eval walks in smaller batches if needed (8-32 sequences at a time).
**Warning signs:** CUDA OOM errors during evaluation.

## Code Examples

### SVD Metric Suite (Complete Implementation Pattern)
```python
import torch

EPS = 1e-12
CONDITION_CAP = 1e6

def stable_rank(S: torch.Tensor) -> torch.Tensor:
    """||M||^2_F / ||M||^2_2 = sum(s_i^2) / s_1^2."""
    s_sq = S ** 2
    return s_sq.sum(dim=-1) / (s_sq[..., 0] + EPS)

def spectral_entropy(S: torch.Tensor) -> torch.Tensor:
    """-sum(p_i * log(p_i)) where p_i = sigma_i / sum(sigma)."""
    p = S / (S.sum(dim=-1, keepdim=True) + EPS)
    ent = -(p * torch.log(p + EPS)).sum(dim=-1)
    return torch.clamp(ent, min=0.0)

def spectral_gap_1_2(S: torch.Tensor) -> torch.Tensor:
    """sigma_1 - sigma_2."""
    return S[..., 0] - S[..., 1]

def spectral_gap_2_3(S: torch.Tensor) -> torch.Tensor:
    """sigma_2 - sigma_3 (generalized gap at k=2)."""
    return S[..., 1] - S[..., 2]

def spectral_gap_4_5(S: torch.Tensor) -> torch.Tensor:
    """sigma_4 - sigma_5 (generalized gap at k=4)."""
    return S[..., 3] - S[..., 4]

def condition_number(S: torch.Tensor) -> torch.Tensor:
    """sigma_1 / sigma_n, capped at 1e6."""
    raw = S[..., 0] / (S[..., -1] + EPS)
    return torch.clamp(raw, max=CONDITION_CAP)

def rank1_residual_norm(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
    """||M - sigma_1 * u_1 * v_1^T||_F / ||M||_F."""
    # Frobenius norm of M: sqrt(sum(s_i^2))
    fro_M = torch.sqrt((S ** 2).sum(dim=-1) + EPS)
    # Frobenius norm of residual: sqrt(sum(s_i^2 for i>=1))
    fro_residual = torch.sqrt((S[..., 1:] ** 2).sum(dim=-1) + EPS)
    return fro_residual / (fro_M + EPS)

def read_write_alignment(U: torch.Tensor, Vh: torch.Tensor) -> torch.Tensor:
    """Cosine angle between top left and right singular vectors (WvWo only)."""
    u1 = U[..., :, 0]   # top left singular vector
    v1 = Vh[..., 0, :]   # top right singular vector
    dot = torch.sum(u1 * v1, dim=-1)
    return torch.abs(dot)  # cosine similarity (absolute value)

def grassmannian_distance(
    U_prev: torch.Tensor, U_curr: torch.Tensor, k: int = 2
) -> torch.Tensor:
    """Grassmannian distance between k-dimensional subspaces."""
    # Principal angles via SVD of U_prev^T @ U_curr
    cos_angles = torch.linalg.svdvals(U_prev[..., :, :k].transpose(-2, -1) @ U_curr[..., :, :k])
    cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
    angles = torch.arccos(cos_angles)
    return torch.sqrt((angles ** 2).sum(dim=-1))
```

### Pre-SVD Numerical Guard
```python
def guard_matrix_for_svd(M: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Clamp non-finite values before SVD. Returns (cleaned_matrix, guard_activated)."""
    has_nonfinite = not torch.isfinite(M).all()
    if has_nonfinite:
        M = torch.nan_to_num(M, nan=0.0, posinf=1e6, neginf=-1e6)
    return M, has_nonfinite
```

### AVWo Computation
```python
def compute_avwo(
    attention_weights: torch.Tensor,  # [B, n_layers, T, T]
    values: torch.Tensor,             # [B, n_layers, T, D]
    model: nn.Module,
) -> torch.Tensor:
    """Compute AVWo (net residual update) for all layers.

    Returns [B, n_layers, T, D] -- the attention-weighted value vectors
    projected through W_o.
    """
    B, n_layers, T, D = values.shape
    avwo_layers = []
    for layer_idx in range(n_layers):
        A = attention_weights[:, layer_idx]  # [B, T, T]
        V = values[:, layer_idx]              # [B, T, D]
        AV = A @ V                            # [B, T, D]
        # W_o.weight is [D, D] (nn.Linear convention: [out, in])
        Wo = model.blocks[layer_idx].attention.W_o.weight  # [D, D]
        # AV @ Wo^T would give the actual output, but for SVD target
        # we want the matrix that gets projected, so: AV @ Wo
        # Actually: nn.Linear computes x @ W^T, so the output is AV @ Wo.T
        # But for the SVD target of the net residual update, we want the
        # full transformation: the [T, D] matrix that represents the update
        avwo = AV @ Wo.T  # [B, T, D] -- same as what W_o would output
        avwo_layers.append(avwo)
    return torch.stack(avwo_layers, dim=1)  # [B, n_layers, T, D]
```

### Behavioral Classification
```python
from enum import IntEnum

class RuleOutcome(IntEnum):
    NOT_APPLICABLE = 0
    FOLLOWED = 1
    VIOLATED = 2

def classify_steps(
    generated: torch.Tensor,       # [B, L]
    graph_data: GraphData,
    jumper_map: dict[int, JumperInfo],
    w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify all steps in all sequences.

    Returns:
        edge_valid: bool array [B, L-1]
        rule_outcome: int array [B, L-1] (RuleOutcome values)
        failure_index: int array [B] (-1 for no failure)
    """
    seqs = generated.cpu().numpy()
    B, L = seqs.shape
    edge_valid = np.zeros((B, L - 1), dtype=bool)
    rule_outcome = np.full((B, L - 1), RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
    failure_index = np.full(B, -1, dtype=np.int32)

    indptr = graph_data.adjacency.indptr
    indices = graph_data.adjacency.indices
    block_assignments = graph_data.block_assignments

    for b in range(B):
        active_constraints = []  # (deadline_step, target_block)

        for t in range(L - 1):
            u = int(seqs[b, t])
            v = int(seqs[b, t + 1])

            # Edge validity
            neighbors = indices[indptr[u]:indptr[u + 1]]
            edge_valid[b, t] = v in neighbors

            # Track jumper encounters
            if u in jumper_map:
                j = jumper_map[u]
                active_constraints.append((t + j.r, j.target_block))

            # Check rule deadlines
            for deadline, target_block in active_constraints:
                if t + 1 == deadline:
                    actual_block = int(block_assignments[v])
                    if actual_block == target_block:
                        rule_outcome[b, t] = RuleOutcome.FOLLOWED
                    else:
                        rule_outcome[b, t] = RuleOutcome.VIOLATED
                        if failure_index[b] == -1:
                            failure_index[b] = t + 1
                    break  # only one constraint can resolve per step

    return edge_valid, rule_outcome, failure_index
```

### Token Metrics NPZ Schema
```python
def save_token_metrics(
    metrics: dict[str, np.ndarray],
    behavioral: dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Save all per-step metrics to compressed NPZ.

    Keys follow the convention:
        SVD: {target}.layer_{i}.{metric_name}  e.g., 'qkt.layer_0.stable_rank'
        Behavioral: edge_valid, rule_outcome, failure_index
    """
    all_arrays = {}
    all_arrays.update(metrics)       # SVD metric arrays [n_sequences, n_steps]
    all_arrays.update(behavioral)    # behavioral label arrays
    np.savez_compressed(output_path, **all_arrays)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.svd()` | `torch.linalg.svd()` | PyTorch 1.9+ | New API returns Vh (not V), consistent with NumPy/SciPy convention |
| Manual SVD loops | Batched `torch.linalg.svd` | PyTorch 2.0+ | GPU-accelerated batched SVD, significant speedup |
| `torch.symeig` for symmetric matrices | `torch.linalg.eigh` | PyTorch 1.9+ | Better numerical stability |
| Manual singular values | `torch.linalg.svdvals` | PyTorch 1.9+ | More efficient when U, Vh not needed |

**Deprecated/outdated:**
- `torch.svd()`: Deprecated in favor of `torch.linalg.svd()`. Old API returns V (not Vh), easy to confuse.
- `torch.eig()`: Deprecated in favor of `torch.linalg.eig()`.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_evaluation.py -x` |
| Full suite command | `pytest tests/ -x` |
| Estimated runtime | ~30 seconds (SVD computation on CPU for small matrices) |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | 4-class behavioral labeling | unit | `pytest tests/test_behavioral.py -x` | No - Wave 0 gap |
| EVAL-02 | Edge validity per step | unit | `pytest tests/test_behavioral.py::TestEdgeValidity -x` | No - Wave 0 gap |
| EVAL-03 | Rule compliance per step | unit | `pytest tests/test_behavioral.py::TestRuleCompliance -x` | No - Wave 0 gap |
| EVAL-04 | failure_index annotation | unit | `pytest tests/test_behavioral.py::TestFailureIndex -x` | No - Wave 0 gap |
| EVAL-05 | Fused forward pass | integration | `pytest tests/test_evaluation_pipeline.py::TestFusedPass -x` | No - Wave 0 gap |
| SVD-01 | SVD on 3 targets per layer | unit | `pytest tests/test_svd_metrics.py::TestSVDTargets -x` | No - Wave 0 gap |
| SVD-02 | torch.linalg.svd batched | unit | `pytest tests/test_svd_metrics.py::TestBatchedSVD -x` | No - Wave 0 gap |
| SVD-03 | 7 scalar metrics | unit | `pytest tests/test_svd_metrics.py::TestMetricFunctions -x` | No - Wave 0 gap |
| SVD-04 | NPZ output format | unit | `pytest tests/test_evaluation_pipeline.py::TestNPZOutput -x` | No - Wave 0 gap |
| SVD-05 | Numerical guards | unit | `pytest tests/test_svd_metrics.py::TestNumericalGuards -x` | No - Wave 0 gap |
| SVD-06 | Skip positions < w | integration | `pytest tests/test_evaluation_pipeline.py::TestWarmupSkip -x` | No - Wave 0 gap |
| SVD-07 | Unit tests against known matrices | unit | `pytest tests/test_svd_metrics.py::TestAnalyticalMatrices -x` | No - Wave 0 gap |

### Nyquist Sampling Rate
- **Minimum sample interval:** After every committed task -> run: `pytest tests/test_svd_metrics.py tests/test_behavioral.py -x`
- **Full suite trigger:** Before merging final task of any plan wave
- **Phase-complete gate:** Full suite green before verification
- **Estimated feedback latency per task:** ~30 seconds

### Wave 0 Gaps (must be created before implementation)
- [ ] `tests/test_svd_metrics.py` -- covers SVD-01, SVD-02, SVD-03, SVD-05, SVD-07
- [ ] `tests/test_behavioral.py` -- covers EVAL-01, EVAL-02, EVAL-03, EVAL-04
- [ ] `tests/test_evaluation_pipeline.py` -- covers EVAL-05, SVD-04, SVD-06
- [ ] `src/evaluation/__init__.py` -- package init

## Open Questions

1. **AVWo Weight Convention Clarification**
   - What we know: The model computes `W_o(attention_weights @ v)` which internally does `(A @ V) @ W_o.weight.T`. The existing `get_wvwo()` computes `W_v.weight.T @ W_o.weight`. For AVWo as an SVD target, we want the [T, D] matrix representing the net residual update per position.
   - What's unclear: Whether AVWo should use `(A @ V) @ W_o.weight` (keeping in nn.Linear's storage convention) or `(A @ V) @ W_o.weight.T` (matching the actual computation). The former gives the product in weight-storage space, the latter gives the actual output.
   - Recommendation: Use `(A @ V) @ W_o.weight.T` since this matches what actually gets added to the residual stream. The model's forward pass computes `self.W_o(att_weights @ v)` which is `(A @ V) @ W_o.weight.T + bias` (no bias in this model). This is the physically meaningful matrix.

2. **Grassmannian Subspace Dimension k**
   - What we know: Grassmannian distance compares k-dimensional subspaces (top-k singular vectors) between consecutive steps. Larger k captures more of the subspace but is noisier.
   - What's unclear: What value of k to use (1, 2, 4?).
   - Recommendation: Use k=2 as default (captures the dominant 2D subspace). This is Claude's discretion. Make k a parameter with default=2 for flexibility.

3. **Batch Size for Evaluation Generation**
   - What we know: All eval walks are used as seeds. For anchor config, eval corpus has many walks. Autoregressive generation with SVD extraction is memory-intensive.
   - What's unclear: Optimal batch size balancing GPU utilization vs memory.
   - Recommendation: Start with batch_size=32, profile on anchor config. Fall back to 8 if OOM. This is Claude's discretion per CONTEXT.md.

## Sources

### Primary (HIGH confidence)
- PyTorch 2.10 documentation: `torch.linalg.svd`, `torch.linalg.svdvals`, `torch.linalg.norm`
- Existing codebase: `src/model/transformer.py` (ExtractionMode, get_wvwo), `src/training/evaluate.py` (greedy_generate pattern), `src/graph/types.py` (GraphData), `src/graph/jumpers.py` (JumperInfo), `src/results/schema.py` (write_result with token_metrics)
- Verified via live testing: `torch.linalg.svd` throws `LinAlgError` on NaN input (not silent), `torch.linalg.svdvals` returns identical values to full SVD's S component, batched SVD works on 4D tensors

### Secondary (MEDIUM confidence)
- SVD metric definitions from numerical linear algebra: stable rank, spectral entropy, condition number are standard definitions
- Grassmannian distance via principal angles is the standard geodesic distance on the Grassmann manifold

### Tertiary (LOW confidence)
- Optimal Grassmannian subspace dimension k=2: reasonable default but may need tuning based on actual attention matrix rank structure

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all PyTorch built-ins already in project, verified via live testing
- Architecture: HIGH - builds directly on existing codebase patterns (ExtractionMode, greedy_generate, write_result)
- SVD metrics: HIGH - standard numerical linear algebra, verified formulas and edge cases
- Behavioral classification: HIGH - extends existing compliance evaluation pattern from Phase 5
- Numerical guards: HIGH - verified torch.linalg.svd NaN behavior via live testing
- Pitfalls: HIGH - verified critical pitfall (SVD NaN throws) experimentally

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (stable domain, no fast-moving APIs)
