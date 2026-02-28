# Phase 16: Multi-Head Ablation - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Extend the transformer to support multi-head attention (1h/2h/4h) with per-head SVD extraction and signal concentration analysis. The d_k dimension is held constant at 128 (d_model scales as n_heads * d_k). Per-head QK^T matrices are extracted for SVD analysis. Per-head AUROC and signal concentration metrics (entropy, Gini coefficient) determine whether predictive signal concentrates in specific heads or distributes across all heads. An ablation comparison runs on matched configs (same graph, same walks, only n_heads varies). This is the most invasive phase -- it touches the model, config, evaluation pipeline, and analysis layers. Spectrum trajectory, compliance curves, and other advanced analyses are handled in Phase 15.

</domain>

<decisions>
## Implementation Decisions

### Architecture: d_k constant, d_model scales
- d_k = 128 held constant across all head counts
- 1h: d_model=128, 2h: d_model=256, 4h: d_model=512
- This ensures equal per-head dimensionality across ablation configs
- Larger d_model means more parameters -- this is intentional to isolate the multi-head splitting effect

### Attention module: Modify CausalSelfAttention in-place
- Modify the existing CausalSelfAttention class to support n_heads parameter
- When n_heads=1, behavior is identical to v1.0 (backward compatible)
- Q, K, V projections go from [B, T, d_model] to [B, n_heads, T, d_head] internally
- W_o projects concatenated heads [B, T, d_model] back to residual stream
- Per-head QK^T extraction: [B, n_heads, T, T] stored in AttentionInternals

### Tensor shape changes (head dimension)
- AttentionInternals.qkt: [B, T, T] -> [B, n_heads, T, T]
- AttentionInternals.attention_weights: [B, T, T] -> [B, n_heads, T, T]
- AttentionInternals.values: [B, T, D] -> [B, n_heads, T, d_head]
- ForwardOutput.qkt: [B, n_layers, T, T] -> [B, n_layers, n_heads, T, T]
- ForwardOutput.attention_weights: [B, n_layers, T, T] -> [B, n_layers, n_heads, T, T]
- ForwardOutput.values: [B, n_layers, T, D] -> [B, n_layers, n_heads, T, d_head]
- For single-head (n_heads=1), shapes still include the head dimension (n_heads=1 not squeezed)

### NPZ key format and backward compatibility
- Multi-head keys: `target.layer_N.head_H.metric_name` (e.g., `qkt.layer_0.head_0.grassmannian_distance`)
- Single-head runs emit BOTH formats:
  - Legacy v1.0: `qkt.layer_0.grassmannian_distance`
  - New v1.1: `qkt.layer_0.head_0.grassmannian_distance`
- Multi-head runs emit only v1.1 format (no sensible v1.0 equivalent)

### WvWo per-head computation
- Per-head OV circuit: W_v[:, h*d_head:(h+1)*d_head].T @ W_o[h*d_head:(h+1)*d_head, :]
- Yields [d_head, d_model] per head (rectangular, not square)
- SVD metrics that require square matrices are skipped for multi-head WvWo or computed on the rectangular matrix
- get_wvwo() returns [n_layers, n_heads, d_head, d_model] for multi-head

### Signal concentration analysis
- Per-head AUROC computed for each metric at each lookback distance
- Entropy of per-head AUROC distribution: high entropy = signal distributed, low entropy = concentrated
- Gini coefficient: complement measure of concentration
- Report as descriptive statistics with confidence intervals, not hypothesis tests
- Keep primary metrics at head-aggregated level; per-head analysis is exploratory

### Config validation
- Remove the n_heads != 1 constraint in ExperimentConfig.__post_init__
- Add: n_heads must be in (1, 2, 4)
- Add: d_model must be divisible by n_heads
- Add: d_model // n_heads >= 16 (minimum per-head dimension)

### Claude's Discretion
- Exact implementation of the multi-head reshape operations in attention.py
- How to batch SVD computation across heads (loop vs. batch)
- Signal concentration visualization choices
- Test fixture design for multi-head variants
- Whether to compute per-head AVWo or only per-head QK^T metrics

</decisions>

<specifics>
## Specific Ideas

- The ablation should use the SAME graph and SAME walks for all head counts -- only the model architecture changes
- Per-head Grassmannian distance should be the key signal metric (primary hypothesis is about Grassmannian distance)
- If signal concentrates in one head, this validates the single-head architecture; if distributed, multi-head adds robustness
- The rank of per-head QK^T is bounded by d_k (=128 for all configs), so metrics like stable rank retain the same dynamic range
- Create a separate signal_concentration.py module in src/analysis/ for the entropy/Gini computation

</specifics>

<deferred>
## Deferred Ideas

- Multi-head beyond 4 heads (out of scope per REQUIREMENTS.md)
- Ablation with d_k varying (holding d_model constant) -- different experimental design, future work
- Per-head visualization overlays in the HTML report -- could be Phase 17 if needed
- Automated head pruning based on signal concentration -- research direction, not v1.1

</deferred>

---

*Phase: 16-multi-head-ablation*
*Context gathered: 2026-02-26*
