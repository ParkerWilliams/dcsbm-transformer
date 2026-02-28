# Phase 16: Multi-Head Ablation - Research

**Researched:** 2026-02-26
**Scope:** Extending single-head transformer to multi-head (1h/2h/4h) with per-head SVD extraction and signal concentration analysis

## Codebase Analysis

### Files requiring modification (HIGH confidence -- direct code reading)

| File | Current State | Required Changes | Risk |
|------|--------------|-----------------|------|
| `src/config/experiment.py` | `n_heads != 1` check at L85-86, ModelConfig has `n_heads: int = 1` | Relax constraint to `n_heads in (1,2,4)`, add d_model divisibility check | MEDIUM |
| `src/model/attention.py` | Single-head: Q,K,V are [B,T,D], no reshape, W_q/W_k/W_v/W_o are [D,D] | Add n_heads parameter, reshape to [B,H,T,d_head], per-head QK^T extraction | HIGH |
| `src/model/block.py` | TransformerBlock.__init__ takes (d_model, max_seq_len, dropout) | Add n_heads parameter, pass through to CausalSelfAttention | LOW |
| `src/model/transformer.py` | TransformerLM.__init__ takes (vocab_size, d_model, n_layers, max_seq_len, dropout), get_wvwo() returns [n_layers, D, D] | Add n_heads, pass through, update stacking to include head dim, update get_wvwo() for per-head OV circuit | MEDIUM |
| `src/model/types.py` | AttentionInternals: qkt [B,T,T], ForwardOutput: qkt [B,n_layers,T,T] | Add head dimension to all attention tensors | MEDIUM |
| `src/evaluation/pipeline.py` | fused_evaluate loops over layers, indexes qkt as [B, T, T] per layer | Add inner loop over heads, per-head SVD, dual key emission, per-head spectrum storage | HIGH |
| `src/analysis/auroc_horizon.py` | PRIMARY_METRICS, _is_primary_metric parses `target.layer_N.metric_name` | Parse head index from keys, support per-head AUROC computation | MEDIUM |

### New files to create

| File | Purpose |
|------|---------|
| `src/analysis/signal_concentration.py` | Entropy/Gini of per-head AUROC distribution, signal concentration analysis |
| `tests/test_multi_head.py` | Multi-head attention tests: shape validation, per-head extraction, backward compatibility |
| `tests/test_signal_concentration.py` | Signal concentration metric tests |

### Key architectural constraints

1. **d_k constant at 128**: d_model = n_heads * d_k. For 1h: d_model=128, 2h: d_model=256, 4h: d_model=512. This is set in the ROADMAP success criteria.

2. **Per-head QK^T rank**: QK^T is [T, T] regardless of d_k, but effective rank is bounded by d_k. Since d_k=128 for all configs, the rank constraint does not change across head counts. SVD metrics retain the same dynamic range.

3. **W_o structure**: In standard multi-head attention, W_o is [d_model, d_model]. The concatenated head outputs are [B, T, d_model], and W_o projects back. Per-head W_o slice: W_o[h*d_head:(h+1)*d_head, :] gives [d_head, d_model].

4. **ForwardOutput head dimension**: The stacking `torch.stack(all_qkt, dim=1)` currently gives [B, n_layers, T, T]. With multi-head, each layer's qkt is [B, n_heads, T, T], so stacking gives [B, n_layers, n_heads, T, T].

5. **Backward compatibility**: When n_heads=1, the n_heads dimension is kept (not squeezed). This means all downstream code must handle the head dimension explicitly, but the single-head case is just H=1.

### Evaluation pipeline impact analysis

The `fused_evaluate` function in `src/evaluation/pipeline.py` has several loops that must change:

1. **WvWo pre-computation** (L157): Must iterate over heads within each layer
2. **SVD metric key allocation** (L177-183): Keys must include head index
3. **QK^T SVD collection** (L260-316): Must index into head dimension
4. **AVWo computation** (L318-364): Must compute per-head AVWo
5. **WvWo metrics broadcast** (L367-378): Must include head index
6. **Grassmannian tracking** (L220-225): Must track per (target, layer, head) tuple
7. **Spectrum storage** (L186-195): Must include head dimension

### _compute_avwo_for_layer changes

Currently computes AVWo = (A @ V) @ W_o.T for the full attention output. For multi-head:
- A is per-head [B, T, T], V is per-head [B, T, d_head]
- AV = A @ V gives [B, T, d_head] per head
- W_o slice: W_o.weight[h*d_head:(h+1)*d_head, :] gives [d_head, d_model]
- Per-head AVWo = AV @ W_o_slice.T gives [B, T, d_model]

Alternatively, since per-head V and A already encapsulate the head's contribution:
- Per-head residual update: (A_h @ V_h) @ W_o_h.T
- This gives [B, T, d_model] per head (contribution of head h to the residual stream)

### Signal concentration metrics

From FEATURES.md research:
- **Entropy**: H = -sum(p_h * log(p_h)) where p_h = AUROC_h / sum(AUROC)
- **Gini coefficient**: Measures inequality of AUROC distribution across heads
- **Max-to-mean ratio**: Simple interpretable metric: max(AUROC_h) / mean(AUROC_h)
- From PITFALLS.md Pitfall 13: Frame as descriptive statistics, not hypothesis test. Report per-head AUROC with confidence intervals.

### Testing strategy

1. **Unit tests for multi-head attention**: Verify output shapes [B, n_heads, T, T] for QK^T, correct scaling by 1/sqrt(d_head), causal mask application per head
2. **Integration test**: Run 1h and 2h models on same input, verify per-head extraction produces distinct matrices (heads should specialize)
3. **Backward compatibility**: Run single-head model, verify dual key emission (both legacy and v1.1 format)
4. **Shape regression**: Verify ForwardOutput shapes match expected [B, n_layers, n_heads, T, T]
5. **Signal concentration**: Test entropy/Gini on known distributions (uniform -> max entropy, degenerate -> min entropy)

## Sources

- Existing codebase: `src/model/attention.py`, `src/model/transformer.py`, `src/model/types.py`, `src/model/block.py` (HIGH confidence -- direct reading)
- `.planning/research/ARCHITECTURE.md` Feature 3 section (HIGH confidence -- detailed implementation plan)
- `.planning/research/PITFALLS.md` Pitfall 2, 5, 13 (HIGH confidence -- known risks)
- `.planning/research/FEATURES.md` Feature 3 section (HIGH confidence -- feature landscape)
- `.planning/ROADMAP.md` Phase 16 requirements and success criteria (HIGH confidence)
- `.planning/REQUIREMENTS.md` MHAD-01 through MHAD-04 (HIGH confidence)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Attention module rewrite | HIGH | Standard multi-head attention, well-understood |
| Tensor shape changes | MEDIUM | Many consumers must update; testing critical |
| NPZ dual key emission | HIGH | Simple conditional in pipeline loop |
| WvWo per-head computation | MEDIUM | Rectangular matrix SVD needs careful handling |
| Signal concentration metrics | HIGH | Standard statistical measures |
| Backward compatibility | MEDIUM | Dual emission strategy sound but adds code paths |
| Pipeline SVD loop changes | MEDIUM | Complex nested loops, risk of index errors |
