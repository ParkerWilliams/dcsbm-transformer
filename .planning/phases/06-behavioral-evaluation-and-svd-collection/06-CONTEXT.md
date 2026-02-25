# Phase 6: Behavioral Evaluation and SVD Collection - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

A single fused evaluation pass through autoregressive-generated sequences that produces 4-class behavioral labels at every step and SVD metrics across three targets (QK^T, WvWo, AVWo) with numerical stability guarantees. The model generates freely from a single start token; labels are checked against the DCSBM graph structure directly. Predictive horizon analysis and AUROC computation are Phase 7.

</domain>

<decisions>
## Implementation Decisions

### Evaluation semantics
- Autoregressive free-generation: model generates each token from its own previous output
- Seed: single start token per eval walk (the walk's first vertex)
- Every step gets a 4-class label: edge valid/invalid x rule followed/violated/not-applicable
- Edge validity checked against the DCSBM adjacency matrix
- Rule compliance checked against jumper metadata (is the generated token in the target block at step encounter+r?)
- "Not-applicable" assigned to steps where no jumper rule is active

### Sequence handling
- Generation continues after rule violations (no early stopping at failure_index)
- All eval walks used as seeds (one autoregressive generation per eval walk)
- Default sequence length: 4w tokens
- Tail extension: if a jumper with rule length r is encountered at position > 4w-r, extend generation to encounter+r+1 (resolve the jump, then one more step)
- Jumper encounters detected from the generated path itself (model lands on a jumper vertex during free generation)

### SVD collection scope
- SVD metrics collected for all layers (not just final layer)
- WvWo (OV circuit) computed once per model state (static weight matrices, does not change during eval)
- QK^T and AVWo computed per-step as usual
- Grassmannian distance stored as 8th metric per target (subspace rotation tracking)
- Numerical guard activations logged and counted — summary stats stored in result.json

### Output organization
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

</decisions>

<specifics>
## Specific Ideas

- "We only need to generate enough walks that a sufficient number of jumper encounters are observed" — the all-eval-walks approach ensures statistical power without explicit minimum thresholds
- Walk length is a configuration-level constant (2w, 4w, 8w are sweep parameters), so 2D arrays are clean within a config — no ragged sequences except from tail extension
- AUROC at lookback j uses vectorized indexing: metrics[:, failure_indices - j] — the 2D layout enables this in one operation across all sequences
- WvWo being static means it can serve as a baseline/reference for the input-dependent targets (QK^T, AVWo)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-behavioral-evaluation-and-svd-collection*
*Context gathered: 2026-02-25*
