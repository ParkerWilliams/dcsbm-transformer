# Phase 19: SVD Metric Extraction - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify every SVD-related metric formula and matrix construction against its mathematical definition, including numerical fidelity of spectrum storage. Covers QK^T construction, WvWo/AVWo matrices, all singular-value-derived metrics, Grassmannian distance, float16 vs float32 fidelity, and Frenet-Serret curvature/torsion. This is an audit-and-fix phase — no new features.

</domain>

<decisions>
## Implementation Decisions

### Float16 fidelity strategy (SVD-05)
- Side-by-side comparison: compute curvature/torsion from same spectra in float16 vs float32, report max absolute error and relative error
- Use synthetic spectra with analytically known curvature (not real model output) — deterministic, reproducible, isolates precision issue
- Threshold: if relative error > 10%, recommend and apply switch to float32
- Fix on discovery: if float16 fails the 10% threshold, change `pipeline.py` storage from float16 to float32 (consistent with Phase 18 fix-on-discovery policy)

### Grassmannian distance definition (SVD-04)
- Verify against geodesic distance definition: d = sqrt(sum(theta_i^2)) where theta_i = arccos(sigma_i), per Edelman et al. (1998)
- Test with rotation matrices having analytically computable principal angles — compare implementation output to hand-calculated geodesic distance
- Test edge cases: identical subspaces (d=0), orthogonal subspaces (d=pi/2*sqrt(k)), near-degenerate cases — verify clipping doesn't hide bugs
- Verify formula for k=1, k=2, k=3 (not just default k=2) — the formula should generalize

### Curvature/torsion discrete formula (SVD-06)
- Test formulas on unsmoothed analytic curves — skip Savitzky-Golay to isolate the discrete differential geometry from the filter
- Use circle (known curvature = 1/r, zero torsion) and helix (known curvature and nonzero torsion) as test curves
- Convergence test with step refinement: test at multiple sampling rates (100, 1000, 10000 points), show error decreases as expected (O(h) for forward differences)
- Explicitly verify index mapping: a[t] → orig_idx=t+1 for curvature, j[t] → orig_idx=t+2 for torsion — test with a curve having a known peak-curvature location

### QK^T construction and matrix verification scope (SVD-01, SVD-02, SVD-03)
- QK^T: create small model with known weights, compute QK^T = (x@Wq) @ (x@Wk)^T / sqrt(d_k) manually, compare to extracted matrix
- Explicitly test both masks: SVD target gets zero-filled mask, attention path gets -inf mask — critical distinction
- WvWo/AVWo: verify matrix construction correctness for single-head (1h) and 4-head configurations (2h is redundant if 1 and 4 work)
- All 5 singular-value metrics (condition number, spectral gap, entropy, effective rank, stable rank): each verified against analytically known matrices with predetermined singular values

### Claude's Discretion
- Specific matrix dimensions and weight values for test fixtures
- Tolerance thresholds for convergence tests
- Internal organization of test files within tests/audit/
- Order of requirement verification within plans

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/evaluation/svd_metrics.py` — All 9 SVD metric functions (stable_rank, spectral_entropy, spectral_gap_1_2/2_3/4_5, condition_number, rank1_residual_norm, read_write_alignment, grassmannian_distance), plus compute_all_metrics aggregator
- `src/analysis/spectrum.py` — Frenet-Serret curvature/torsion with Savitzky-Golay smoothing, ordering crossing detection
- `src/model/attention.py:CausalSelfAttention` — QK^T extraction with dual mask (zero-fill for SVD, -inf for softmax)
- `src/evaluation/pipeline.py` — Fused evaluation: QK^T/WvWo/AVWo extraction, spectrum storage in float16
- `tests/audit/` — Established audit test directory from Phase 18 (26 existing tests)

### Established Patterns
- Audit test pattern: descriptive class per formula aspect, 1-2 line mathematical reasoning comment per assertion
- Independent algebra check pattern: recompute result with separate code, compare to implementation
- PyTorch for SVD metrics (torch.linalg.svd, torch.linalg.svdvals), NumPy for spectrum analysis
- guard_matrix_for_svd clamps non-finite values before SVD computation

### Integration Points
- `pipeline.py` stores spectrum as float16 at line 234 — may need changing to float32 based on audit findings
- `model.get_wvwo()` returns [n_layers, n_heads, d_model, d_model] — WvWo is static (computed once)
- `_compute_avwo_for_layer` constructs AVWo per step per head with W_o slicing for multi-head
- Grassmannian distance computed between consecutive steps using top-k columns of U matrices

</code_context>

<specifics>
## Specific Ideas

- Phase 18 established that all graph/walk formulas were correct — no production code changes needed. SVD phase may find actual discrepancies given float16 concern and multi-head slicing complexity.
- The v1.1 deferred concern about spectrum trajectory float32 storage and curvature/torsion float16 quantization should be resolved definitively in this phase.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 19-svd-metric-extraction*
*Context gathered: 2026-03-05*
