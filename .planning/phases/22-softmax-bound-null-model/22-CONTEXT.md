# Phase 22: Softmax Bound & Null Model - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify the softmax filtering bound derivation (LaTeX + empirical code) and null model baseline methodology are mathematically sound and correctly implemented. Covers LaTeX derivation correctness, empirical bound verification code, bound assumptions, null model Grassmannian drift parity, Mann-Whitney U, column-filtered adjacency, Holm-Bonferroni family separation, and Marchenko-Pastur formulas. This is an audit-and-fix phase — no new features.

</domain>

<decisions>
## Implementation Decisions

### LaTeX derivation audit (SFTX-01, SFTX-03)
- Verify the derivation chain only: Prop 3.7 (softmax Lipschitz) → Prop 4.1 (AV bound) → Prop 5.1 (W_O bound) → Thm 6.1 (composed bound). Accept 1/2 Lipschitz constant as a known result from Gao & Pavel (2017) — do not re-derive.
- Verify the sqrt(d_k) cancellation algebraically: build a test that computes the bound both ways (unscaled QK^T + explicit sqrt(d_k), and scaled QK^T where factors cancel) and asserts they match. This is a common off-by-sqrt(d_k) bug source.
- Verify assumptions are stated and respected in code: (1) causal masking applied before softmax, (2) V and W_O held fixed during perturbation, (3) single-head attention (head 0 used). Do NOT test bound behavior under assumption violation — that's research, not audit.
- Verify Weyl's inequality usage consistency: the code measures ||sigma(perturbed) - sigma(original)||_2 (L2 of SV difference vector), while the bound is on ||Delta(AVWo)||_F (Frobenius of matrix diff). Verify the relationship between these quantities and that the bound applies to the measured quantity.

### Empirical bound verification (SFTX-02)
- Trace epsilon definition end-to-end: verify the perturbation construction `eps * qkt_fro * direction` is consistent with the LaTeX epsilon definition (relative to unscaled QK^T), accounting for the sqrt(d_k) cancellation.
- Verify both SV-L2 and Frobenius quantities in tests: compute both ||sigma - sigma'||_2 AND ||AVWo - AVWo'||_F, verify Mirsky's inequality holds between them, and verify the Frobenius version is bounded by the theoretical bound. Strongest verification chain.
- Use synthetic fixtures with known matrices (controlled QK^T, V, W_O) where the bound can be computed analytically. Verify ratio < 1.0 deterministically. Do NOT test the 5% violation rate threshold logic — that's for noisy real data, not audit.
- Verify masking consistency in adversarial direction: `generate_adversarial_direction` uses SVD of causally-masked QK^T (zeros in upper triangle), while `inject_perturbation` uses -inf masking for softmax. Verify the upper-triangle zeros in the direction don't create an inconsistency.

### Null model code-path parity (NULL-01, NULL-03)
- AST import check: verify null_model.py imports and calls fused_evaluate from the same module as run_experiment.py. Code-path identity proven at the AST level (Phase 20 pattern).
- Column-filtered adjacency: Test 1 — build small graph with known jumpers, filter columns, verify adjacency has zero entries to jumper vertices. Test 2 — inject a walk visiting a jumper vertex into the verification step, confirm RuntimeError is raised.
- Verify position matching: build test with known event_positions, verify both null and violation drift are extracted at identical positions (same j offsets, same bounds checking via extract_position_matched_drift). Position mismatch would invalidate MW-U comparison.
- Skip the fallback discard path (`_generate_null_walks_discard`) — it's defensive code for pathological graphs (<10 valid start vertices). Audit the primary column-filtered path thoroughly.

### Mann-Whitney U & family separation (NULL-02, NULL-04)
- AST import check for holm_bonferroni and cohens_d: verify null_model.py imports from statistical_controls.py (same functions verified correct in Phase 21). Do NOT re-verify the math.
- Code-path separation check (NULL-04): verify compare_null_vs_violation calls holm_bonferroni on its own p_array (MW-U p-values from lookback distances), and this array never includes p-values from the primary AUROC analysis. Structural check via code reading + AST.
- Verify MW-U call parameters: 'two-sided' is appropriate (detect any difference, not directional), method='auto' selects correct algorithm, U statistic stored correctly in results dict.
- Verify Marchenko-Pastur formulas: build synthetic eigenvalues from a known random matrix, verify MP PDF integrates to 1, CDF is monotone, and KS test returns high p-value for data drawn from true MP distribution. Also verify sigma^2 calibration formula E[lambda] = sigma^2 * (1 + gamma).

### Claude's Discretion
- Specific matrix dimensions and values for synthetic test fixtures
- Internal organization of test classes within tests/audit/
- Order of requirement verification within plans
- Tolerance thresholds for numerical assertions
- Whether to use the Phase 20 AST verification helper or write new AST checks

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `docs/softmax_bound.tex` — 542-line LaTeX derivation with 3 stages, audit target for SFTX-01
- `src/analysis/perturbation_bound.py:compute_theoretical_bound` — Bound formula implementation, audit target for SFTX-02
- `src/analysis/perturbation_bound.py:inject_perturbation` — Perturbation injection pipeline (QK^T → softmax → AV → AVWo)
- `src/analysis/perturbation_bound.py:compute_spectral_change` — L2 norm of SV difference vector measurement
- `src/analysis/perturbation_bound.py:generate_adversarial_direction` — Adversarial direction from SVD of causally-masked QK^T
- `src/analysis/null_model.py:generate_null_walks` — Column-filtered adjacency approach with post-hoc verification
- `src/analysis/null_model.py:compare_null_vs_violation` — MW-U + Cohen's d + Holm-Bonferroni at each lookback
- `src/analysis/null_model.py:marchenko_pastur_pdf/cdf` — MP distribution implementations
- `src/analysis/null_model.py:run_mp_ks_test` — KS test against calibrated MP distribution
- `src/evaluation/pipeline.py:fused_evaluate` — Shared SVD extraction path (used by both primary and null evaluation)
- `tests/audit/` — Established audit test directory (57+ existing tests from Phases 18-21)

### Established Patterns
- Audit test pattern: descriptive class per formula aspect, 1-2 line mathematical reasoning comment per assertion (Phase 18)
- Independent algebra check pattern: recompute with separate code, compare (Phase 19)
- AST-based import verification for code-path audits (Phase 20)
- Fix-on-discovery policy (Phase 18)

### Integration Points
- `perturbation_bound.py` imports ExtractionMode from `src/model/types` and calls model forward pass with SVD_TARGETS mode
- `null_model.py` imports fused_evaluate from `src/evaluation/pipeline` — same path as run_experiment.py
- `null_model.py` imports holm_bonferroni and cohens_d from `src/analysis/statistical_controls` — verified in Phase 21
- `null_model.py` imports extract_events and filter_contaminated_events from `src/analysis/event_extraction` — verified in Phase 20

</code_context>

<specifics>
## Specific Ideas

- The sqrt(d_k) cancellation between LaTeX and code is a concrete audit target — the docstring explains it but no test verifies it
- Phase 20 found that null_model.py does NOT import from auroc_horizon.py — this is correct because null model uses MW-U (a different statistical question), not AUROC
- The adversarial direction's zero-masking vs softmax's -inf-masking is a subtle consistency point worth explicit verification
- Mirsky's inequality chain (SV-L2 <= matrix Frobenius <= bound) should be verified in both directions in synthetic fixtures
- The Marchenko-Pastur sigma^2 calibration from data (E[lambda] = sigma^2 * (1 + gamma)) is a non-obvious formula that needs verification against random matrix theory

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 22-softmax-bound-null-model*
*Context gathered: 2026-03-10*
