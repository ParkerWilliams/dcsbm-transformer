---
phase: 22-softmax-bound-null-model
verified: 2026-03-10T19:00:00Z
status: passed
score: 15/15 must-haves verified
re_verification: false
---

# Phase 22: Softmax Bound and Null Model Verification Report

**Phase Goal:** The softmax filtering bound derivation and null model baseline methodology are mathematically sound and correctly implemented
**Verified:** 2026-03-10T19:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | LaTeX derivation chain Prop 3.7 -> Prop 4.1 -> Prop 5.1 -> Thm 6.1 is step-by-step correct | VERIFIED | TestDerivationChain::test_three_stage_composition passes; TestSoftmaxLipschitz::test_softmax_jacobian_formula passes; TestWeylInequality passes |
| 2  | sqrt(d_k) cancellation between LaTeX and code is algebraically verified | VERIFIED | TestSqrtDkCancellation::test_sqrt_dk_cancellation_algebraic asserts identical within 1e-12; test_compute_theoretical_bound_matches_formula passes |
| 3  | Empirical bound verification produces ratio < 1.0 on synthetic fixtures with known matrices | VERIFIED | TestSyntheticBoundVerification::test_adversarial_bound_ratio_below_one and test_random_direction_bound_ratio_below_one both pass for eps=[0.01,0.05,0.10] |
| 4  | Mirsky's inequality chain (SV-L2 <= Frobenius <= bound) holds in both directions | VERIFIED | TestMirskysInequality::test_mirsky_chain_synthetic and test_mirsky_chain_with_theoretical_bound_multiple_magnitudes pass for eps=[0.01,0.05,0.1,0.25] |
| 5  | Bound assumptions (causal masking, V/W_O fixed, single-head) are stated and respected in code | VERIFIED | TestBoundAssumptions all 3 tests pass: causal mask nullifies upper triangle, V/W_O unchanged, head index 0 confirmed via source inspection |
| 6  | Adversarial direction masking (zero-fill vs -inf) is consistent between generation and injection | VERIFIED | TestMaskingConsistency both tests pass: zero-fill direction + -inf softmax produces identical AVWo |
| 7  | null_model.py imports fused_evaluate from src.evaluation.pipeline (code-path identity) | VERIFIED | AST test passes; grep confirms `from src.evaluation.pipeline import EvaluationResult, fused_evaluate` at line 22 |
| 8  | null_model.py imports holm_bonferroni and cohens_d from statistical_controls.py | VERIFIED | AST test passes; grep confirms `from src.analysis.statistical_controls import cohens_d, holm_bonferroni` at line 19 |
| 9  | Column-filtered adjacency produces zero entries to jumper vertices | VERIFIED | TestColumnFilteredAdjacency::test_filtered_adjacency_zeros_jumper_columns passes |
| 10 | Walks on filtered graph never visit jumper vertices (RuntimeError on violation) | VERIFIED | test_null_walks_never_visit_jumper_vertices and test_runtime_error_on_walk_visiting_jumper both pass |
| 11 | Mann-Whitney U matches scipy.stats.mannwhitneyu on identical inputs | VERIFIED | TestMannWhitneyUCorrectness::test_mwu_matches_scipy_exactly passes within 1e-10 |
| 12 | compare_null_vs_violation calls holm_bonferroni on its own p_array (separate family) | VERIFIED | TestHolmBonferroniSeparation structural and functional tests pass; no external p-value parameter in signature |
| 13 | Position matching extracts drift at identical positions for both null and violation | VERIFIED | TestPositionMatchedDrift all 3 tests pass including structural AST check of run_null_analysis |
| 14 | Marchenko-Pastur PDF integrates to 1, CDF is monotone, KS test passes on known MP data | VERIFIED | TestMarchenkoPasturPDF, TestMarchenkoPasturCDF, TestMPKSTest all pass |
| 15 | MP sigma^2 calibration is correctly implemented (bug fixed from incorrect formula) | VERIFIED | sigma2 = mean(sv^2) confirmed at null_model.py lines 312 and 662; old / (1+gamma) formula eliminated |

**Score:** 15/15 truths verified

---

### Required Artifacts

| Artifact | Min Lines | Actual Lines | Status | Details |
|----------|-----------|--------------|--------|---------|
| `tests/audit/test_softmax_bound.py` | 150 | 419 | VERIFIED | 12 tests across 5 classes; all pass |
| `tests/audit/test_perturbation_bound.py` | 200 | 381 | VERIFIED | 11 tests across 5 classes; all pass |
| `tests/audit/test_null_model_parity.py` | 250 | 575 | VERIFIED | 24 tests across 5 classes; all pass |
| `tests/audit/test_marchenko_pastur.py` | 100 | 395 | VERIFIED | 21 tests across 4 classes; all pass |

All four artifacts exceed their minimum line requirements and are fully substantive (no stubs, no placeholders).

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/audit/test_softmax_bound.py` | `src/analysis/perturbation_bound.py:compute_theoretical_bound` | algebraic verification | WIRED | Imported and called in TestSqrtDkCancellation and TestDerivationChain |
| `tests/audit/test_perturbation_bound.py` | `src/analysis/perturbation_bound.py:inject_perturbation` | synthetic fixture verification | WIRED | Imported and called across TestMirskysInequality, TestSyntheticBoundVerification, TestMaskingConsistency |
| `tests/audit/test_perturbation_bound.py` | `src/analysis/perturbation_bound.py:generate_adversarial_direction` | masking consistency check | WIRED | Imported and called across TestPerturbationConstruction, TestMirskysInequality, TestMaskingConsistency |
| `src/analysis/null_model.py` | `src/evaluation/pipeline.py:fused_evaluate` | AST import verification | WIRED | `from src.evaluation.pipeline import EvaluationResult, fused_evaluate` at line 22; fused_evaluate called in run_null_evaluation |
| `src/analysis/null_model.py` | `src/analysis/statistical_controls.py:holm_bonferroni` | AST import verification | WIRED | `from src.analysis.statistical_controls import cohens_d, holm_bonferroni` at line 19; used in compare_null_vs_violation |
| `src/analysis/null_model.py:compare_null_vs_violation` | `scipy.stats.mannwhitneyu` | MW-U test call | WIRED | `from scipy.stats import kstest, mannwhitneyu` at line 16; called at line 382 with alternative="two-sided", method="auto" |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| SFTX-01 | 22-01 | LaTeX derivation of QK^T -> AVWo epsilon-bound is mathematically correct | SATISFIED | TestDerivationChain (three-stage composition), TestSoftmaxLipschitz (Lipschitz 1/2), TestSubmultiplicativity, TestWeylInequality all pass |
| SFTX-02 | 22-01 | Empirical verification code correctly tests the bound's predictions against observed data | SATISFIED | TestSyntheticBoundVerification ratio < 1.0 for adversarial and 5 random-seed directions; TestMirskysInequality chain verified at 4 magnitudes |
| SFTX-03 | 22-01 | Bound assumptions are explicitly stated and implementation respects them | SATISFIED | TestBoundAssumptions (causal mask, V/W_O fixed, single-head index 0), TestSqrtDkCancellation algebraic proof, TestMaskingConsistency |
| NULL-01 | 22-02 | Grassmannian drift on jumper-free sequences is computed identically to primary analysis | SATISFIED | AST tests confirm fused_evaluate, extract_events, filter_contaminated_events all imported from correct modules; run_null_evaluation calls fused_evaluate |
| NULL-02 | 22-02 | Mann-Whitney U comparison uses correct test statistic and p-value computation | SATISFIED | MW-U matches scipy within 1e-10; MP KS test verified with correct sigma^2 calibration |
| NULL-03 | 22-02 | Column-filtered adjacency correctly removes all jumper paths (zero contamination) | SATISFIED | Zero jumper columns verified; null walks never visit jumpers; RuntimeError on post-hoc violation |
| NULL-04 | 22-02 | Null model and primary model use separate Holm-Bonferroni families | SATISFIED | Function signature structurally prevents external p-values; raw_p_values built only from mannwhitneyu within compare_null_vs_violation; family size 3 matches lookbacks exactly |

No orphaned requirements: all 7 IDs appear in plan frontmatter and are implemented.

---

### Anti-Patterns Found

None. Scanned all four test files and both modified source files (perturbation_bound.py, null_model.py) for TODO/FIXME/HACK/placeholder comments, empty implementations, and stub returns. None found.

Notable fix: `src/analysis/null_model.py` lines 312 and 662 show the corrected formula `sigma2 = float(np.mean(sv_squared))`. The old erroneous `/ (1 + gamma)` divisor is absent from the file -- the bug was discovered and fixed during the audit (committed in 8d0761d).

---

### Human Verification Required

None. All truths are verifiable programmatically:
- Mathematical inequalities verified by running tests with concrete tensors
- Code-path parity verified via AST inspection
- Import wiring verified by grepping source files
- Bug fix confirmed by checking the absence of the old formula and the presence of the corrected formula

---

### Commits Verified

All four commits documented in the summaries exist and match the file changes:

| Commit | Task | Files |
|--------|------|-------|
| 334638a | Softmax bound derivation audit | tests/audit/test_softmax_bound.py (419 lines) |
| 828f275 | Perturbation bound empirical audit | tests/audit/test_perturbation_bound.py (381 lines) |
| 3e381a9 | Null model parity audit | tests/audit/test_null_model_parity.py (575 lines) |
| 8d0761d | Marchenko-Pastur audit + sigma^2 fix | tests/audit/test_marchenko_pastur.py (395 lines) + src/analysis/null_model.py fix |

---

### Summary

Phase 22 fully achieves its goal. The softmax filtering bound derivation (SFTX-01 through SFTX-03) is mathematically verified step-by-step with algebraic proof of sqrt(d_k) cancellation, empirical validation on synthetic fixtures, and confirmed assumption compliance. The null model baseline methodology (NULL-01 through NULL-04) is audited with code-path parity proven at AST level, column-filtered adjacency verified correct, Mann-Whitney U confirmed against scipy, Holm-Bonferroni family separation structurally guaranteed, and Marchenko-Pastur distribution formulas validated including a critical sigma^2 calibration bug fix. All 68 audit tests pass with no regressions.

---

_Verified: 2026-03-10T19:00:00Z_
_Verifier: Claude (gsd-verifier)_
