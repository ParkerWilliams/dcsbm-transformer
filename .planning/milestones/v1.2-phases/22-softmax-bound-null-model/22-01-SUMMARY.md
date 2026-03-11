---
phase: 22-softmax-bound-null-model
plan: 01
subsystem: analysis
tags: [softmax, perturbation-bound, lipschitz, weyl, mirsky, spectral-norm, audit]

# Dependency graph
requires:
  - phase: 19-svd-metric-formulas
    provides: SVD metric formula verification patterns
provides:
  - "LaTeX derivation chain verified correct: Prop 3.7 -> Prop 4.1 -> Prop 5.1 -> Thm 6.1"
  - "sqrt(d_k) cancellation proven algebraically between LaTeX and code"
  - "Empirical bound verification: ratio < 1.0 on synthetic fixtures"
  - "Mirsky inequality chain verified: SV-L2 <= Frobenius <= bound"
  - "Bound assumptions audited: causal mask, V/W_O fixed, single-head"
  - "Masking consistency verified: zero-fill direction + -inf softmax"
affects: [22-softmax-bound-null-model]

# Tech tracking
tech-stack:
  added: []
  patterns: [softmax-lipschitz-verification, algebraic-cancellation-proof, inequality-chain-audit]

key-files:
  created:
    - tests/audit/test_softmax_bound.py
    - tests/audit/test_perturbation_bound.py
  modified: []

key-decisions:
  - "LaTeX derivation verified correct step-by-step -- no formula errors found, no production code changes needed"
  - "sqrt(d_k) cancellation algebraically proven: unscaled and scaled formulations produce identical bounds within 1e-12"
  - "Bound assumptions (causal mask, V/W_O fixed, single-head index 0) confirmed via code inspection and behavioral tests"
  - "Masking consistency between zero-fill (SVD) and -inf (softmax) verified as non-problematic"

patterns-established:
  - "Algebraic cancellation proof pattern: compute same quantity two ways (unscaled/scaled), assert identical"
  - "Inequality chain verification: SV-L2 <= Frobenius <= bound tested at multiple magnitudes"

requirements-completed: [SFTX-01, SFTX-02, SFTX-03]

# Metrics
duration: 8min
completed: 2026-03-10
---

# Phase 22 Plan 01: Softmax Bound Derivation Audit Summary

**LaTeX derivation chain (Prop 3.7 -> Thm 6.1) verified correct, sqrt(d_k) cancellation proven algebraically, empirical bound ratio < 1.0 on synthetic fixtures with Mirsky chain**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-10T18:13:17Z
- **Completed:** 2026-03-10T18:20:51Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Verified complete LaTeX derivation chain: softmax Lipschitz 1/2, submultiplicativity, Weyl's inequality, three-stage composition (Prop 3.7 -> Prop 4.1 -> Prop 5.1 -> Thm 6.1)
- Proved sqrt(d_k) cancellation algebraically: unscaled formula (with explicit sqrt(d_k)) equals scaled formula (what code uses) within 1e-12
- Verified empirical bound ratio < 1.0 deterministically for adversarial and 5 random directions across eps=[0.01, 0.05, 0.10]
- Verified Mirsky inequality chain SV-L2 <= Frobenius <= bound at 4 magnitudes
- Confirmed masking consistency: zero-fill in direction generation + -inf in softmax injection = no inconsistency
- Verified all bound assumptions: causal masking, V/W_O held fixed, single-head (index 0)

## Task Commits

Each task was committed atomically:

1. **Task 1: Audit LaTeX derivation chain and sqrt(d_k) cancellation** - `334638a` (test)
2. **Task 2: Audit empirical bound verification and masking consistency** - `828f275` (test)

## Files Created/Modified
- `tests/audit/test_softmax_bound.py` - 12 tests: softmax Lipschitz, submultiplicativity, Weyl, sqrt(d_k) cancellation, derivation chain, assumptions (419 lines)
- `tests/audit/test_perturbation_bound.py` - 11 tests: perturbation construction, Mirsky chain, synthetic bound, masking consistency, Weyl usage (381 lines)

## Decisions Made
- LaTeX derivation verified correct step-by-step -- no formula errors found, no production code changes needed
- sqrt(d_k) cancellation algebraically proven: unscaled and scaled formulations produce identical bounds within 1e-12
- Bound assumptions (causal mask, V/W_O fixed, single-head index 0) confirmed via code inspection and behavioral tests
- Masking consistency between zero-fill (SVD) and -inf (softmax) verified as non-problematic

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Softmax bound derivation and empirical verification audited
- Ready for Phase 22 Plan 02 (null model code-path parity and Marchenko-Pastur)
- All 281 audit tests pass with no regressions

## Self-Check: PASSED

- FOUND: tests/audit/test_softmax_bound.py
- FOUND: tests/audit/test_perturbation_bound.py
- FOUND: 22-01-SUMMARY.md
- FOUND: commit 334638a (Task 1)
- FOUND: commit 828f275 (Task 2)

---
*Phase: 22-softmax-bound-null-model*
*Completed: 2026-03-10*
