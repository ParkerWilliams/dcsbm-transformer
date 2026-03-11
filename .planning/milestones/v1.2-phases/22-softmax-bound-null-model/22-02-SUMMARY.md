---
phase: 22-softmax-bound-null-model
plan: 02
subsystem: testing
tags: [null-model, marchenko-pastur, mann-whitney-u, holm-bonferroni, ast-audit, code-path-parity]

# Dependency graph
requires:
  - phase: 20-auroc-predictive-horizon
    provides: "AST-based import verification pattern for code-path audits"
  - phase: 21-statistical-controls
    provides: "holm_bonferroni and cohens_d verified correct (STAT-03, STAT-04)"
provides:
  - "Null model code-path parity proven via AST (fused_evaluate, extract_events, holm_bonferroni, cohens_d)"
  - "Column-filtered adjacency verified: zero jumper columns, no jumper visits, RuntimeError on violation"
  - "MW-U verified against scipy.stats.mannwhitneyu within 1e-10"
  - "Holm-Bonferroni family separation structurally verified (no external p-value contamination)"
  - "Position-matched drift extraction verified at correct offsets"
  - "MP PDF/CDF verified: integration to 1, monotonicity, boundary values"
  - "Sigma^2 calibration bug fixed: E[lambda_MP] = sigma^2, not sigma^2*(1+gamma)"
  - "KS test verified: accepts true MP data, rejects non-MP data"
affects: [22-softmax-bound-null-model]

# Tech tracking
tech-stack:
  added: []
  patterns: [mp-distribution-verification, random-matrix-theory-calibration]

key-files:
  created:
    - tests/audit/test_null_model_parity.py
    - tests/audit/test_marchenko_pastur.py
  modified:
    - src/analysis/null_model.py

key-decisions:
  - "MP sigma^2 calibration corrected: E[lambda_MP(gamma, sigma^2)] = sigma^2, not sigma^2*(1+gamma)"
  - "For gamma > 1, MP continuous density integrates to 1/gamma (point mass of 1-1/gamma at zero)"
  - "40-vertex test graph used for column filtering tests (minimum 10 valid start vertices required)"

patterns-established:
  - "Random matrix theory verification: generate X ~ N(0, sigma^2), compute eigenvalues of (1/n) X X^T, verify against MP"
  - "MP distribution validation chain: PDF integration, CDF monotonicity, KS test positive and negative controls"

requirements-completed: [NULL-01, NULL-02, NULL-03, NULL-04]

# Metrics
duration: 13min
completed: 2026-03-10
---

# Phase 22 Plan 02: Null Model Parity and Marchenko-Pastur Audit Summary

**45 audit tests verifying null model code-path parity via AST, column-filtered adjacency correctness, MW-U against scipy, Holm-Bonferroni family separation, position matching, and MP distribution formulas with sigma^2 calibration bug fix**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-10T18:13:11Z
- **Completed:** 2026-03-10T18:26:20Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Verified null model uses identical SVD extraction path as primary analysis via AST (fused_evaluate, extract_events, filter_contaminated_events all from correct modules)
- Verified column-filtered adjacency zeros jumper columns, null walks never visit jumper vertices, RuntimeError raised on violation
- MW-U matches scipy.stats.mannwhitneyu within 1e-10; Holm-Bonferroni family separation confirmed structurally
- Fixed sigma^2 calibration bug in run_mp_ks_test: E[lambda_MP] = sigma^2, not sigma^2*(1+gamma)
- MP PDF integrates to 1.0 for gamma <= 1, CDF is monotone with correct boundary values
- KS test correctly accepts true MP data and rejects non-MP uniform data

## Task Commits

Each task was committed atomically:

1. **Task 1: Audit null model code-path parity, column filtering, MW-U, and family separation** - `3e381a9` (test)
2. **Task 2: Audit Marchenko-Pastur distribution formulas and sigma^2 calibration** - `8d0761d` (test + fix)

## Files Created/Modified
- `tests/audit/test_null_model_parity.py` - 24 tests: AST import parity, column-filtered adjacency, MW-U correctness, Holm-Bonferroni family separation, position-matched drift
- `tests/audit/test_marchenko_pastur.py` - 21 tests: MP PDF integration/non-negativity, CDF monotonicity/boundaries, sigma^2 calibration, KS test positive/negative controls
- `src/analysis/null_model.py` - Fixed sigma^2 calibration formula in run_mp_ks_test and run_null_analysis

## Decisions Made
- MP sigma^2 calibration corrected: the mean of the MP distribution with parameters (gamma, sigma^2) is sigma^2 (verified by integrating x * f_MP(x) over the support), not sigma^2 * (1 + gamma) as previously implemented
- For gamma > 1, the continuous part of MP integrates to 1/gamma (the remaining 1-1/gamma mass is a point mass at zero from n-m zero eigenvalues); test handles this correctly
- Used 40-vertex test graph for column filtering tests because generate_null_walks requires >= 10 valid start vertices after filtering

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed sigma^2 calibration formula in run_mp_ks_test**
- **Found during:** Task 2 (Marchenko-Pastur audit)
- **Issue:** `run_mp_ks_test` computed `sigma2 = mean(sv^2) / (1 + gamma)`, but E[lambda_MP(gamma, sigma^2)] = sigma^2, so the correct calibration is `sigma2 = mean(sv^2)`. The (1+gamma) divisor caused KS test to reject true MP data (p=0.001).
- **Fix:** Changed formula to `sigma2 = float(np.mean(sv_squared))` in both `run_mp_ks_test` and the inline computation in `run_null_analysis`. Updated docstring to reflect correct mathematical relationship.
- **Files modified:** `src/analysis/null_model.py` (lines 309, 662)
- **Verification:** KS test now correctly accepts true MP data (p > 0.05) and rejects non-MP data (p < 0.05). Sigma^2 estimation matches true value within 5% across 5 random matrix trials.
- **Committed in:** `8d0761d` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Critical bug fix -- without it, the MP KS test would always reject true MP data due to incorrect sigma^2 calibration. No scope creep.

## Issues Encountered
None beyond the sigma^2 calibration bug discovered and fixed during audit.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All null model audit requirements (NULL-01 through NULL-04) verified complete
- Phase 22 complete -- both softmax bound (Plan 01) and null model (Plan 02) audited
- Ready for Phase 23

---
*Phase: 22-softmax-bound-null-model*
*Completed: 2026-03-10*
