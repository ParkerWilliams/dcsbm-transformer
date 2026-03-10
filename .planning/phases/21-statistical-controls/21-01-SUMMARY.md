---
phase: 21-statistical-controls
plan: 01
subsystem: testing
tags: [permutation-test, bootstrap, bca, holm-bonferroni, cohens-d, statistical-controls]

# Dependency graph
requires:
  - phase: 20-auroc-horizon
    provides: "Verified AUROC computation and predictive horizon formulas"
provides:
  - "Audit tests for shuffle permutation null (H0 uniformity, immutability, group size, signal detection)"
  - "Audit tests for BCa bootstrap CI delegation and known-answer verification"
  - "Audit tests for Holm-Bonferroni step-down correction with textbook example"
  - "Audit tests for Cohen's d pooled standard deviation formula"
affects: [21-02-PLAN]

# Tech tracking
tech-stack:
  added: []
  patterns: [mock-based delegation verification, KS-test H0 uniformity check]

key-files:
  created:
    - tests/audit/test_shuffle_permutation.py
    - tests/audit/test_bootstrap_bca.py
    - tests/audit/test_holm_bonferroni.py
    - tests/audit/test_cohens_d.py
  modified: []

key-decisions:
  - "All four statistical primitives verified correct -- no production code changes needed"
  - "Shuffle permutation H0 uniformity tested via 100 independent trials with KS test at alpha=0.01"
  - "BCa delegation verified via unittest.mock.patch on scipy.stats.bootstrap"

patterns-established:
  - "KS-test uniformity check: run N independent trials, collect p-values, test U[0,1] via kstest"
  - "Mock-based delegation: patch library call, verify method/args, then run end-to-end for known-answer"

requirements-completed: [STAT-01, STAT-02, STAT-03, STAT-04]

# Metrics
duration: 11min
completed: 2026-03-10
---

# Phase 21 Plan 01: Statistical Controls Audit Summary

**All four statistical testing primitives (shuffle null, BCa bootstrap, Holm-Bonferroni, Cohen's d) verified correct against hand calculations and textbook examples -- no production code changes needed**

## Performance

- **Duration:** 11 min
- **Started:** 2026-03-10T01:47:52Z
- **Completed:** 2026-03-10T01:59:38Z
- **Tasks:** 2
- **Files created:** 4

## Accomplishments
- Shuffle permutation null verified: H0 uniformity via KS test (100 trials), metric array immutability, group size preservation per-permutation, signal detection positive control
- BCa bootstrap CI delegation confirmed: scipy.stats.bootstrap called with method='BCa', vectorized=True, confidence_level propagated correctly; known-answer tests for perfect and overlapping distributions
- Holm-Bonferroni correction verified: textbook 5-hypothesis example matches exactly, monotonicity enforcement tested with random inputs, edge cases (empty, single, identical, zeros, ones), 0-based/1-based formula equivalence proven
- Cohen's d verified: hand-calculated pooled std matches within 1e-10, sign convention correct, NaN guards work for n<2 and zero pooled_std, independent manual-loop pooled std computation agrees

## Task Commits

Each task was committed atomically:

1. **Task 1: Audit shuffle permutation null (STAT-01) and Holm-Bonferroni correction (STAT-03)** - `db2285a` (test)
2. **Task 2: Audit BCa bootstrap CIs (STAT-02) and Cohen's d (STAT-04)** - `68a80ef` (test)

## Files Created/Modified
- `tests/audit/test_shuffle_permutation.py` - 5 tests: H0 uniformity, metric immutability, group size preservation, signal detection, no-signal control (214 lines)
- `tests/audit/test_holm_bonferroni.py` - 14 tests: textbook example, all-significant, none-significant, monotonicity, edge cases, formula equivalence (205 lines)
- `tests/audit/test_bootstrap_bca.py` - 9 tests: BCa delegation, confidence_level propagation, vectorized flag, known-answer CIs, NaN handling (188 lines)
- `tests/audit/test_cohens_d.py` - 13 tests: hand calculation, exact case, equal groups, sign convention, NaN guards, independent pooled std (196 lines)

## Decisions Made
- All four statistical primitives verified correct -- no production code changes needed
- Used KS test against U[0,1] with 100 independent trials for H0 uniformity (fast, 2.5 minutes, sufficient power)
- BCa delegation verified via unittest.mock.patch rather than source inspection to avoid brittle string matching
- Cohen's d independent verification uses manual Python loop (sum-of-squares) rather than alternative numpy path to maximize independence

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- STAT-01 through STAT-04 verified, ready for Plan 02 (STAT-05 Spearman redundancy and STAT-06 exploratory/confirmatory split)
- All 41 audit tests pass with no regressions

## Self-Check: PASSED

- All 4 test files exist on disk
- All 2 task commits verified in git log
- All 41 tests pass in combined verification run

---
*Phase: 21-statistical-controls*
*Completed: 2026-03-10*
