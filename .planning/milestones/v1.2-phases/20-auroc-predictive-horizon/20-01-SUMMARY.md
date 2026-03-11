---
phase: 20-auroc-predictive-horizon
plan: 01
subsystem: testing
tags: [auroc, mann-whitney-u, sklearn, scipy, fence-post, lookback-indexing, predictive-horizon]

# Dependency graph
requires:
  - phase: 18-behavioral-classification
    provides: "Verified 4-class behavioral labels (UNCONSTRAINED/PENDING/FOLLOWED/VIOLATED)"
  - phase: 19-svd-metric-extraction
    provides: "Verified SVD metric formulas and float32 storage"
provides:
  - "AUROC formula audit confirming auroc_from_groups matches sklearn and Mann-Whitney U"
  - "Lookback indexing audit confirming no fence-post errors in compute_auroc_curve"
  - "Predictive horizon logic audit confirming strict inequality and edge case handling"
affects: [20-02-PLAN, auroc-horizon, event-extraction, predictive-horizon]

# Tech tracking
tech-stack:
  added: []
  patterns: [planted-signal-fence-post-testing, three-way-oracle-comparison]

key-files:
  created:
    - tests/audit/test_auroc_computation.py
    - tests/audit/test_lookback_indexing.py
  modified: []

key-decisions:
  - "auroc_from_groups is mathematically correct -- matches sklearn and Mann-Whitney U within 1e-10"
  - "Lookback indexing has no fence-post error -- planted signals confirm j=1 retrieves resolution_step-1"
  - "j=1 means the metric at the resolution step (last value before outcome), not one step before"
  - "compute_predictive_horizon correctly uses strict inequality (val > threshold)"

patterns-established:
  - "Planted-signal testing: distinctive values at known positions to detect off-by-one indexing errors"
  - "Three-way oracle comparison: verify implementation against two independent references (sklearn + scipy)"

requirements-completed: [AUROC-01, AUROC-02]

# Metrics
duration: 8min
completed: 2026-03-06
---

# Phase 20 Plan 01: AUROC Formula and Lookback Indexing Audit Summary

**AUROC rank-based formula verified against sklearn/scipy/Mann-Whitney U within 1e-10; lookback indexing confirmed fence-post-correct via planted-signal tests at 29 test points**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-06T21:03:40Z
- **Completed:** 2026-03-06T21:11:46Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Verified `auroc_from_groups` matches `sklearn.metrics.roc_auc_score` within 1e-10 across 4 distribution types (overlapping, separated, identical, reversed)
- Confirmed Mann-Whitney U equivalence: `U/(n_v*n_c)` matches our AUROC within 1e-10, three-way comparison passes
- Confirmed lookback indexing is fence-post-correct: planted signal at column 7 yields AUROC=1.0 at j=1 only, shifted signal at column 6 yields AUROC=1.0 at j=2 only
- Verified predictive horizon uses strict inequality (0.75 does NOT count), handles NaN, empty, and all-NaN curves correctly
- Analytic Gaussian distribution test: empirical AUROC within 0.02 of theoretical Phi(sqrt(2)) ~ 0.9214
- No production code changes needed -- all formulas verified correct

## Task Commits

Each task was committed atomically:

1. **Task 1: AUROC formula audit tests (AUROC-01)** - `ffa4efb` (test)
2. **Task 2: Lookback indexing fence-post audit tests (AUROC-02)** - `08f86d6` (test)

## Files Created/Modified
- `tests/audit/test_auroc_computation.py` - 14 tests: sklearn comparison (4), Mann-Whitney U equivalence (3), analytic distributions (2), edge cases (5)
- `tests/audit/test_lookback_indexing.py` - 15 tests: planted-signal retrieval (4), metric array shape/bounds (3), predictive horizon (8)

## Decisions Made
- auroc_from_groups is mathematically correct -- no production code changes needed
- Lookback indexing has no fence-post error -- j=1 correctly retrieves metric_array[:, resolution_step - 1]
- j=1 semantic meaning documented: it is the metric at the resolution step (the last attention pattern before the outcome is determined)
- compute_predictive_horizon correctly uses strict inequality (val > threshold, not val >= threshold)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- AUROC formula and lookback indexing verified correct (AUROC-01, AUROC-02)
- Ready for 20-02: cross-path horizon consistency and event extraction boundary audits (AUROC-03, AUROC-04)
- No blockers or concerns

## Self-Check: PASSED

- [x] tests/audit/test_auroc_computation.py exists
- [x] tests/audit/test_lookback_indexing.py exists
- [x] 20-01-SUMMARY.md exists
- [x] Commit ffa4efb found
- [x] Commit 08f86d6 found

---
*Phase: 20-auroc-predictive-horizon*
*Completed: 2026-03-06*
