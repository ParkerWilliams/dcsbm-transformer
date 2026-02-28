---
phase: 12-null-model-baseline
plan: 01
subsystem: analysis
tags: [null-model, marchenko-pastur, random-matrix-theory, kolmogorov-smirnov, grassmannian-drift]

# Dependency graph
requires:
  - phase: 04-walk-generation
    provides: "generate_batch_unguided_walks for filtered adjacency walking"
  - phase: 06-evaluation-pipeline
    provides: "fused_evaluate for running null walks through trained model"
provides:
  - "Null walk generator (jumper-free walks via column-filtered adjacency)"
  - "Position-matched drift extraction for null vs violation comparison"
  - "Marchenko-Pastur PDF/CDF for random matrix reference"
  - "KS goodness-of-fit test for QK^T singular values against MP"
affects: [12-02-statistical-comparison, 12-null-model-baseline]

# Tech tracking
tech-stack:
  added: []
  patterns: [column-filtered-adjacency, vectorized-cdf, data-calibrated-sigma2]

key-files:
  created:
    - src/analysis/null_model.py
    - tests/test_null_model.py
  modified: []

key-decisions:
  - "Column-filtered adjacency (zero out jumper columns) instead of discard approach -- guarantees 100% jumper-free walks without overgeneration"
  - "Vectorized CDF implementation to handle scipy.stats.kstest passing arrays to callable"
  - "Data-calibrated sigma2 via MP mean formula: sigma2 = mean(sv^2) / (1 + gamma)"

patterns-established:
  - "Standalone analysis module pattern: takes existing experiment artifacts, produces independent analysis"
  - "Fallback strategy: filtered approach with discard fallback for pathological graphs"

requirements-completed: [NULL-01, NULL-03]

# Metrics
duration: 5min
completed: 2026-02-26
---

# Phase 12 Plan 01: Null Model Baseline Summary

**Null walk generator with column-filtered adjacency, position-matched drift extraction, and Marchenko-Pastur KS test for QK^T singular value reference**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-26T17:36:42Z
- **Completed:** 2026-02-26T17:41:31Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Standalone null model analysis module with 6 exported functions
- Null walk generator that produces jumper-free walks by zeroing out adjacency columns for jumper vertices, with dead-end handling and discard fallback
- Position-matched drift extraction that pools metric values at event positions minus lookback distances with NaN filtering
- Marchenko-Pastur PDF integrates to 1.000000; CDF handles both scalar and array inputs
- KS test correctly accepts random Wishart data (high p-value) and rejects structured data (low p-value)
- All 16 new tests pass; no regressions in 358-test suite

## Task Commits

Each task was committed atomically:

1. **Task 1: Write tests for null model** - `973160c7` (test)
2. **Task 2: Implement null model module** - `1bf930e1` (feat)

## Files Created/Modified
- `src/analysis/null_model.py` - Null walk generator, null evaluation wrapper, position-matched drift extraction, Marchenko-Pastur PDF/CDF, KS test
- `tests/test_null_model.py` - 16 test cases covering null walk generation (6), drift extraction (3), and MP reference (7)

## Decisions Made
- **Column-filtered adjacency:** Zero out columns for jumper vertices in the adjacency matrix before walking, rather than generating on full graph and discarding. This guarantees 100% jumper-free walks without overgeneration. With 8/500 jumper vertices, the discard approach would reject ~98.4% of walks (too wasteful).
- **Vectorized CDF:** scipy.stats.kstest passes arrays to the CDF callable, so marchenko_pastur_cdf handles both scalar and array inputs using np.atleast_1d with element-wise quad integration.
- **Data-calibrated sigma2:** Using MP mean formula E[lambda] = sigma2*(1+gamma) to calibrate sigma2 from empirical squared singular values, rather than assuming sigma2=1.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Vectorized marchenko_pastur_cdf for scipy.stats.kstest compatibility**
- **Found during:** Task 2 (implementation)
- **Issue:** scipy.stats.kstest passes arrays to the CDF callable, but the original implementation used scalar comparison operators (x <= lam_minus) which fail on arrays
- **Fix:** Rewrote CDF to handle both scalar and array inputs using np.asarray/np.atleast_1d with element-wise integration loop
- **Files modified:** src/analysis/null_model.py
- **Verification:** All 16 tests pass including test_random_matrix and test_structured_matrix
- **Committed in:** 1bf930e1

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for scipy compatibility. No scope creep.

## Issues Encountered
None beyond the auto-fixed CDF vectorization.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `generate_null_walks()` ready for Plan 12-02 to generate null walks for statistical comparison
- `extract_position_matched_drift()` ready for Plan 12-02 to align null and violation distributions
- `run_mp_ks_test()` ready for Plan 12-02 to compare QK^T singular values at anchor positions
- `run_null_evaluation()` ready to feed null walks through trained model
- All functions are standalone and can be applied to any existing experiment

## Self-Check: PASSED

- All created files exist (src/analysis/null_model.py, tests/test_null_model.py)
- All commits verified (973160c7, 1bf930e1)
- 16/16 tests pass, 358/358 full suite passes

---
*Phase: 12-null-model-baseline*
*Completed: 2026-02-26*
