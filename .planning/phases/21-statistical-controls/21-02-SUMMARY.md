---
phase: 21-statistical-controls
plan: 02
subsystem: testing
tags: [spearman, correlation, redundancy, split, stratification, audit]

# Dependency graph
requires:
  - phase: 20-auroc-horizon
    provides: "AUROC computation and event extraction verified correct"
provides:
  - "Spearman rank correlation for measurement redundancy (STAT-05 bug fixed)"
  - "Exploratory/confirmatory split verification (STAT-06)"
  - "31 audit tests covering correlation methods, thresholds, split proportions, determinism, edge cases"
affects: [23-confirmatory-pipeline]

# Tech tracking
tech-stack:
  added: [scipy.stats.spearmanr]
  patterns: [spearman-for-measurement-pearson-for-predictive]

key-files:
  created:
    - tests/audit/test_correlation_redundancy.py
    - tests/audit/test_exploratory_confirmatory_split.py
  modified:
    - src/analysis/statistical_controls.py

key-decisions:
  - "Measurement mode correlation switched from Pearson (np.corrcoef) to Spearman (scipy.stats.spearmanr) per STAT-05 requirement"
  - "Predictive mode retains Pearson on AUROC curves -- bounded scale makes linear correlation appropriate"
  - "spearmanr scalar return for 2-variable case handled by explicit 2x2 matrix construction"

patterns-established:
  - "Spearman for measurement redundancy, Pearson for predictive redundancy"

requirements-completed: [STAT-05, STAT-06]

# Metrics
duration: 12min
completed: 2026-03-10
---

# Phase 21 Plan 02: Correlation Redundancy & Split Audit Summary

**Fixed Pearson-to-Spearman bug in measurement redundancy (STAT-05) and verified exploratory/confirmatory split produces balanced stratified deterministic assignments (STAT-06)**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-10T01:47:59Z
- **Completed:** 2026-03-10T02:00:44Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Identified and fixed Pearson/Spearman discrepancy: STAT-05 requires Spearman rank correlation but code used np.corrcoef (Pearson); production code updated to scipy.stats.spearmanr for measurement mode
- Confirmed |r| > 0.9 threshold is strict inequality (line 329: `r_val > 0.9`), boundary tests verify 0.89 not flagged, 0.91 flagged, 0.90 exactly not flagged
- Verified exploratory/confirmatory split: 50/50 within each stratum, stratification preserves input ratios, exact determinism with SPLIT_SEED=2026, all edge cases handled correctly
- 31 total audit tests added (15 correlation + 16 split), all passing alongside 747 existing tests with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Audit Spearman/Pearson correlation redundancy (STAT-05)** - `da68493` (fix)
2. **Task 2: Audit exploratory/confirmatory split (STAT-06)** - `5a28b92` (test)

## Files Created/Modified
- `tests/audit/test_correlation_redundancy.py` - 15 audit tests: method identification (Spearman vs Pearson), threshold boundary, correlation correctness, predictive mode, edge cases
- `tests/audit/test_exploratory_confirmatory_split.py` - 16 audit tests: proportions, stratification, determinism, edge cases (empty/single/all-violation/odd-count), value validation, seed documentation
- `src/analysis/statistical_controls.py` - Fixed measurement mode to use spearmanr instead of np.corrcoef; added spearmanr import; handled 2-variable scalar return case

## Decisions Made
- **Spearman fix applied:** STAT-05 requirement explicitly says "Spearman correlation for redundancy analysis" and the code used Pearson. Pre-registration (docs/pre-registration.md) does not mention Spearman specifically for redundancy, but the REQUIREMENTS.md specification takes precedence. Fixed production code to use scipy.stats.spearmanr for measurement mode.
- **Predictive mode stays Pearson:** AUROC values are already on a bounded [0, 1] scale where linear correlation is appropriate. Only measurement mode (raw metric values) needs rank-based correlation.
- **2-variable edge case:** scipy.stats.spearmanr returns a scalar (not a matrix) for exactly 2 variables; explicit 2x2 matrix construction added.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Measurement mode used Pearson instead of Spearman**
- **Found during:** Task 1 (correlation method identification)
- **Issue:** STAT-05 requires Spearman rank correlation but compute_correlation_matrix used np.corrcoef (Pearson) for measurement mode
- **Fix:** Changed to scipy.stats.spearmanr; added import; handled 2-variable scalar return
- **Files modified:** src/analysis/statistical_controls.py
- **Verification:** Monotone non-linear data (y=x^3) produces r=1.0 (Spearman) not r~0.95 (Pearson); 747 existing tests pass
- **Committed in:** da68493 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix was the primary purpose of STAT-05 audit. The plan explicitly anticipated this discrepancy and instructed resolution.

## Issues Encountered
None -- plan executed smoothly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 21 plans 01 and 02 cover all six STAT requirements
- Statistical controls machinery fully audited: shuffle permutation null, BCa bootstrap, Holm-Bonferroni, Cohen's d, Spearman correlation, and split assignment
- Ready for Phase 22 (Softmax Filtering Bound) or Phase 23 (Confirmatory Pipeline)

## Self-Check: PASSED

- tests/audit/test_correlation_redundancy.py: FOUND
- tests/audit/test_exploratory_confirmatory_split.py: FOUND
- 21-02-SUMMARY.md: FOUND
- Commit da68493: FOUND
- Commit 5a28b92: FOUND

---
*Phase: 21-statistical-controls*
*Completed: 2026-03-10*
