---
phase: 19-svd-metric-extraction
plan: 03
subsystem: testing
tags: [frenet-serret, curvature, torsion, discrete-differential-geometry, convergence, numpy]

# Dependency graph
requires:
  - phase: 15-advanced-analysis
    provides: spectral_curvature and spectral_torsion implementations in spectrum.py
provides:
  - Frenet-Serret curvature/torsion audit tests with analytic curve verification and convergence analysis
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns: [descending-base-values for crossing-mask-safe synthetic spectra]

key-files:
  created: [tests/audit/test_curvature_torsion.py]
  modified: []

key-decisions:
  - "Discrete curvature on circle achieves O(h^2) convergence, not O(h) as initially expected -- formula is more accurate than minimum theoretical guarantee"
  - "Synthetic spectra use descending base values (100, 90, 80, ...) with small oscillation to avoid triggering ordering-crossing mask, isolating formula audit from crossing logic"

patterns-established:
  - "Analytic curve test pattern: construct circle/helix with known curvature/torsion, compare discrete implementation output to analytic values"
  - "Convergence test pattern: measure error at multiple resolutions (100, 1000, 10000 points), verify monotonic decrease and rate"

requirements-completed: [SVD-06]

# Metrics
duration: 23min
completed: 2026-03-05
---

# Phase 19 Plan 03: Curvature/Torsion Formula Audit Summary

**Discrete Frenet-Serret curvature/torsion formulas verified against circle and helix analytic curves with O(h^2) convergence and correct index mapping**

## Performance

- **Duration:** 23 min
- **Started:** 2026-03-05T19:27:27Z
- **Completed:** 2026-03-05T19:51:02Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Circle curvature converges to 1/r within 5% at N=1000, verified for r=1.0 and r=2.0
- Circle torsion confirmed near-zero (planar curve has no out-of-plane twist)
- Helix curvature and torsion match analytic formulas (kappa = r/(r^2+c^2), tau = c/(r^2+c^2)) within 10%
- Convergence rate is O(h^2) -- better than the O(h) minimum expected from forward differences
- Index mapping verified: curvature peak correctly maps to orig_idx=t+1, torsion peak to orig_idx=t+2
- Boundary NaN behavior confirmed: curvature[0], curvature[-1], torsion[0], torsion[1], torsion[-1] are all NaN

## Task Commits

Each task was committed atomically:

1. **Task 1: Audit curvature formula on circle and verify convergence (SVD-06 part 1)** - `5bf7791` (test)

## Files Created/Modified
- `tests/audit/test_curvature_torsion.py` - 13 audit tests in 7 classes verifying discrete curvature/torsion against analytic curves (420 lines)

## Decisions Made
- Convergence test bounds adjusted: the plan expected O(h) convergence (10x error reduction per 10x resolution), but actual convergence is O(h^2) (~100x reduction). The test now verifies at-least O(h) rate (ratio > 5x) rather than imposing an upper bound, since faster convergence is strictly better.
- Synthetic spectra use descending base values with 10-unit gaps between dimensions (100, 90, 80, ...) to guarantee no ordering crossings are triggered. This isolates the curvature/torsion formula audit from the crossing-mask logic, which is a separate concern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed ordering-crossing false positives on synthetic spectra**
- **Found during:** Task 1 (initial test run)
- **Issue:** Circle/helix generators placed all dimensions at the same base offset (5.0) with oscillations causing adjacent dimensions to swap ordering, triggering the crossing mask and NaN-ing most values (only 124/1000 valid)
- **Fix:** Created `_descending_bases()` helper producing well-separated base values (100, 90, 80, ...) so oscillation amplitude (r=1.0) never causes dimension swaps
- **Files modified:** tests/audit/test_curvature_torsion.py
- **Verification:** All 13 tests pass, >50% valid curvature points at each resolution
- **Committed in:** 5bf7791

**2. [Rule 1 - Bug] Fixed convergence rate test bounds**
- **Found during:** Task 1 (convergence test)
- **Issue:** Test expected 3x-30x error ratio for O(h) convergence, but actual ratio was ~100x (O(h^2) convergence)
- **Fix:** Changed upper bound to allow O(h^2) convergence, kept lower bound at 5x to verify at-least O(h) rate
- **Files modified:** tests/audit/test_curvature_torsion.py
- **Verification:** Convergence test passes with observed ~100x ratio
- **Committed in:** 5bf7791

---

**Total deviations:** 2 auto-fixed (2 bugs in test construction)
**Impact on plan:** Both fixes were in test code, not production code. The curvature/torsion implementation in spectrum.py is verified correct without modification.

## Issues Encountered
- Pre-existing uncommitted files from plans 19-01/19-02 (test_float16_fidelity.py, pipeline.py changes) exist in working directory. The float16 test failure is caused by a partially-committed change set from plan 19-02 and is not related to this plan. Logged as out-of-scope.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All SVD-06 requirements satisfied: curvature/torsion formulas verified correct
- Production code (src/analysis/spectrum.py) confirmed correct -- no changes needed
- 13 new audit tests added to regression suite (100 total audit tests now pass)

## Self-Check: PASSED

- FOUND: tests/audit/test_curvature_torsion.py
- FOUND: commit 5bf7791
- FOUND: 19-03-SUMMARY.md

---
*Phase: 19-svd-metric-extraction*
*Completed: 2026-03-05*
