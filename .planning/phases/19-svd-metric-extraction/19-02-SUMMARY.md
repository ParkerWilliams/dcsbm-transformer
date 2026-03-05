---
phase: 19-svd-metric-extraction
plan: 02
subsystem: testing
tags: [grassmannian, geodesic, float16, float32, curvature, torsion, spectrum, svd]

# Dependency graph
requires:
  - phase: 15
    provides: "spectrum trajectory storage and curvature/torsion analysis"
provides:
  - "Grassmannian distance verified correct against geodesic definition"
  - "Float16 fidelity quantified — catastrophic error documented"
  - "Pipeline spectrum storage upgraded to float32"
affects: [20-statistical-tests, 21-advanced-svd]

# Tech tracking
tech-stack:
  added: []
  patterns: [ordering-safe synthetic spectra for curvature/torsion testing]

key-files:
  created:
    - tests/audit/test_grassmannian_distance.py
    - tests/audit/test_float16_fidelity.py
  modified:
    - src/evaluation/pipeline.py

key-decisions:
  - "Float16 spectrum storage produces 1130% curvature error and 702M% torsion error — switched to float32"
  - "Grassmannian distance formula verified correct — no production code changes needed"
  - "Synthetic spectra must maintain descending order to avoid ordering-crossing NaN masking"

patterns-established:
  - "Ordering-safe spectra: use well-separated base values (10, 7, 4, 1) for synthetic spectrum tests"
  - "Float precision audit: round-trip cast (f32->f16->f32) to simulate storage pipeline"

requirements-completed: [SVD-04, SVD-05]

# Metrics
duration: 26min
completed: 2026-03-05
---

# Phase 19 Plan 02: Grassmannian Distance and Float16 Fidelity Audit Summary

**Grassmannian distance verified correct against Edelman et al. (1998) geodesic definition; float16 spectrum storage produces catastrophic curvature/torsion error (1130%/702M%) and was upgraded to float32 in pipeline.py**

## Performance

- **Duration:** 26 min
- **Started:** 2026-03-05T19:27:28Z
- **Completed:** 2026-03-05T19:54:02Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Grassmannian distance verified: identical subspaces d=0, orthogonal d=pi/2*sqrt(k), known-angle rotations exact, formula correct for k=1,2,3, clipping handles edge cases, batched computation works (16 tests)
- Float16 fidelity quantified: curvature max relative error 1130%, torsion max relative error 702 million % — both catastrophically exceed 10% threshold
- pipeline.py spectrum storage upgraded from float16 to float32 (pre-allocation and storage cast), docstring updated
- All 648 existing tests pass after production code change

## Task Commits

Each task was committed atomically:

1. **Task 1: Audit Grassmannian distance formula and edge cases (SVD-04)** - `c3d5fa4` (test)
2. **Task 2: Audit float16 vs float32 spectrum fidelity and fix (SVD-05)** - `47a493c` (fix)

## Files Created/Modified
- `tests/audit/test_grassmannian_distance.py` - 16 tests verifying geodesic distance on Grassmann manifold (314 lines)
- `tests/audit/test_float16_fidelity.py` - 5 tests quantifying float16 impact on curvature/torsion (248 lines)
- `src/evaluation/pipeline.py` - Changed spectrum storage from float16 to float32 (2 locations + docstring)

## Decisions Made
- **Float16 -> float32:** Float16 quantization produces catastrophic error in downstream curvature/torsion (second and third derivatives amplify the ~3 decimal digit precision limit). Changed both the pre-allocation dtype and the storage cast in pipeline.py.
- **Grassmannian distance correct:** The implementation matches d = sqrt(sum(theta_i^2)) exactly. No production code changes needed.
- **Ordering-safe synthetic spectra:** Test spectra use well-separated base values (10, 7, 4, 1) to avoid triggering the ordering-crossing detector in spectral_curvature/torsion.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Ordering-crossing masking in synthetic spectra**
- **Found during:** Task 2 (float16 fidelity tests)
- **Issue:** Initial test used a circle centered at base=5.0 for both s1 and s2, causing s1 < s2 for most of the trajectory. The spectral_curvature function's ordering-crossing detector masked these as NaN, leaving only 24 valid points.
- **Fix:** Redesigned synthetic spectra with well-separated base values (10, 7, 4, 1) and small radius (0.3) to maintain strict descending order s1 > s2 > s3 > s4 at all time steps.
- **Files modified:** tests/audit/test_float16_fidelity.py
- **Verification:** 198 valid curvature points and 281 valid torsion points after fix
- **Committed in:** 47a493c (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Auto-fix was necessary to produce meaningful fidelity measurements. No scope creep.

## Issues Encountered
None beyond the deviation documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All SVD metrics and spectrum storage now verified or fixed
- Float32 spectrum storage resolves the v1.1 deferred concern about float16 quantization
- Ready for Phase 19 Plan 03 (curvature/torsion discrete formula audit)

## Self-Check: PASSED

- All files exist (test_grassmannian_distance.py: 314 lines, test_float16_fidelity.py: 245 lines)
- All commits exist (c3d5fa4, 47a493c)
- Minimum line counts met (SVD-04: 314 >= 120, SVD-05: 245 >= 80)

---
*Phase: 19-svd-metric-extraction*
*Completed: 2026-03-05*
