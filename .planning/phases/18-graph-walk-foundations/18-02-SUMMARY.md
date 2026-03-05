---
phase: 18-graph-walk-foundations
plan: 02
subsystem: evaluation
tags: [behavioral-classification, rule-outcome, enum, 4-class, audit-tests]

# Dependency graph
requires:
  - phase: 18-01
    provides: "Verified graph/walk/jumper mathematical correctness"
provides:
  - "4-class RuleOutcome enum: UNCONSTRAINED, PENDING, FOLLOWED, VIOLATED"
  - "Updated classify_steps with PENDING countdown labeling"
  - "12 audit tests for behavioral classification (GRAPH-04)"
affects: [19-behavioral-evaluation, 20-auroc-analysis, 21-visualization]

# Tech tracking
tech-stack:
  added: []
  patterns: ["4-class IntEnum for behavioral classification with logical ordering"]

key-files:
  created:
    - tests/audit/test_behavioral_classification.py
  modified:
    - src/evaluation/behavioral.py
    - src/evaluation/pipeline.py
    - src/analysis/event_extraction.py
    - src/visualization/confusion.py
    - tests/test_behavioral.py
    - tests/test_event_extraction.py
    - tests/test_auroc_horizon.py
    - tests/test_calibration.py
    - tests/test_pr_curves.py
    - tests/test_spectrum.py
    - tests/test_statistical_controls.py
    - tests/test_visualization.py
    - src/analysis/null_model.py

key-decisions:
  - "PENDING labels steps where constraint active but deadline in future, distinct from UNCONSTRAINED (no constraint)"
  - "Consumers (event_extraction, confusion) filter to only resolved outcomes (FOLLOWED/VIOLATED) -- semantic parity with old NOT_APPLICABLE filter"

patterns-established:
  - "4-class behavioral enum: UNCONSTRAINED=0 < PENDING=1 < FOLLOWED=2 < VIOLATED=3"
  - "Filter pattern for resolved-only outcomes: outcome in (FOLLOWED, VIOLATED)"

requirements-completed: [GRAPH-04]

# Metrics
duration: 8min
completed: 2026-03-05
---

# Phase 18 Plan 02: Behavioral Classification Summary

**4-class RuleOutcome enum (UNCONSTRAINED/PENDING/FOLLOWED/VIOLATED) with countdown labeling and 12 audit tests covering all classes and consumer compatibility**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-05T07:58:48Z
- **Completed:** 2026-03-05T08:06:48Z
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments
- Replaced 3-class RuleOutcome (NOT_APPLICABLE/FOLLOWED/VIOLATED) with 4-class enum (UNCONSTRAINED/PENDING/FOLLOWED/VIOLATED)
- classify_steps now labels steps during countdown window as PENDING, distinguishing "no constraint" from "constraint active but unresolved"
- Updated all 3 immediate consumers (pipeline.py, event_extraction.py, confusion.py) to handle 4-class enum
- Created 12 audit tests covering enum values, UNCONSTRAINED vs PENDING distinction, countdown sequences, resolution outcomes, and consumer compatibility
- All 581 tests pass with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Expand RuleOutcome to 4-class and update classify_steps** - `a66b9d1` (feat)
2. **Task 2: Update existing tests and create audit tests** - `fa2c418` (test)

## Files Created/Modified
- `src/evaluation/behavioral.py` - 4-class RuleOutcome enum, classify_steps with PENDING logic
- `src/evaluation/pipeline.py` - Array init with UNCONSTRAINED
- `src/analysis/event_extraction.py` - Filter to resolved outcomes only (FOLLOWED/VIOLATED)
- `src/visualization/confusion.py` - Applicable mask for resolved outcomes only
- `tests/audit/test_behavioral_classification.py` - 12 new audit tests for GRAPH-04
- `tests/test_behavioral.py` - Updated NOT_APPLICABLE refs, added PENDING tests
- `tests/test_event_extraction.py` - Updated refs, added PENDING skip test
- `tests/test_auroc_horizon.py` - Updated NOT_APPLICABLE to UNCONSTRAINED
- `tests/test_calibration.py` - Updated NOT_APPLICABLE to UNCONSTRAINED
- `tests/test_pr_curves.py` - Updated NOT_APPLICABLE to UNCONSTRAINED
- `tests/test_spectrum.py` - Updated NOT_APPLICABLE to UNCONSTRAINED
- `tests/test_statistical_controls.py` - Updated NOT_APPLICABLE to UNCONSTRAINED
- `tests/test_visualization.py` - Updated NOT_APPLICABLE to UNCONSTRAINED
- `src/analysis/null_model.py` - Updated comment to UNCONSTRAINED

## Decisions Made
- PENDING labels steps where a jumper constraint is active but its deadline has not yet been reached, giving downstream analysis richer information than the old NOT_APPLICABLE which conflated "no constraint" with "waiting for resolution"
- Consumers filter to resolved outcomes only (FOLLOWED/VIOLATED) which preserves semantic parity with the old NOT_APPLICABLE filter while correctly handling the new 4-class model

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed NOT_APPLICABLE references in 6 additional test files**
- **Found during:** Task 2 (full test suite regression check)
- **Issue:** 6 test files outside the plan's scope (test_auroc_horizon, test_calibration, test_pr_curves, test_spectrum, test_statistical_controls, test_visualization) used RuleOutcome.NOT_APPLICABLE for synthetic data array init
- **Fix:** Replaced all NOT_APPLICABLE with UNCONSTRAINED in these test files
- **Files modified:** tests/test_auroc_horizon.py, tests/test_calibration.py, tests/test_pr_curves.py, tests/test_spectrum.py, tests/test_statistical_controls.py, tests/test_visualization.py
- **Verification:** All 581 tests pass
- **Committed in:** fa2c418 (Task 2 commit)

**2. [Rule 1 - Bug] Updated null_model.py comment referencing NOT_APPLICABLE**
- **Found during:** Task 1 (grep for NOT_APPLICABLE references)
- **Issue:** Docstring comment in null_model.py still referenced NOT_APPLICABLE
- **Fix:** Updated to UNCONSTRAINED
- **Files modified:** src/analysis/null_model.py
- **Committed in:** fa2c418 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs from enum rename)
**Impact on plan:** Both fixes necessary to prevent test failures. No scope creep -- purely mechanical renames required by the enum change.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 4-class behavioral classification complete, ready for downstream analysis phases
- Files using only FOLLOWED/VIOLATED (auroc_horizon, statistical_controls, pr_curves, calibration, spectrum) are unaffected by the value change since they use enum comparison
- Phase 18 complete (both plans done)

## Self-Check: PASSED

All 8 key files verified present. Both task commits (a66b9d1, fa2c418) verified in git history.

---
*Phase: 18-graph-walk-foundations*
*Completed: 2026-03-05*
