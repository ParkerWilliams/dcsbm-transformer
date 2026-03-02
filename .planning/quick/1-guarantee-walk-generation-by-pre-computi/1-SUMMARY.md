---
phase: quick-1
plan: 1
subsystem: walk-generation
tags: [path-splicing, compliance, random-walks, precomputation]

# Dependency graph
requires: []
provides:
  - "precompute_viable_paths() for guaranteed jumper compliance"
  - "Path-splicing walk generation replacing probabilistic guided stepping"
  - "TestViablePathPrecomputation test suite"
affects: [walk-corpus, training-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: ["path-splicing for constraint satisfaction", "pre-computed path pools"]

key-files:
  created: []
  modified:
    - "src/walk/compliance.py"
    - "src/walk/generator.py"
    - "tests/test_walk_generator.py"

key-decisions:
  - "Kept precompute_path_counts and guided_step as preserved code (not removed) since they may serve other uses"
  - "Only the primary jumper triggering a splice records an event; intermediate jumpers inside a splice are overridden"
  - "Kept _validate_walks as belt-and-suspenders compliance check even though splicing guarantees correctness"
  - "Path precomputation uses seed+4000 for independent RNG stream from walk seeds"

patterns-established:
  - "Path splicing: pre-compute viable paths, splice at encounter, guarantee compliance by construction"

requirements-completed: ["QUICK-1"]

# Metrics
duration: 6min
completed: 2026-03-02
---

# Quick Task 1: Guarantee Walk Generation by Pre-computing Viable Paths

**Pre-computed viable path pools with splice-based walk generation, eliminating all probabilistic discard/retry logic for jumper compliance**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-02T19:11:48Z
- **Completed:** 2026-03-02T19:18:04Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Added `precompute_viable_paths()` that generates pools of r-step random walks from each jumper vertex to its target block, with configurable pool size (default 200) and attempt limits
- Refactored `generate_single_guided_walk()` to splice pre-computed paths into walks at jumper encounters, replacing step-by-step weighted neighbor selection
- Achieved 100% compliance rate by construction (zero discards due to constraint infeasibility)
- All 90 non-torch tests pass including 4 new viable path precomputation tests
- Smoke test with anchor config (500 vertices, 4 blocks) confirms all 155 events across 100 walks are compliant

## Task Commits

Each task was committed atomically:

1. **Task 1: Pre-compute viable jumper paths in compliance.py** - `b3ab4c5` (feat)
2. **Task 2: Refactor walk generator to use path splicing and update tests** - `b9a564a` (feat)

## Files Created/Modified
- `src/walk/compliance.py` - Added `precompute_viable_paths()` function (96 lines) with JumperInfo import
- `src/walk/generator.py` - Replaced path_counts-based guided walking with viable_paths splicing; updated imports; added error logging for dead-end vertices
- `tests/test_walk_generator.py` - Updated existing tests to use `precompute_viable_paths`; added `TestViablePathPrecomputation` class with 4 tests (existence, target block, edge validity, no discard)

## Decisions Made
- Kept `precompute_path_counts()` and `guided_step()` in compliance.py as preserved code rather than deleting them, since they may be useful for other analyses
- Only the primary jumper that triggers a splice records a `JumperEvent`; intermediate jumper vertices within a splice segment have their constraints overridden by the splice (avoids false compliance expectations)
- Retained `_validate_walks()` post-hoc compliance checking as defense-in-depth, even though path splicing guarantees compliance by construction
- Used `seed + 4000` for path precomputation RNG to keep it independent of walk generation seeds while remaining deterministic

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed per-walk compliance counting in test_guided_walk_compliance**
- **Found during:** Task 2 (test updates)
- **Issue:** Original test counted per-event compliance instead of per-walk, causing compliant_count to exceed total when walks had multiple events
- **Fix:** Changed to per-walk compliance tracking (all events in a walk must be compliant for the walk to count)
- **Files modified:** tests/test_walk_generator.py
- **Verification:** Test passes with correct assertion (compliant_count == total)
- **Committed in:** b9a564a (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Test logic correction necessary for correct assertion. No scope creep.

## Issues Encountered
- Missing `dacite` dependency in venv (pre-existing environment issue, not related to changes). Installed with pip.
- Full test suite has torch-dependent tests that cannot run in this environment (pre-existing, not related to changes). Ran all 90 non-torch tests successfully.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Walk generation now guarantees compliance by construction
- Path splicing approach is backward-compatible (same WalkResult output type)
- Reproducibility preserved (same seed produces same walks)

## Self-Check: PASSED

All files verified present, all commits verified in git log.

---
*Plan: quick-1*
*Completed: 2026-03-02*
