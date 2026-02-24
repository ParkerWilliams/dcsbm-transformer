---
phase: 02-dcsbm-graph-generation
plan: 02
subsystem: graph-generation
tags: [block-jumpers, non-triviality, sparse-reachability, variable-r]

requires:
  - phase: 02-dcsbm-graph-generation
    provides: GraphData, adjacency matrix, block_assignments
provides:
  - JumperInfo frozen dataclass
  - designate_jumpers with variable r cycling across R_SCALES
  - check_non_trivial via sparse matrix-vector reachability
  - verify_all_jumpers batch validation
  - R_SCALES constant and compute_r_values utility
affects: [walk-generation, behavioral-evaluation, graph-caching]

tech-stack:
  added: []
  patterns: [iterative-sparse-reachability, binary-clip-overflow-prevention, reassignment-retry]

key-files:
  created:
    - src/graph/validation.py
    - src/graph/jumpers.py
    - tests/test_jumpers.py
  modified:
    - src/graph/__init__.py

key-decisions:
  - "Global r-value cycling across all blocks (not per-block) ensures all 8 r-scales are represented"
  - "Binary clipping at each iteration step prevents integer overflow in path counting"
  - "Reassignment tries alternative vertices in same block before skipping"

patterns-established:
  - "Sparse vector-matrix reachability: iterate vec @ adj with binary clip per step"
  - "Variable r-per-graph: distribute R_SCALES uniformly across jumpers globally"

requirements-completed: [GRPH-02, GRPH-04]

duration: 3min
completed: 2026-02-24
---

# Phase 2 Plan 02: Block Jumpers Summary

**Block jumper designation with variable r values from 8 scale factors, non-triviality verification via sparse reachability, and reassignment retry logic**

## Performance

- **Duration:** 3 min
- **Completed:** 2026-02-24
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Sparse reachability computation using iterative matrix-vector multiplication with binary clipping
- Non-triviality verification: target block reachable AND non-target blocks also reachable at distance r
- Variable r assignment cycling through R_SCALES=(0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0)*w globally across all blocks
- Reassignment retry: up to block_size/2 alternative vertices when initial choice fails non-triviality
- 19 tests covering reachability, non-triviality, designation count, r-value coverage, and reassignment

## Task Commits

1. **Task 1: Non-triviality verification** - `8451843` (feat, combined with Task 2)
2. **Task 2: Block jumper designation** - `8451843` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Global r-value cycling instead of per-block**
- **Found during:** Task 2 (jumper designation)
- **Issue:** Per-block `jumper_idx` reset to 0 in each block, causing only first 2 r-values to be assigned
- **Fix:** Changed to global `global_jumper_idx` counter that increments across all blocks
- **Verification:** test_all_r_values_represented now passes
- **Committed in:** 8451843

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for correct r-value distribution. No scope creep.

## Issues Encountered
None

## Next Phase Readiness
- Jumper designation complete, ready for graph caching (Plan 02-03) and walk generation (Phase 3)

---
*Phase: 02-dcsbm-graph-generation*
*Completed: 2026-02-24*
