---
phase: 03-walk-generation
plan: 01
subsystem: walk-generation
tags: [numpy, scipy, csr-matrix, random-walk, path-counting, guided-walk]

requires:
  - phase: 02-dcsbm-graph-generation
    provides: GraphData adjacency CSR matrix, JumperInfo, block_assignments
provides:
  - JumperEvent and WalkResult dataclasses for walk metadata
  - Path-count precomputation via sparse matrix-vector multiply
  - Guided walk generation with jumper constraint satisfaction
  - Vectorized batch walk generation for unguided walks
  - Top-level generate_walks with two-phase strategy
affects: [walk-corpus, walk-caching, transformer-training, evaluation]

tech-stack:
  added: []
  patterns: [path-count-guided-walking, csr-vectorized-batch-walks, per-walk-seed-isolation]

key-files:
  created:
    - src/walk/types.py
    - src/walk/compliance.py
    - src/walk/generator.py
    - src/walk/__init__.py
    - tests/test_walk_generator.py
  modified: []

key-decisions:
  - "Convert numpy int types to Python int in JumperEvent fields for compatibility"
  - "Compliance validation inside generate_single_guided_walk discards non-compliant walks"
  - "5% overgeneration margin for seed pool to minimize extra RNG calls"

patterns-established:
  - "Path-count guided walking: precompute N[k] = adj @ N[k-1] with max-normalization, then weight neighbor selection"
  - "Per-walk seed isolation: each walk gets its own RNG from master seed for reproducibility despite discards"
  - "Two-phase corpus strategy: jumper-seeded guided walks + random-start batch with guided re-run on jumper encounters"

requirements-completed: [WALK-01, WALK-04]

duration: 4min
completed: 2026-02-24
---

# Phase 03 Plan 01: Walk Generation Engine Summary

**Path-count guided walk generator with vectorized batch mode, per-walk seed isolation, and 100% jumper compliance validation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-24T23:09:44Z
- **Completed:** 2026-02-24T23:13:44Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- JumperEvent and WalkResult frozen dataclasses for walk metadata
- precompute_path_counts using iterative sparse matrix-vector multiplication with max-normalization to prevent overflow
- guided_step with multiplicative weight combination for nested jumper constraints
- generate_batch_unguided_walks with vectorized CSR index operations
- generate_walks orchestrating jumper-seeded guided + random-start batch strategies
- 8 tests covering edge validity, compliance, events, batch shape, nested jumpers, normalization, reproducibility, and discard handling

## Task Commits

Each task was committed atomically:

1. **Task 1: Walk types and path-count precomputation** - `7a2df7d` (feat)
2. **Task 2: Walk generator with guided and batch modes** - `9de08aa` (feat)

## Files Created/Modified
- `src/walk/types.py` - JumperEvent and WalkResult frozen dataclasses
- `src/walk/compliance.py` - precompute_path_counts and guided_step functions
- `src/walk/generator.py` - generate_single_guided_walk, generate_batch_unguided_walks, generate_walks
- `src/walk/__init__.py` - Package init (empty, populated in Plan 02)
- `tests/test_walk_generator.py` - 8 test classes covering all walk generation behavior

## Decisions Made
- Convert numpy int types to Python int in JumperEvent fields to ensure isinstance(x, int) checks pass
- Compliance validation inside generate_single_guided_walk: walks failing compliance check are discarded (returned as None)
- 5% overgeneration margin for walk seed pool to minimize extra master RNG calls

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] numpy int32 not recognized as Python int in JumperEvent**
- **Found during:** Task 2 (test execution)
- **Issue:** JumperEvent fields stored np.int32 values, causing isinstance(x, int) assertions to fail
- **Fix:** Added explicit int() conversion when creating JumperEvent instances
- **Files modified:** src/walk/generator.py
- **Verification:** All 8 tests pass
- **Committed in:** 9de08aa (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Trivial type coercion fix. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Walk generation engine complete, ready for corpus assembly (Plan 02)
- Path-count precomputation and guided step logic available for corpus builder
- Per-walk seed isolation ensures reproducibility for cached walks

---
*Phase: 03-walk-generation*
*Completed: 2026-02-24*
