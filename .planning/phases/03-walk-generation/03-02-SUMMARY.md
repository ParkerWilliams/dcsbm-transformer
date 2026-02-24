---
phase: 03-walk-generation
plan: 02
subsystem: walk-generation
tags: [numpy, npz, caching, corpus, train-eval-split, sha256]

requires:
  - phase: 03-walk-generation
    provides: WalkResult, generate_walks, path-count compliance
provides:
  - Corpus assembly with independent train/eval generation
  - Validate corpus for jumper fraction, path diversity, edge validity, compliance
  - Walk caching with config-hash keys and atomic NPZ storage
  - Public API for walk module
affects: [transformer-training, evaluation, sweep-pipeline]

tech-stack:
  added: []
  patterns: [config-hash-based-caching, atomic-npz-storage, parallel-array-event-serialization]

key-files:
  created:
    - src/walk/corpus.py
    - src/walk/cache.py
    - tests/test_walk_corpus.py
  modified:
    - src/walk/__init__.py

key-decisions:
  - "Train seed offset +2000, eval seed offset +3000 from config.seed"
  - "Events stored as parallel arrays (walk_ids, vertex_ids, steps, etc.) in NPZ for efficient serialization"
  - "Cache key includes graph_config_hash so graph param changes auto-invalidate walks"

patterns-established:
  - "Walk caching: key = SHA-256(graph_hash + walk_length + corpus_size + split + seed)"
  - "Atomic NPZ storage: walks + events always in same file"
  - "Parallel array event serialization: flat arrays grouped by walk_id on load"

requirements-completed: [WALK-02, WALK-03, WALK-05]

duration: 5min
completed: 2026-02-24
---

# Phase 03 Plan 02: Corpus Assembly and Caching Summary

**Walk corpus pipeline with independent train/eval generation, config-hash-based NPZ caching, and comprehensive validation (100n threshold, jumper fraction, path diversity)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-24T23:15:00Z
- **Completed:** 2026-02-24T23:20:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- generate_corpus produces independent train (90%) and eval (10%) WalkResult objects with seed offsets +2000/+3000
- validate_corpus checks jumper fraction (>= 50%), path diversity (>= 3 per jumper), edge validity (sample-based), and rule compliance
- Walk caching via save_walks/load_walks with atomic NPZ storage (walks + events always in sync)
- generate_or_load_walks transparently caches both splits, following Phase 2 graph cache pattern
- Public API exported from src/walk/__init__.py with all 9 public symbols
- 10 tests passing covering all corpus and cache behavior

## Task Commits

Each task was committed atomically:

1. **Task 1: Corpus assembly and validation** - `4c61e5b` (feat)
2. **Task 2: Walk caching, public API, and integration tests** - `d1acfd8` (feat)

## Files Created/Modified
- `src/walk/corpus.py` - generate_corpus with train/eval splitting, validate_corpus for quality checks
- `src/walk/cache.py` - walk_cache_key, save_walks, load_walks, generate_or_load_walks
- `src/walk/__init__.py` - Public API exports for all walk module functions
- `tests/test_walk_corpus.py` - 10 test classes for corpus and cache behavior

## Decisions Made
- Train seed = config.seed + 2000, eval seed = config.seed + 3000 (per CONTEXT decision)
- Events serialized as parallel arrays (event_walk_ids, event_vertex_ids, etc.) in NPZ for compact storage and O(n) reconstruction
- Cache key includes graph_config_hash to auto-invalidate walks when graph parameters change

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Walk module complete with generation, corpus assembly, validation, and caching
- All 113 project tests pass (18 new walk tests + 95 existing)
- Phase complete, ready for transition

---
*Phase: 03-walk-generation*
*Completed: 2026-02-24*
