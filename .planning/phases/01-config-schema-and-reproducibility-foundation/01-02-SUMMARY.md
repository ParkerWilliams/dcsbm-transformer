---
phase: 01-config-schema-and-reproducibility-foundation
plan: 02
subsystem: reproducibility
tags: [seed, determinism, torch, numpy, random, cuda, git-hash, subprocess]

requires:
  - phase: 01-config-schema-and-reproducibility-foundation
    provides: ExperimentConfig, write_result, validate_result
provides:
  - set_seed() controlling all RNG sources from single master seed
  - verify_seed_determinism() self-test
  - seed_worker() for DataLoader reproducibility
  - get_git_hash() with dirty detection
  - write_result with live code_hash
affects: [training, evaluation, sweep, results]

tech-stack:
  added: []
  patterns: [centralized-seed-management, git-dirty-detection]

key-files:
  created:
    - src/reproducibility/seed.py
    - src/reproducibility/git_hash.py
    - src/reproducibility/__init__.py
    - tests/test_reproducibility.py
  modified:
    - src/results/schema.py

key-decisions:
  - "Seeds set in strict order: random -> numpy -> torch -> cuda -> cudnn -> deterministic_algorithms -> cublas"
  - "git diff --quiet checks both staged and unstaged separately for dirty detection"
  - "verify_seed_determinism tests 10 values from each source (sufficient for statistical confidence)"

patterns-established:
  - "Seed management: always call set_seed(config.seed) before any random operation"
  - "Git provenance: every result.json automatically includes code_hash in metadata"

requirements-completed: [TRNG-02, TRNG-07]

duration: 4min
completed: 2026-02-24
---

# Phase 01 Plan 02: Reproducibility Summary

**Centralized seed management across random/numpy/torch/CUDA with git hash code provenance in result.json metadata**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-24T09:46:00Z
- **Completed:** 2026-02-24T09:50:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- set_seed() seeds Python random, NumPy, PyTorch CPU/GPU, enables cuDNN determinism, enforces deterministic algorithms, and configures cuBLAS workspace
- verify_seed_determinism() proves identical sequences from all three RNG sources after re-seeding
- get_git_hash() returns short SHA with -dirty suffix when uncommitted changes exist
- write_result() now includes live git hash instead of placeholder "unknown"
- 13 new tests covering seed determinism, worker seeding, git hash format, and full end-to-end flow
- All 46 tests pass (18 config + 15 result + 13 reproducibility)

## Task Commits

Each task was committed atomically:

1. **Task 1: Seed management and git hash modules** - `b39dc06` (feat)
2. **Task 2: Wire git hash into result writer and add tests** - `90afca0` (feat)

## Files Created/Modified
- `src/reproducibility/seed.py` - set_seed, verify_seed_determinism, seed_worker
- `src/reproducibility/git_hash.py` - get_git_hash with dirty detection
- `src/reproducibility/__init__.py` - Re-exports for public API
- `src/results/schema.py` - Updated write_result to call get_git_hash()
- `tests/test_reproducibility.py` - 13 tests for seed and git hash behavior

## Decisions Made
- Seeds set in strict order matching PyTorch reproducibility documentation
- Both staged and unstaged changes detected for dirty flag
- verify_seed_determinism uses 10 values per source (statistically sufficient)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 complete: config system + results schema + seed management + git hash
- All 46 tests pass across config, results, and reproducibility
- Ready for Phase 1 verification and transition to Phase 2

---
*Phase: 01-config-schema-and-reproducibility-foundation*
*Completed: 2026-02-24*
