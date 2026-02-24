---
phase: 02-dcsbm-graph-generation
plan: 01
subsystem: graph-generation
tags: [dcsbm, scipy, numpy, sparse-matrix, degree-correction, zipf]

requires:
  - phase: 01-config-schema-and-reproducibility-foundation
    provides: ExperimentConfig, GraphConfig, config hashing, seed management
provides:
  - GraphData frozen dataclass for graph storage
  - sample_theta with Zipf alpha=1.0 per-block normalization
  - build_probability_matrix with block structure and degree correction
  - generate_dcsbm_graph with validation and retry
  - validate_graph checking connectivity, density, min expected degree
affects: [walk-generation, graph-caching, block-jumpers]

tech-stack:
  added: [scipy>=1.14]
  patterns: [sparse-csr-adjacency, bernoulli-edge-sampling, retry-with-seed-increment]

key-files:
  created:
    - src/graph/types.py
    - src/graph/degree_correction.py
    - src/graph/dcsbm.py
    - src/graph/__init__.py
    - tests/test_graph_generation.py
  modified:
    - pyproject.toml

key-decisions:
  - "Used scipy.sparse.csr_matrix for adjacency storage (efficient row operations for path computation)"
  - "Validation uses probability matrix P for expected degree check (not realized degree) per success criteria"
  - "Edge density tolerance computed per block pair using the actual P matrix means"

patterns-established:
  - "Graph generation with retry: increment seed on validation failure, max 10 attempts"
  - "Per-block theta normalization ensures density is controlled by p_in/p_out only"
  - "Frozen GraphData dataclass without __slots__ for numpy/scipy compatibility"

requirements-completed: [GRPH-01, GRPH-03]

duration: 3min
completed: 2026-02-24
---

# Phase 2 Plan 01: DCSBM Generator Summary

**Custom DCSBM directed graph generator with Zipf degree correction, block structure validation gates, and retry-on-failure using scipy sparse matrices**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-24
- **Completed:** 2026-02-24
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- GraphData frozen dataclass holding sparse adjacency, block assignments, theta, and generation metadata
- Zipf alpha=1.0 theta sampling with per-block normalization (CV=2.448 confirms heterogeneity)
- DCSBM probability matrix with block structure and degree correction
- Validation gates: strong connectivity, no self-loops, min expected degree >= 3, edge density within 2-sigma
- Retry mechanism with seed incrementing (max 10 attempts)
- 17 tests covering generation, validation, reproducibility, error cases

## Task Commits

1. **Task 1: Graph data types and degree correction module** - `36a2396` (feat)
2. **Task 2: DCSBM generator with validation and retry** - `662e85c` (feat)

## Files Created/Modified
- `pyproject.toml` - Added scipy>=1.14 dependency
- `src/graph/types.py` - GraphData frozen dataclass
- `src/graph/degree_correction.py` - Zipf theta sampling with per-block normalization
- `src/graph/dcsbm.py` - Core generator: probability matrix, edge sampling, validation, retry
- `src/graph/__init__.py` - Public API re-exports
- `tests/test_graph_generation.py` - 17 tests for generation and validation

## Decisions Made
- Used scipy.sparse.csr_matrix for adjacency (efficient row operations needed for path computation in Plan 02)
- Validation checks expected degree from P matrix rather than realized degree per success criteria wording
- Edge density tolerance uses per-block-pair means from P matrix for accuracy with degree correction

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Graph generation foundation complete
- Ready for Plan 02-02 (block jumpers) which needs GraphData and adjacency matrix
- Ready for Plan 02-03 (caching) which needs generate_dcsbm_graph

---
*Phase: 02-dcsbm-graph-generation*
*Completed: 2026-02-24*
