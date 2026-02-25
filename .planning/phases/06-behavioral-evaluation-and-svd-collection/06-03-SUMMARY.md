---
phase: 06-behavioral-evaluation-and-svd-collection
plan: 03
subsystem: evaluation
tags: [fused-pipeline, svd-collection, autoregressive-generation, npz-output]

requires:
  - phase: 06-behavioral-evaluation-and-svd-collection
    provides: "SVD metric functions from 06-01"
  - phase: 06-behavioral-evaluation-and-svd-collection
    provides: "Behavioral classification from 06-02"
  - phase: 04-transformer-architecture
    provides: "TransformerLM with ExtractionMode.SVD_TARGETS"
provides:
  - "fused_evaluate function combining generation + SVD + behavioral labeling"
  - "EvaluationResult dataclass with complete evaluation outputs"
  - "save_evaluation_results for token_metrics.npz and summary dict"
  - "token_metrics.npz with target.layer.metric keyed arrays"
affects: [phase-07-predictive-horizon, phase-08-analysis]

tech-stack:
  added: []
  patterns: [fused-generation-loop, compute-and-discard-svd, npz-keyed-output]

key-files:
  created:
    - src/evaluation/pipeline.py
    - tests/test_evaluation_pipeline.py
  modified:
    - src/evaluation/__init__.py
    - src/evaluation/svd_metrics.py

key-decisions:
  - "AVWo computed as (A @ V) @ W_o.weight.T matching actual residual stream contribution"
  - "read_write_alignment only computed for square matrices (WvWo), skipped for non-square (QK^T, AVWo)"
  - "WvWo SVD computed once and broadcast to all steps (static weight matrix)"
  - "Grassmannian distance uses k=2 subspace dimension with U_prev caching"
  - "NaN for WvWo Grassmannian distance (static, no step-to-step change)"

patterns-established:
  - "Fused generation loop: ExtractionMode.SVD_TARGETS + compute_all_metrics + classify_steps"
  - "Compute-and-discard: SVD intermediates freed per step, only scalar metrics retained"
  - "NPZ key convention: target.layer_N.metric_name (e.g., qkt.layer_0.stable_rank)"

requirements-completed: [EVAL-05, SVD-01, SVD-04, SVD-06]

duration: 5min
completed: 2026-02-25
---

# Phase 6 Plan 03: Fused Evaluation Pipeline Summary

**Fused autoregressive generation with per-step SVD collection on 3 targets (QK^T, WvWo, AVWo) and behavioral labeling in a single inference pass**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-25T21:30:00Z
- **Completed:** 2026-02-25T21:35:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Fused evaluation loop using ExtractionMode.SVD_TARGETS (EVAL-05)
- SVD on 3 targets x all layers at every step (SVD-01)
- token_metrics.npz with target.layer.metric keyed arrays (SVD-04)
- NaN for positions < w (warmup skip per SVD-06)
- Tail extension for late jumper encounters
- save_evaluation_results writes NPZ and returns result.json summary
- 16 integration tests covering all pipeline aspects

## Task Commits

Each task was committed atomically:

1. **Task 1: Fused evaluation pipeline** - `34549368` (feat)
2. **Task 2: Integration tests + NPZ output** - `f4e41f6b` (feat)

## Files Created/Modified
- `src/evaluation/pipeline.py` - fused_evaluate, EvaluationResult, save_evaluation_results
- `tests/test_evaluation_pipeline.py` - 16 integration tests
- `src/evaluation/__init__.py` - Updated exports with pipeline types
- `src/evaluation/svd_metrics.py` - Fixed read_write_alignment for non-square matrices

## Decisions Made
- AVWo = (A @ V) @ W_o.weight.T matching actual residual stream contribution
- read_write_alignment only computed for square matrices (WvWo only)
- WvWo SVD computed once and broadcast (static weight matrices)
- Grassmannian distance k=2 with U_prev caching per layer per target
- NaN for WvWo Grassmannian distance (static, no step-to-step change)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] read_write_alignment dimension mismatch on non-square matrices**
- **Found during:** Task 2 (integration tests)
- **Issue:** read_write_alignment assumes U and Vh have same dimensions, but AVWo SVD produces U=[B,T,k] and Vh=[B,k,D] with T!=D
- **Fix:** Added dimension check in compute_all_metrics: only compute read_write_alignment when m==n (square matrix)
- **Files modified:** src/evaluation/svd_metrics.py
- **Verification:** All 236 tests pass
- **Committed in:** f4e41f6b (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix essential for correct non-square matrix handling. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Fused evaluation pipeline ready for Phase 7 predictive horizon analysis
- token_metrics.npz format documented and tested for downstream consumption
- Phase complete, ready for transition

---
*Phase: 06-behavioral-evaluation-and-svd-collection*
*Completed: 2026-02-25*
