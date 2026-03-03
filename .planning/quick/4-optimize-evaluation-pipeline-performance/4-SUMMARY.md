---
phase: quick-4
plan: 01
subsystem: evaluation
tags: [numpy-vectorization, gpu-cpu-transfer, auroc, svd-metrics, performance]

# Dependency graph
requires:
  - phase: quick-3
    provides: "Performance profiler identifying the hot paths to optimize"
provides:
  - "Deferred GPU->CPU bulk transfer in evaluation pipeline (eliminates per-step synchronization)"
  - "Vectorized slice assignments for SVD metric storage (eliminates per-element Python loops)"
  - "Vectorized AUROC curve computation using numpy advanced indexing"
  - "Pre-extracted value matrix for shuffle controls (eliminates per-permutation event object creation)"
  - "WvWo quadratic nesting bug fix"
affects: [evaluation, analysis, runpod-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns: ["GPU tensor pre-allocation with deferred bulk CPU transfer", "numpy advanced indexing for event-based lookback extraction", "pre-extracted value matrices for permutation tests"]

key-files:
  created: []
  modified:
    - "src/evaluation/pipeline.py"
    - "src/analysis/auroc_horizon.py"

key-decisions:
  - "Use shared tensor references for legacy key aliasing during GPU generation, shared numpy arrays after transfer"
  - "WvWo loop de-nested from QK^T/AVWo layer loop to fix quadratic bug (was iterating n_layers * n_heads times per step unnecessarily)"
  - "Pre-extract all metric values into values_by_j matrix for shuffle controls instead of re-creating AnalysisEvent objects per permutation"

patterns-established:
  - "Deferred CPU transfer: pre-allocate GPU tensors, store during generation, bulk transfer once after all batches"
  - "Vectorized event extraction: convert event fields to numpy arrays, use advanced indexing instead of per-event Python loops"

requirements-completed: [PERF-01, PERF-02, PERF-03, PERF-04]

# Metrics
duration: 8min
completed: 2026-03-03
---

# Quick Task 4: Optimize Evaluation Pipeline Performance Summary

**Deferred GPU-CPU transfers, vectorized SVD metric storage with slice assignments, vectorized AUROC lookback extraction, and pre-extracted shuffle control value matrices**

## Performance

- **Duration:** 8 min
- **Started:** 2026-03-03T23:08:39Z
- **Completed:** 2026-03-03T23:17:00Z
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- Eliminated all per-element `for b_idx` loops from SVD metric storage in pipeline.py (replaced with vectorized slice assignments)
- Deferred all `.cpu().numpy()` calls to after the generation loop completes (single bulk GPU->CPU transfer)
- Fixed WvWo quadratic nesting bug: was iterating n_layers * n_heads times per step inside the already-iterating QK^T/AVWo loop
- Vectorized AUROC curve computation using numpy advanced indexing instead of per-event Python list-append loops
- Optimized shuffle controls to pre-extract all metric values once into a (r_value, n_total) matrix, then permute boolean masks instead of creating AnalysisEvent objects
- Vectorized n_valid_by_lookback computation in run_auroc_analysis
- All 540 tests pass with zero failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Defer CPU transfers and vectorize metric storage in pipeline.py** - `15594be` (perf)
2. **Task 2: Vectorize AUROC curve computation and optimize shuffle controls** - `2baa333` (perf)
3. **Task 3: Run full test suite to verify no regressions** - verification only, no code changes

## Files Created/Modified
- `src/evaluation/pipeline.py` - Deferred GPU->CPU transfer, vectorized slice assignments for all SVD metric storage, WvWo bug fix, vectorized behavioral copy
- `src/analysis/auroc_horizon.py` - Vectorized compute_auroc_curve, optimized run_shuffle_control with pre-extracted value matrix, vectorized n_valid_by_lookback

## Decisions Made
- Legacy key aliasing uses shared tensor references during GPU generation phase, then shared numpy arrays after bulk CPU transfer (avoids double-transferring identical data)
- WvWo block uses separate loop variable names (layer_idx_w, head_idx_w) to avoid shadowing the outer QK^T/AVWo loop variables after de-nesting
- Post-generation behavioral copy simplified: edge_valid and rule_outcome use uniform batch slice instead of per-sequence copy_len (max_steps-1 is the uniform bound)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed WvWo quadratic nesting bug**
- **Found during:** Task 1 (pipeline.py optimization)
- **Issue:** WvWo broadcast loop was INCORRECTLY nested inside the QK^T/AVWo `for layer_idx / for head_idx` loop, causing it to execute n_layers * n_heads times per step instead of once
- **Fix:** De-nested WvWo block to run at the same level as the outer QK^T/AVWo loop, with separate loop variable names
- **Files modified:** src/evaluation/pipeline.py
- **Verification:** All 16 pipeline tests pass, WvWo values still static across steps
- **Committed in:** 15594be (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix per plan specification)
**Impact on plan:** Bug fix was explicitly called out in the plan as a required correction. No scope creep.

## Issues Encountered
- Torch installation was corrupted (missing torch.utils module); resolved by reinstalling CPU-only torch after clearing pip cache to reclaim disk space
- No disk space initially for full CUDA torch reinstall; CPU-only variant (much smaller) sufficient for test execution

## Next Phase Readiness
- All 4 optimizations confirmed working with full test suite (540 passed, 1 skipped)
- Pipeline ready for production evaluation on runpod
- No remaining `for b_idx` loops in metric storage section
- No `.cpu().numpy()` calls inside the step loop

---
*Phase: quick-4*
*Completed: 2026-03-03*
