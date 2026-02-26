# Plan 11-01 Summary: Pre-Registration Framework

**Phase:** 11-pre-registration-framework
**Plan:** 01
**Status:** Complete
**Duration:** ~15 min

## What Was Built

Established the pre-registration framework for v1.1 confirmatory analysis:

1. **Pre-registration document** (`docs/pre-registration.md`) specifying:
   - Primary hypothesis: Grassmannian distance of QK^T increases before rule violations
   - Primary metric: `qkt.layer_{N}.grassmannian_distance`
   - Alpha level: 0.05 with Holm-Bonferroni correction across all lookback distances
   - Three-outcome decision criterion: Confirm (Gate 1 + Gate 2), Inconclusive (Gate 1 only), Reject (Gate 1 fails)
   - Held-out protocol: 50/50 exploratory/confirmatory split
   - All secondary SVD metrics listed as exploratory (not used for decision)

2. **Held-out evaluation split** (`src/evaluation/split.py`):
   - `assign_split()`: Deterministic stratified 50/50 assignment using fixed seed (2026)
   - Stratified by violation status (equal proportions in each set)
   - 11 TDD test cases covering determinism, stratification, edge cases

3. **Pipeline integration**:
   - `save_evaluation_results()` accepts optional `split_labels` parameter
   - Split data stored in NPZ (integer encoding: 0=exploratory, 1=confirmatory)
   - Split metadata in result.json summary dict (seed, counts, violation breakdown)
   - Schema validation for `split_assignment` fields (backward compatible)

4. **Deviation log** (`docs/deviation-log.md`):
   - Template for recording changes to pre-registered plan
   - Cross-referenced from pre-registration document

## Key Files

### Created
- `docs/pre-registration.md` -- Pre-registration document (locked before confirmatory analysis)
- `docs/deviation-log.md` -- Deviation log template
- `src/evaluation/split.py` -- Held-out split assignment function
- `tests/test_split.py` -- 11 tests for split function

### Modified
- `src/evaluation/pipeline.py` -- Added split_labels parameter to save_evaluation_results()
- `src/results/schema.py` -- Added optional split_assignment validation

## Test Results

- 11/11 split tests pass
- 342/342 full suite tests pass (no regressions)

## Requirements Addressed

| Requirement | Status | Evidence |
|-------------|--------|----------|
| PREG-01 | Complete | docs/pre-registration.md committed with hypothesis, metric, alpha, correction, decision criterion |
| PREG-02 | Complete | src/evaluation/split.py + pipeline integration with result.json tagging |
| PREG-03 | Complete | docs/deviation-log.md committed and referenced from pre-registration |

## Deviations

None.

## Self-Check: PASSED

- [x] Pre-registration document exists with all required sections
- [x] Three-outcome decision criterion (Confirm/Inconclusive/Reject) defined
- [x] Held-out split is deterministic and stratified
- [x] Result.json and NPZ support split tagging
- [x] Deviation log exists and is cross-referenced
- [x] All 342 tests pass
- [x] No regressions
