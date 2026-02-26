---
phase: 07-predictive-horizon-and-statistical-analysis
plan: 01
subsystem: analysis
tags: [event-extraction, auroc, predictive-horizon, shuffle-controls, contamination-filter]

requires:
  - phase: 06-behavioral-evaluation-and-svd-collection
    provides: "EvaluationResult with SVD metrics, behavioral labels, and generated sequences"
  - phase: 06-behavioral-evaluation-and-svd-collection
    provides: "token_metrics.npz with target.layer.metric keyed arrays"
provides:
  - "AnalysisEvent dataclass for structured jumper encounter records"
  - "extract_events for deriving encounters from generated sequences"
  - "filter_contaminated_events with audit metrics and per-r breakdown"
  - "stratify_by_r for independent per-r-value analysis"
  - "auroc_from_groups via rank-based Mann-Whitney method"
  - "compute_auroc_curve at each lookback distance j=1..r"
  - "compute_predictive_horizon as furthest j where AUROC exceeds threshold"
  - "run_shuffle_control for positional artifact detection"
  - "run_auroc_analysis full pipeline orchestration"
  - "NPZ extension: generated array saved for standalone re-analysis"
affects: [phase-07-plan-02-statistical-controls, phase-08-results, phase-09-visualization]

tech-stack:
  added: []
  patterns: [rank-based-auroc, contamination-filter, event-stratification, shuffle-permutation-test]

key-files:
  created:
    - src/analysis/__init__.py
    - src/analysis/event_extraction.py
    - src/analysis/auroc_horizon.py
    - tests/test_event_extraction.py
    - tests/test_auroc_horizon.py
  modified:
    - src/evaluation/pipeline.py
    - tests/test_evaluation_pipeline.py

key-decisions:
  - "resolution_step = encounter_step + r; rule_outcome indexed at resolution_step - 1 (matches behavioral.py deadline convention)"
  - "Only violation events contaminate subsequent encounters; FOLLOWED events do not set last_violation_end"
  - "is_first_violation determined by matching failure_index[walk] == outcome_idx"
  - "PRIMARY_METRICS frozenset of 5 pre-registered metrics identified by target.metric_name pattern"
  - "Event count tiers: skip (0-1), low_n (2-4), moderate_n (5-9), full (10+) per RESEARCH.md"
  - "Shuffle control uses max AUROC across lookback distances for permutation comparison"

patterns-established:
  - "Event extraction from generated sequences via jumper_map lookup + rule_outcome cross-reference"
  - "Contamination filtering with per-walk violation window tracking and audit metrics"
  - "AUROC computation via scipy.stats.rankdata with NaN-filtered lookback values"
  - "Structured result dict matching result.json predictive_horizon schema"

requirements-completed: [PRED-01, PRED-02, PRED-03, PRED-04, PRED-05]

duration: 8min
completed: 2026-02-26
---

# Phase 7 Plan 01: Event Extraction and AUROC Horizon Summary

**Rank-based AUROC at each lookback distance with contamination filtering, predictive horizon detection, and shuffle controls for positional artifact validation**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-26T00:11:12Z
- **Completed:** 2026-02-26T00:19:33Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments
- Event extraction pipeline: scans generated sequences for jumper vertices, cross-references rule_outcome with proper indexing alignment (resolution_step - 1)
- Contamination filter: excludes encounters overlapping prior violation windows, only violations contaminate (not successes), with comprehensive audit metrics and >30% flagging
- AUROC computation via rank-based method (scipy.stats.rankdata) at each lookback j=1..r, NaN-safe with minimum event count enforcement
- Predictive horizon: furthest j where AUROC exceeds configurable threshold (default 0.75)
- Shuffle controls: 10,000 permutation tests flagging metrics where shuffled p95 AUROC > 0.6 (positional artifacts)
- Full pipeline orchestrator (run_auroc_analysis) with structured output matching result.json schema
- NPZ extension: generated array now saved for standalone re-analysis from saved files
- 23 TDD tests covering all functions with synthetic data

## Task Commits

Each task was committed atomically:

1. **Task 1: Event extraction with contamination filtering (TDD)** - `0ff5891b` (feat)
2. **Task 2: AUROC computation, predictive horizon, and shuffle controls (TDD)** - `388b1011` (feat)

## Files Created/Modified
- `src/analysis/__init__.py` - Public API for analysis module
- `src/analysis/event_extraction.py` - AnalysisEvent dataclass, extract_events, filter_contaminated_events, stratify_by_r
- `src/analysis/auroc_horizon.py` - auroc_from_groups, compute_auroc_curve, compute_predictive_horizon, run_shuffle_control, run_auroc_analysis
- `tests/test_event_extraction.py` - 9 tests for event extraction and contamination filtering
- `tests/test_auroc_horizon.py` - 14 tests for AUROC computation, horizon, shuffle, and integration
- `src/evaluation/pipeline.py` - Added generated array to NPZ output
- `tests/test_evaluation_pipeline.py` - Updated NPZ key convention test for new "generated" key

## Decisions Made
- resolution_step = encounter_step + r; rule_outcome indexed at resolution_step - 1 (matches behavioral.py's `if t + 1 == deadline` convention)
- Only violations contaminate subsequent encounters (FOLLOWED events do not update last_violation_end), per CONTEXT.md locked decision
- is_first_violation identified by matching failure_index[walk] to the outcome index, ensuring only the first violation per walk is marked
- PRIMARY_METRICS identified by stripping layer from key (target.layer_N.metric_name -> target.metric_name)
- Event count tiers from RESEARCH.md: skip (0-1 events), low_n (2-4, point estimate only), moderate_n (5-9, AUROC + effect size), full (10+, complete analysis with bootstrap)
- Shuffle control compares max AUROC across all lookback distances (not per-lookback), which naturally inflates with more lookbacks -- this is correct per-test design

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated NPZ key convention test for new "generated" key**
- **Found during:** Task 1 (NPZ extension)
- **Issue:** Existing test `test_npz_key_convention` validates all NPZ keys follow `target.layer_N.metric_name` pattern, but the new "generated" key doesn't match this pattern
- **Fix:** Added "generated" to the skip-list of non-metric keys alongside edge_valid, rule_outcome, etc.
- **Files modified:** tests/test_evaluation_pipeline.py
- **Verification:** All 259 tests pass
- **Committed in:** 0ff5891b (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor test update required by the NPZ extension. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Event extraction and AUROC pipeline ready for Phase 7 Plan 02 (statistical controls: BCa bootstrap, Holm-Bonferroni, Cohen's d, correlation matrices, metric ranking)
- Output structure matches result.json predictive_horizon schema for downstream phases
- All 259 tests pass with no regressions

---
*Phase: 07-predictive-horizon-and-statistical-analysis*
*Completed: 2026-02-26*
