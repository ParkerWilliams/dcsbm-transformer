---
phase: 07-predictive-horizon-and-statistical-analysis
plan: 02
subsystem: analysis
tags: [holm-bonferroni, bootstrap-ci, cohens-d, correlation-matrix, redundancy, metric-ranking, headline-comparison]

requires:
  - phase: 07-predictive-horizon-and-statistical-analysis
    provides: "AUROC analysis pipeline, event extraction, PRIMARY_METRICS frozenset"
provides:
  - "Holm-Bonferroni step-down correction for 5 pre-registered primary metrics"
  - "BCa bootstrap confidence intervals on AUROC with percentile fallback"
  - "Cohen's d effect sizes per metric per lookback distance"
  - "Two correlation matrices: measurement redundancy and predictive redundancy"
  - "Redundancy flagging at |r| > 0.9 threshold"
  - "Metric importance ranking per layer with redundancy annotations"
  - "Headline QK^T vs AVWo predictive horizon comparison"
  - "apply_statistical_controls top-level orchestrator for JSON-serializable output"
affects: [phase-08-results, phase-09-visualization]

tech-stack:
  added: []
  patterns: [bca-bootstrap-fallback, holm-bonferroni-step-down, correlation-redundancy-flagging, per-layer-ranking]

key-files:
  created:
    - src/analysis/statistical_controls.py
    - tests/test_statistical_controls.py
  modified:
    - src/analysis/__init__.py

key-decisions:
  - "BCa bootstrap with automatic fallback to percentile method when BCa produces NaN or raises"
  - "Holm-Bonferroni applies to exactly 5 primary metrics, correction factor at most 5 (not 21)"
  - "Cohen's d returns NaN for pooled_std < 1e-12 or insufficient samples (< 2 per group)"
  - "Measurement correlation uses metric values at resolution_step - 1 for event positions"
  - "Predictive correlation replaces NaN AUROC values with 0.5 for correlation computation"
  - "WvWo metrics excluded from primary metrics and Holm-Bonferroni correction (static per-checkpoint reference)"
  - "Headline comparison identifies QK^T/AVWo metrics by key prefix (qkt./avwo.) with primary metric filtering"

patterns-established:
  - "BCa-then-percentile bootstrap fallback for degenerate distributions near AUROC boundaries"
  - "Per-layer metric ranking with redundancy annotation from measurement correlation matrix"
  - "Vectorized AUROC statistic function for scipy.stats.bootstrap (axis parameter, concatenate+rankdata)"
  - "Structured result dict extending auroc_results with statistical controls for JSON serialization"

requirements-completed: [STAT-01, STAT-02, STAT-03, STAT-04, STAT-05]

duration: 6min
completed: 2026-02-26
---

# Phase 7 Plan 02: Statistical Controls Summary

**Holm-Bonferroni correction, BCa bootstrap CIs, Cohen's d effect sizes, correlation/redundancy analysis, metric ranking, and headline QK^T vs AVWo comparison**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-26T00:22:17Z
- **Completed:** 2026-02-26T00:28:36Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments
- Holm-Bonferroni step-down correction applying to exactly 5 pre-registered primary metrics (not 21), with correct monotonicity enforcement and clip-at-1.0
- BCa bootstrap confidence intervals on AUROC via scipy.stats.bootstrap with automatic percentile fallback when BCa produces NaN or raises (handles degenerate distributions near AUROC 0 and 1)
- Cohen's d effect sizes computed per metric per lookback distance with proper pooled standard deviation and NaN guards
- Two correlation matrices: measurement redundancy (raw values at event positions) and predictive redundancy (AUROC curves across lookback distances), both with |r| > 0.9 redundancy flagging
- Metric importance ranking per layer ordered by max AUROC descending, annotated with redundancy flags and primary/exploratory labels
- Headline QK^T vs AVWo comparison reporting max predictive horizon per r-value with gap and direction
- Full apply_statistical_controls orchestrator producing JSON-serializable output matching result.json schema
- 21 TDD tests covering all functions with synthetic data, 280 total tests passing

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: Failing tests for statistical controls** - `71047993` (test)
2. **Task 1 GREEN: Implement statistical controls module** - `768134c1` (feat)

## Files Created/Modified
- `src/analysis/statistical_controls.py` - Holm-Bonferroni, BCa bootstrap, Cohen's d, correlation matrices, metric ranking, headline comparison, apply_statistical_controls orchestrator
- `tests/test_statistical_controls.py` - 21 test cases covering all statistical control functions
- `src/analysis/__init__.py` - Updated public API exports for statistical controls

## Decisions Made
- BCa bootstrap with automatic fallback to percentile method when BCa produces NaN or raises -- handles degenerate AUROC distributions at boundaries (0 or 1)
- Holm-Bonferroni applies to exactly 5 primary metrics per CONTEXT.md locked decision, correction factor at most 5 (not 21 total metrics)
- Cohen's d returns NaN for pooled_std < 1e-12 (identical groups) or insufficient samples (< 2 per group)
- Measurement correlation uses metric values at resolution_step - 1 as representative event positions, ensuring consistent alignment across metrics
- Predictive correlation replaces NaN AUROC values with 0.5 before computing Pearson correlation, treating missing data as "no signal"
- WvWo metrics excluded from primary metrics and Holm-Bonferroni correction per CONTEXT.md -- they are static per-checkpoint reference metrics
- Headline comparison identifies QK^T vs AVWo metrics by key prefix (qkt./avwo.) and filters to primary metrics only

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Complete statistical analysis module ready for Phase 8 (results assembly)
- All analysis functions (event extraction, AUROC horizon, statistical controls) are independently importable and tested
- Output structure is JSON-serializable, matching result.json predictive_horizon schema
- 280 tests pass with no regressions across all phases

---
*Phase: 07-predictive-horizon-and-statistical-analysis*
*Completed: 2026-02-26*
