---
phase: 20-auroc-predictive-horizon
plan: 02
subsystem: testing
tags: [auroc, horizon, event-extraction, contamination-filter, ast-audit]

# Dependency graph
requires:
  - phase: 18-behavioral-classification
    provides: "4-class RuleOutcome enum and behavioral classification verified correct"
  - phase: 19-svd-metric-extraction
    provides: "SVD metric formulas verified, float32 storage confirmed"
provides:
  - "Cross-path audit confirming all AUROC consumers delegate to auroc_horizon.py"
  - "0.75 threshold consistency verified across all analysis and display paths"
  - "Event extraction boundary tests with contamination filter asymmetry verification"
  - "Cross-module seam test verifying behavioral.py -> event_extraction.py contract"
affects: [20-auroc-predictive-horizon]

# Tech tracking
tech-stack:
  added: []
  patterns: [ast-based-import-audit, living-regression-test, asymmetric-contamination-verification]

key-files:
  created:
    - tests/audit/test_horizon_consistency.py
    - tests/audit/test_event_extraction.py
  modified: []

key-decisions:
  - "AST-based import verification avoids brittle string matching for code-path audits"
  - "Living regression test catalogs all 0.75 occurrences in src/ to detect future drift"
  - "null_model.py confirmed to NOT compute AUROC -- uses Mann-Whitney U for a different statistical question"

patterns-established:
  - "Code-path audit pattern: use ast.parse + ImportFrom node inspection to verify module dependencies"
  - "Living catalog test: enumerate all occurrences of a critical constant and flag uncataloged additions"

requirements-completed: [AUROC-03, AUROC-04]

# Metrics
duration: 7min
completed: 2026-03-06
---

# Phase 20 Plan 02: Cross-Path Horizon Consistency & Event Extraction Boundary Audit Summary

**28 audit tests confirming all AUROC consumers delegate to auroc_horizon.py, 0.75 threshold is consistent across all code paths, and event extraction correctly filters to FOLLOWED/VIOLATED with asymmetric contamination logic**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-06T21:04:08Z
- **Completed:** 2026-03-06T21:12:07Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- All AUROC consumers (statistical_controls, spectrum, pr_curves, calibration) verified importing from auroc_horizon.py via AST inspection -- no duplicated implementations
- null_model.py confirmed to NOT compute AUROC (uses Mann-Whitney U for drift comparison, a different statistical question)
- signal_concentration.py confirmed to receive pre-computed AUROC values only (no computation)
- 0.75 threshold verified consistent across compute_predictive_horizon, run_auroc_analysis, visualization/auroc.py, visualization/heatmap.py, and reporting/single.py
- Living regression test catalogs all 0.75 occurrences in src/ and flags any uncataloged additions
- extract_events correctly yields only FOLLOWED/VIOLATED events (UNCONSTRAINED/PENDING filtered)
- is_first_violation correctly identifies the first violation per walk via failure_index match
- Contamination filter asymmetry verified: violations contaminate, FOLLOWED does NOT
- Cross-module seam verifies resolution_step indexing: outcome_idx = resolution_step - 1

## Task Commits

Each task was committed atomically:

1. **Task 1: Cross-path horizon consistency audit (AUROC-03)** - `e910df7` (test)
2. **Task 2: Event extraction boundary audit (AUROC-04)** - `23c4785` (test)

## Files Created/Modified
- `tests/audit/test_horizon_consistency.py` - 12 tests: AST-based import audit + threshold consistency + 0.75 catalog regression test
- `tests/audit/test_event_extraction.py` - 16 tests: outcome filtering + is_first_violation + contamination asymmetry + cross-module seam

## Decisions Made
- Used AST-based import verification (ast.parse + ImportFrom nodes) instead of string matching for robustness against comment/docstring false positives
- Added living regression test for 0.75 occurrences to catch future threshold drift automatically
- Confirmed null_model.py architecture: Mann-Whitney U for drift comparison is intentionally different from AUROC predictive horizon

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- AUROC-03 and AUROC-04 requirements are fully covered
- Combined with Plan 01 (AUROC-01 and AUROC-02), the full AUROC pipeline audit is complete
- All 28 new audit tests pass alongside existing 705 tests (no regressions)

## Self-Check: PASSED

- FOUND: tests/audit/test_horizon_consistency.py
- FOUND: tests/audit/test_event_extraction.py
- FOUND: 20-02-SUMMARY.md
- FOUND: commit e910df7
- FOUND: commit 23c4785

---
*Phase: 20-auroc-predictive-horizon*
*Completed: 2026-03-06*
