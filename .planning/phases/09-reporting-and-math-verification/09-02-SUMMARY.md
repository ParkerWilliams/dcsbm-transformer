---
phase: 09-reporting-and-math-verification
plan: 02
subsystem: reporting
tags: [jinja2, html, comparison-report, sparkline, matplotlib, base64]

# Dependency graph
requires:
  - phase: 09-reporting-and-math-verification
    provides: "embed_figure, build_reproduction_block, single_report template patterns"
  - phase: 08-visualization
    provides: "render.py load_result_data, PALETTE colorblind-safe palette"
provides:
  - "generate_comparison_report() multi-experiment HTML report generator"
  - "compute_config_diff() flattened config comparison with differs flag"
  - "generate_sparkline() mini bar chart as base64 data URI"
  - "compute_verdict() auto-generated experiment winner summary"
  - "Jinja2 comparison_report.html template with sparklines, diff-highlight, reproduction blocks"
affects: [10-sweep-execution]

# Tech tracking
tech-stack:
  added: []
  patterns: [sparkline-bar-chart, config-diff-highlighting, verdict-computation, curve-overlay-grid]

key-files:
  created:
    - src/reporting/comparison.py
    - src/reporting/templates/comparison_report.html
  modified:
    - src/reporting/__init__.py
    - tests/test_reporting.py

key-decisions:
  - "Renamed metric row key to experiment_values to avoid Jinja2 dict.values() method collision"
  - "Sparkline implemented as horizontal bar chart using matplotlib with project colorblind-safe PALETTE"
  - "Verdict uses higher-is-better default, with loss/error keywords triggering lower-is-better comparison"
  - "Curve overlay grid uses side-by-side subplots with imshow of original figure PNGs"

patterns-established:
  - "Jinja2 bracket notation row['values'] for dict keys that collide with Python dict methods"
  - "generate_sparkline returns base64 data URI directly (no intermediate file)"
  - "_flatten_config with dot-separated keys for nested dict comparison"

requirements-completed: [REPT-02, REPT-03]

# Metrics
duration: 4min
completed: 2026-02-26
---

# Phase 9 Plan 02: Comparison Report Summary

**Multi-experiment comparison HTML report with sparkline bar charts, config diff highlighting, auto-generated verdict, and curve overlay grids**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T02:36:17Z
- **Completed:** 2026-02-26T02:40:17Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Multi-experiment comparison report generator with scalar metrics table, sparkline mini bar charts, and auto-generated verdict identifying winning experiment
- Config diff table comparing flattened dot-separated keys across experiments with CSS highlighting for differing rows
- Curve overlay grid building composite subplot figures from same-named PNGs across experiments
- Per-experiment reproduction blocks with copy buttons inherited from Plan 01 utilities
- 18 total tests passing (10 from Plan 01 + 8 new comparison tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement comparison report generator with sparklines and config diff** - `988e4164` (feat)
2. **Task 2: Add comparison report tests and update exports** - `9e730309` (feat)

## Files Created/Modified
- `src/reporting/comparison.py` - Multi-experiment comparison report generator with sparklines, config diff, verdict, curve overlays
- `src/reporting/templates/comparison_report.html` - Jinja2 template with academic styling, sparkline cells, diff-highlight CSS, verdict box
- `src/reporting/__init__.py` - Added generate_comparison_report export
- `tests/test_reporting.py` - Added 8 comparison report tests (flatten, diff, sparkline, verdict, full report, single experiment)

## Decisions Made
- Renamed metric row dict key from `values` to `experiment_values` to avoid Jinja2 resolving `row.values` as the Python dict `.values()` method instead of the key lookup
- Sparkline implemented as tiny horizontal bar chart via matplotlib (not SVG line chart) per user decision for "mini bar charts showing relative values"
- Verdict logic defaults to higher-is-better, except for metrics with "loss" or "error" in name which use lower-is-better
- Curve overlay uses imshow of original PNG files in side-by-side subplots rather than re-rendering from data

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Jinja2 dict.values() method collision in template**
- **Found during:** Task 2 (test execution)
- **Issue:** Template `row.values` resolved to dict `.values()` method instead of the `values` key, causing TypeError
- **Fix:** Renamed key to `experiment_values` in Python; used bracket notation `row["values"]` for config diff rows
- **Files modified:** src/reporting/comparison.py, src/reporting/templates/comparison_report.html
- **Verification:** All 18 tests pass
- **Committed in:** 9e730309 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Essential fix for template rendering correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed template collision above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All three Phase 9 plans (single report, comparison report, math PDF) complete
- Reporting module provides generate_single_report, generate_comparison_report, generate_math_pdf
- Ready for Phase 10 sweep execution with full reporting pipeline

## Self-Check: PASSED

All 5 files verified on disk. Both task commits (988e4164, 9e730309) verified in git history.

---
*Phase: 09-reporting-and-math-verification*
*Completed: 2026-02-26*
