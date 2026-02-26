---
phase: 09-reporting-and-math-verification
plan: 01
subsystem: reporting
tags: [jinja2, html, base64, reporting, reproduction]

# Dependency graph
requires:
  - phase: 08-visualization
    provides: "render.py load_result_data, figure generation pipeline"
  - phase: 01-foundation
    provides: "git_hash.py for code provenance tracking"
provides:
  - "embed_figure() base64 embedding utility for PNG/SVG/JPEG"
  - "build_reproduction_block() from result.json metadata"
  - "generate_single_report() self-contained HTML report generator"
  - "Jinja2 single_report.html template with academic styling"
affects: [09-02-comparison-report]

# Tech tracking
tech-stack:
  added: [jinja2-templates]
  patterns: [base64-data-uri-embedding, structured-config-tables, reproduction-block]

key-files:
  created:
    - src/reporting/__init__.py
    - src/reporting/embed.py
    - src/reporting/reproduction.py
    - src/reporting/single.py
    - src/reporting/templates/single_report.html
    - tests/test_reporting.py

key-decisions:
  - "Figure categorization by filename prefix (training_curves, confusion_matrix, auroc_*, event_aligned_*, distribution_*)"
  - "Config tables structured as Model/Training/Data categories with N/A defaults for missing params"
  - "Statistical tests extracted from metrics.statistical_controls.headline_comparison.primary_metrics"

patterns-established:
  - "Jinja2 FileSystemLoader from src/reporting/templates/ directory"
  - "embed_figure returns empty string on missing files (graceful degradation)"
  - "Reproduction block strips -dirty suffix from git hash for checkout command"

requirements-completed: [REPT-01, REPT-03]

# Metrics
duration: 3min
completed: 2026-02-26
---

# Phase 9 Plan 01: Single-Experiment Report Summary

**Self-contained HTML report generator with base64 figure embedding, structured config tables, and copy-pasteable reproduction block using Jinja2 templates**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-26T02:27:31Z
- **Completed:** 2026-02-26T02:31:26Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Base64 figure embedding utility supporting PNG, SVG, JPEG with graceful fallback for missing files
- Reproduction block builder extracting runtime git hash and building checkout/run commands with dirty-repo warnings
- Full single-experiment HTML report generator with academic serif styling, structured config tables, statistical test display, and predictive horizon summary
- Jinja2 template with all required sections, "Not available" placeholders for missing data, and clipboard copy buttons
- 10 passing tests covering embed, reproduction, and full report generation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create embed, reproduction, and report module scaffolding** - `2710e612` (feat)
2. **Task 2: Implement single-experiment report generator with tests** - `785e3438` (feat)

## Files Created/Modified
- `src/reporting/__init__.py` - Package exports for embed_figure, build_reproduction_block, generate_single_report
- `src/reporting/embed.py` - Base64 data URI encoding for PNG/SVG/JPEG figure files
- `src/reporting/reproduction.py` - Reproduction block builder from result.json metadata
- `src/reporting/single.py` - Single-experiment HTML report generator with figure collection and config table building
- `src/reporting/templates/single_report.html` - Jinja2 template with academic styling, all sections, copy buttons
- `tests/test_reporting.py` - 10 tests covering embed, reproduction, and report generation

## Decisions Made
- Figure categorization by filename prefix pattern (training_curves, confusion_matrix, auroc_*, event_aligned_*, distribution_*) for automatic template slot assignment
- Config tables organized as Model (d_model, n_layers, n_heads, vocab_size), Training (lr, epochs, batch_size, w, seed), Data (n, K, p_in, p_out, walk_length, corpus_size) with N/A defaults
- Statistical tests extracted from nested metrics.statistical_controls.headline_comparison.primary_metrics path
- Predictive horizon summary computes best metric per r-value from auroc_by_lookback arrays

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- embed.py and reproduction.py are reusable for Plan 02 (comparison report)
- Template patterns established for Jinja2 FileSystemLoader approach
- All 10 tests pass, module exports verified

## Self-Check: PASSED

All 6 created files verified on disk. Both task commits (2710e612, 785e3438) verified in git history.

---
*Phase: 09-reporting-and-math-verification*
*Completed: 2026-02-26*
