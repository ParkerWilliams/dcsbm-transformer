---
phase: 12-null-model-baseline
plan: 02
subsystem: analysis
tags: [null-model, mann-whitney, cohens-d, holm-bonferroni, marchenko-pastur, null-overlay, mp-histogram, html-report]

# Dependency graph
requires:
  - phase: 12-null-model-baseline
    provides: "Null walk generator, position-matched drift extraction, MP KS test (Plan 12-01)"
  - phase: 07-auroc-horizon
    provides: "AUROC analysis and statistical controls (holm_bonferroni, cohens_d)"
  - phase: 08-visualization
    provides: "Event-aligned plots, render orchestrator, style utilities"
  - phase: 09-reporting
    provides: "HTML report template and generation pipeline"
provides:
  - "compare_null_vs_violation() with Mann-Whitney U, Cohen's d, and Holm-Bonferroni correction"
  - "run_null_analysis() orchestrator for full null model pipeline"
  - "Null overlay visualization (gray CI band + median line on event-aligned plots)"
  - "MP histogram with theoretical density curve overlay and KS annotation"
  - "Optional null_model block in result.json schema validation"
  - "Null Model Baseline section in HTML report with statistical summary table"
affects: [12-null-model-baseline, reporting, visualization]

# Tech tracking
tech-stack:
  added: [scipy.stats.mannwhitneyu]
  patterns: [separate-hb-family, null-overlay-on-existing-plots, optional-schema-block]

key-files:
  created:
    - src/visualization/null_overlay.py
    - src/visualization/mp_histogram.py
  modified:
    - src/analysis/null_model.py
    - src/visualization/render.py
    - src/results/schema.py
    - src/reporting/single.py
    - src/reporting/templates/single_report.html
    - tests/test_null_model.py

key-decisions:
  - "Holm-Bonferroni applied as SEPARATE family across null model lookback distances only, not mixed with primary metrics"
  - "signal_exceeds_noise requires BOTH statistical rejection AND Cohen's d >= 0.5 (matching pre-registration Gate 2)"
  - "Null overlay uses gray colors (0.7 band, 0.5 median) to visually communicate noise floor"
  - "Schema validation is backward compatible -- null_model block optional, validated only when present"

patterns-established:
  - "Optional schema block pattern: validate sub-structure only when block exists, no errors when absent"
  - "Overlay visualization pattern: render base plot then add overlays from separate module"

requirements-completed: [NULL-02, NULL-04]

# Metrics
duration: 8min
completed: 2026-02-26
---

# Phase 12 Plan 02: Null Model Statistical Comparison Summary

**Mann-Whitney U comparison with Holm-Bonferroni correction, null overlay visualization, MP histogram, and HTML report integration for null model baseline**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-26T17:44:26Z
- **Completed:** 2026-02-26T17:52:37Z
- **Tasks:** 3
- **Files modified:** 8

## Accomplishments
- compare_null_vs_violation() computes per-lookback Mann-Whitney U, Cohen's d, with Holm-Bonferroni correction as a separate family (not mixed with primary metrics)
- run_null_analysis() orchestrator chains: null walk gen -> model eval -> drift extraction -> MW-U comparison -> Holm-Bonferroni -> MP KS test -> result.json-ready dict
- Null overlay visualization renders gray 95% CI band and solid median line on event-aligned plots
- MP histogram overlays theoretical Marchenko-Pastur density on empirical SV histogram with KS annotation
- HTML report includes full Null Model Baseline section (statistical summary table, MP subsection, figure slots)
- validate_result() accepts optional null_model block without requiring it (backward compatible)
- All 363 tests pass with 5 new statistical comparison tests; no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Statistical comparison functions and orchestrator** - `ceba6a1e` (feat)
2. **Task 2: Null overlay, MP histogram, and render integration** - `aa82d933` (feat)
3. **Task 3: Result schema and HTML report integration** - `3aeae221` (feat)

## Files Created/Modified
- `src/analysis/null_model.py` - Added compare_null_vs_violation(), run_null_analysis() orchestrator, new imports
- `src/visualization/null_overlay.py` - NEW: compute_null_distribution_stats(), plot_event_aligned_with_null()
- `src/visualization/mp_histogram.py` - NEW: plot_mp_histogram() with MP density overlay and KS annotation
- `src/visualization/render.py` - Null overlay and MP histogram integration in render_all()
- `src/results/schema.py` - Optional null_model block validation (config, by_lookback, aggregate)
- `src/reporting/single.py` - Null model data extraction, figure categorization, template variables
- `src/reporting/templates/single_report.html` - Null Model Baseline section with tables and figure slots
- `tests/test_null_model.py` - 5 new tests (17-21) for compare_null_vs_violation()

## Decisions Made
- **Separate Holm-Bonferroni family:** Null model MW-U p-values corrected across lookback distances ONLY, independent of primary metrics family per CONTEXT.md locked decision
- **signal_exceeds_noise criteria:** Requires both statistical rejection (adjusted p <= alpha) AND Cohen's d >= 0.5, matching pre-registration Gate 2 threshold
- **Gray color scheme for null overlay:** NULL_BAND_COLOR=(0.7,0.7,0.7,0.3), NULL_MEDIAN_COLOR=(0.5,0.5,0.5,1.0) per CONTEXT.md locked decision
- **Backward-compatible schema:** null_model block validated only when present; absence produces no errors

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 12 complete: null walk generation, statistical comparison, MP reference, visualization, and reporting all integrated
- run_null_analysis() can be called on any existing experiment with a trained model + graph
- Full null model pipeline stores results in result.json["null_model"] block and saves null_token_metrics.npz
- Render orchestrator auto-generates null overlay and MP histogram figures when null data present
- HTML report conditionally shows Null Model Baseline section with full statistical table

## Self-Check: PASSED

- All 8 created/modified files exist
- All 3 task commits verified (ceba6a1e, aa82d933, 3aeae221)
- 363/363 full suite tests pass, no regressions
- 21/21 null model tests pass (16 existing + 5 new)

---
*Phase: 12-null-model-baseline*
*Completed: 2026-02-26*
