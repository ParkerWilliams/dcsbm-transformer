---
plan: 13-01
phase: 13-evaluation-enrichment
status: complete
started: 2026-02-26
completed: 2026-02-26
duration: ~5min
---

# Plan 13-01: Precision-Recall Curves, AUPRC Computation, and Report Integration

## What was built
- PR curve analysis module (`src/analysis/pr_curves.py`) mirroring AUROC pipeline
- AUPRC computed per metric per lookback distance using sklearn
- Score direction auto-detection via AUROC check (handles both higher/lower = violation)
- PR curve visualization (`src/visualization/pr_curves.py`) with no-skill baseline
- Collapsible HTML report section with AUPRC summary table
- Schema validation for pr_curves, calibration, and svd_benchmark blocks (all Phase 13)
- Render pipeline integration for all Phase 13 figure types
- 9 tests covering core analysis, edge cases, and integration

## Key files
- `src/analysis/pr_curves.py` - Core PR analysis
- `src/visualization/pr_curves.py` - AUPRC vs lookback plots
- `src/visualization/render.py` - Added PR curve render hook
- `src/reporting/single.py` - Added PR/calibration/SVD figure collection + template vars
- `src/reporting/templates/single_report.html` - Added all Phase 13 collapsible sections
- `src/results/schema.py` - Added all Phase 13 schema validation blocks
- `tests/test_pr_curves.py` - 9 tests

## Deviations
- Added all three Phase 13 schema validation blocks and HTML template sections in Plan 01 rather than splitting across plans, since the template and schema are better edited once

## Self-Check: PASSED
- [x] PR curves computed per metric per lookback using same event extraction as AUROC
- [x] Score direction handling works (tested with reversed metrics)
- [x] Collapsible sections render in HTML report
- [x] Schema validation backward-compatible
- [x] All 27 tests pass (9 new + 18 existing reporting)
