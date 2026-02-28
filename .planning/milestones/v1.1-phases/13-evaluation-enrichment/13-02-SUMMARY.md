---
plan: 13-02
phase: 13-evaluation-enrichment
status: complete
started: 2026-02-26
completed: 2026-02-26
duration: ~5min
---

# Plan 13-02: Calibration Diagnostics with Reliability Diagrams and ECE

## What was built
- Calibration analysis module (`src/analysis/calibration.py`) mirroring AUROC/PR pipeline
- Empirical CDF (rank-based) pseudo-probability conversion via `metric_to_pseudo_probability`
- ECE computation with weighted bin-level absolute calibration error
- `compute_calibration_at_lookback` with score direction detection (AUROC check)
- `run_calibration_analysis` orchestrator: extract events, stratify by r, compute ECE at each lookback j=1..r
- Reliability diagram visualization (`src/visualization/calibration.py`) with perfect-calibration diagonal
- Histogram of predicted probabilities below reliability diagram
- Render pipeline integration for calibration_* figures
- 12 tests covering pseudo-probability conversion, ECE computation, calibration analysis, and pipeline structure

## Key files
- `src/analysis/calibration.py` - Core calibration analysis with ECE
- `src/visualization/calibration.py` - Reliability diagram with histogram
- `src/visualization/render.py` - Added calibration render hook
- `tests/test_calibration.py` - 12 tests

## Deviations
- Template, schema, and reporting integration already done in Plan 01 (noted in 13-01-SUMMARY.md)
- Omitted optional `plot_ece_heatmap` function since the HTML template ECE summary table serves the same purpose

## Self-Check: PASSED
- [x] compute_calibration returns ECE, fraction_of_positives, mean_predicted_value, bin_counts
- [x] run_calibration_analysis produces nested dict with ece_by_lookback per metric
- [x] Reliability diagram shows lookback distances as colored lines plus diagonal
- [x] Histogram of predicted probabilities below each diagram
- [x] ECE summary table renders in HTML collapsible section (via Plan 01 template)
- [x] result.json calibration block validates through schema.py (via Plan 01)
- [x] All 39 tests pass (12 new + 27 existing)
