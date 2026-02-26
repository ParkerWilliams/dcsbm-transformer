---
phase: 13-evaluation-enrichment
type: verification
status: passed
verified: 2026-02-26
---

# Phase 13: Evaluation Enrichment -- Verification

## Phase Goal
> Violation prediction quality is assessed beyond AUROC with precision-recall curves and calibration diagnostics, and SVD computational cost is benchmarked with cheaper approximation candidates identified

## Requirement Cross-Reference

| Req ID | Requirement | Status | Evidence |
|--------|-------------|--------|----------|
| PRCL-01 | PR curves and AUPRC per metric per lookback distance | PASS | `src/analysis/pr_curves.py`: `run_pr_analysis()` computes AUPRC via sklearn `average_precision_score` per metric per lookback j=1..r, using same `extract_events`/`stratify_by_r` as AUROC |
| PRCL-02 | Reliability diagrams with ECE for violation prediction | PASS | `src/analysis/calibration.py`: `run_calibration_analysis()` computes ECE via rank-based pseudo-probabilities and sklearn `calibration_curve`; `src/visualization/calibration.py`: `plot_reliability_diagram()` shows per-lookback colored lines with perfect-calibration diagonal and histogram |
| PRCL-03 | PR curves and reliability diagrams in HTML reports | PASS | `src/reporting/templates/single_report.html`: collapsible PR Curves section (lines 269-319) with AUPRC summary table, collapsible Calibration Diagnostics section (lines 321-369) with ECE summary table; `src/reporting/single.py`: passes all data to template; `src/visualization/render.py`: generates pr_curve_* and calibration_* figures |
| OVHD-01 | Wall-clock SVD cost benchmarked by target and matrix dimension | PASS | `src/analysis/svd_benchmark.py`: `_time_svd_method()` uses CUDA events with warmup (n_warmup=5, n_timed=20) for GPU, perf_counter for CPU; `benchmark_svd_for_target()` benchmarks at actual matrix dimensions from config |
| OVHD-02 | Full vs randomized vs values-only SVD with accuracy-cost tradeoff | PASS | `src/analysis/svd_benchmark.py`: `_full_svd`, `_randomized_svd`, `_values_only_svd` wrappers; `_compare_accuracy()` reports relative Frobenius error and SV correlation; `src/visualization/svd_benchmark.py`: `plot_svd_accuracy_tradeoff()` scatter plot |
| OVHD-03 | Cost summary table in HTML reports | PASS | `src/reporting/templates/single_report.html`: collapsible SVD Computational Cost section (lines 371-424) with cost summary table (target, matrix size, timing for all 3 methods, Frobenius error, SV correlation) and total/fastest method summary |

## Success Criteria Verification

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | PR curves and AUPRC computed per metric per lookback using same event extraction | PASS | `pr_curves.py` imports `extract_events`, `filter_contaminated_events`, `stratify_by_r` from `event_extraction.py`; tested in `test_pr_curves.py` (9 tests) |
| 2 | Reliability diagrams with ECE generated for violation prediction | PASS | `calibration.py` computes ECE; `plot_reliability_diagram()` generates figure; tested in `test_calibration.py` (12 tests) |
| 3 | PR curves and reliability diagrams rendered in HTML reports | PASS | All three collapsible `<details>` sections render in template; `single.py` passes data; `render.py` generates figures |
| 4 | Wall-clock SVD cost benchmarked with CUDA events, all three methods compared | PASS | `_time_svd_method()` uses CUDA events for GPU, perf_counter for CPU; accuracy comparison via `_compare_accuracy()`; tested in `test_svd_benchmark.py` (17 tests) |
| 5 | Cost summary table in HTML reports | PASS | Template renders table with matrix size, timing, and accuracy; summary shows total cost and fastest method |

## Must-Haves from Plans

### Plan 13-01
- [x] PR curves computed per metric per lookback using same event extraction as AUROC
- [x] Score direction handling works (tested with reversed metrics)
- [x] Collapsible sections render in HTML report
- [x] Schema validation backward-compatible

### Plan 13-02
- [x] compute_calibration returns ECE, fraction_of_positives, mean_predicted_value, bin_counts
- [x] run_calibration_analysis produces nested dict with ece_by_lookback per metric
- [x] Reliability diagram shows lookback distances as colored lines plus diagonal
- [x] Histogram of predicted probabilities below each diagram
- [x] ECE summary table renders in HTML collapsible section
- [x] result.json calibration block validates through schema.py

### Plan 13-03
- [x] benchmark_svd_methods returns timing for full, randomized, and values-only SVD at actual matrix dimensions
- [x] Accuracy comparison reports relative Frobenius error and SV correlation
- [x] Cost summary table shows matrix size, time per step, and accuracy per target
- [x] Grouped bar chart shows targets on x-axis with SVD methods as colored groups
- [x] SVD benchmark results stored in result.json svd_benchmark block and rendered in HTML report

## Test Results

```
400 passed, 1 skipped, 6 warnings
```

Test files:
- `tests/test_pr_curves.py` - 9 tests (PR analysis core, edge cases, pipeline structure)
- `tests/test_calibration.py` - 12 tests (pseudo-probability, ECE, calibration analysis, pipeline)
- `tests/test_svd_benchmark.py` - 17 tests (SVD wrappers, timing, accuracy, orchestration)
- `tests/test_reporting.py` - 18 tests (existing, no regressions)

## Artifacts Created

| File | Purpose |
|------|---------|
| `src/analysis/pr_curves.py` | PR curve analysis with AUPRC computation |
| `src/analysis/calibration.py` | Calibration diagnostics with ECE |
| `src/analysis/svd_benchmark.py` | SVD overhead benchmarking |
| `src/visualization/pr_curves.py` | AUPRC vs lookback plots |
| `src/visualization/calibration.py` | Reliability diagrams |
| `src/visualization/svd_benchmark.py` | Grouped bar charts and tradeoff scatter |
| `src/visualization/render.py` | Updated with PR, calibration, SVD benchmark hooks |
| `src/reporting/single.py` | Updated with all Phase 13 data passing |
| `src/reporting/templates/single_report.html` | Updated with 3 collapsible sections |
| `src/results/schema.py` | Updated with pr_curves, calibration, svd_benchmark validation |
| `tests/test_pr_curves.py` | PR analysis tests |
| `tests/test_calibration.py` | Calibration tests |
| `tests/test_svd_benchmark.py` | SVD benchmark tests |

## Verdict

**PASSED** -- All 6 requirements (PRCL-01, PRCL-02, PRCL-03, OVHD-01, OVHD-02, OVHD-03) verified against codebase. All 5 success criteria met. 400 tests pass with no regressions.
