---
phase: 20-auroc-predictive-horizon
verified: 2026-03-06T22:00:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 20: AUROC Predictive Horizon Verification Report

**Phase Goal:** The AUROC predictive horizon pipeline is verified correct from event extraction through lookback indexing to horizon determination
**Verified:** 2026-03-06T22:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | auroc_from_groups matches sklearn.metrics.roc_auc_score on identical inputs | VERIFIED | 4 test cases (overlapping, separated, identical, reversed) all pass within 1e-10 |
| 2 | auroc_from_groups matches Mann-Whitney U/(n1*n0) from scipy.stats.mannwhitneyu | VERIFIED | Three-way comparison test confirms all three methods agree within 1e-10 |
| 3 | auroc_from_groups returns NaN for empty groups, 1.0 for perfect separation, ~0.5 for identical distributions | VERIFIED | Edge case tests pass: empty->NaN, [10,11,12] vs [1,2,3]->1.0, [5,5,5] vs [5,5,5]->0.5 |
| 4 | auroc_from_groups handles tied values correctly via midrank | VERIFIED | Tied values test confirms AUROC=0.5 and scipy.rankdata assigns midrank 3.5 to all 6 identical values |
| 5 | Lookback j=1 retrieves metric_array[walk_idx, resolution_step - 1] | VERIFIED | Planted-signal tests confirm j=1 retrieves index resolution_step-1 with perfect separation; j=2 yields chance |
| 6 | Planted signal at known position is retrieved by exactly the correct lookback j, not j+/-1 | VERIFIED | Fence-post sensitivity test: shifting signal from col 7 to col 6 moves AUROC=1.0 from j=1 to j=2 |
| 7 | metric_array bounds (max_steps-1 columns) are respected -- no IndexError for edge positions | VERIFIED | Out-of-bounds test: resolution_step=20 with 19-column array yields NaN at j=1, not IndexError |
| 8 | All code paths computing AUROC delegate to auroc_from_groups from auroc_horizon.py | VERIFIED | AST import inspection confirms statistical_controls, spectrum, pr_curves, calibration all import from auroc_horizon.py |
| 9 | The 0.75 horizon threshold is consistently applied across all analysis and display paths | VERIFIED | Default threshold confirmed via inspect.signature on run_auroc_analysis; AST check on visualization/auroc.py and heatmap.py; living regression catalog finds no uncataloged 0.75 occurrences |
| 10 | extract_events only yields FOLLOWED and VIOLATED events, never UNCONSTRAINED or PENDING | VERIFIED | Test exhausts all 4 RuleOutcome values; PENDING and UNCONSTRAINED produce zero events |
| 11 | Contamination filter excludes events whose encounter overlaps a prior VIOLATION window, but NOT a prior FOLLOWED window | VERIFIED | Asymmetric contamination tests: violation->nearby encounter excluded; followed->nearby encounter NOT excluded |

**Score:** 11/11 truths verified

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/audit/test_auroc_computation.py` | AUROC formula audit tests (AUROC-01) | VERIFIED | 14 test methods; imports `auroc_from_groups`; contains `roc_auc_score`; all pass |
| `tests/audit/test_lookback_indexing.py` | Lookback fence-post audit tests (AUROC-02) | VERIFIED | 15 test methods; contains planted value `999.0`; all pass |
| `tests/audit/test_horizon_consistency.py` | Cross-path horizon consistency audit (AUROC-03) | VERIFIED | 12 test methods; contains `horizon_threshold`; all pass |
| `tests/audit/test_event_extraction.py` | Event extraction boundary audit (AUROC-04) | VERIFIED | 16 test methods; contains `filter_contaminated_events`; all pass |

All artifacts: exist, are substantive (14/15/12/16 test methods respectively, with mathematical reasoning comments), and are wired to production source modules via direct imports.

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/audit/test_auroc_computation.py` | `src/analysis/auroc_horizon.py:auroc_from_groups` | `from src.analysis.auroc_horizon import auroc_from_groups` | WIRED | Import at line 13; function called in every test class |
| `tests/audit/test_lookback_indexing.py` | `src/analysis/auroc_horizon.py:compute_auroc_curve` | `from src.analysis.auroc_horizon import compute_auroc_curve, compute_predictive_horizon` | WIRED | Import at line 12; planted-signal tests call compute_auroc_curve directly |
| `tests/audit/test_horizon_consistency.py` | `src/analysis/auroc_horizon.py` | AST ImportFrom inspection + `inspect.signature` | WIRED | Tests walk AST nodes of statistical_controls, spectrum, pr_curves, calibration; all confirmed |
| `tests/audit/test_event_extraction.py` | `src/analysis/event_extraction.py:extract_events` | `from src.analysis.event_extraction import extract_events, filter_contaminated_events` | WIRED | Import at lines 13-17; all test classes call extract_events or filter_contaminated_events |

Additional production wiring confirmed:

- `src/analysis/statistical_controls.py` imports `auroc_from_groups`, `compute_auroc_curve`, `compute_predictive_horizon` from `src.analysis.auroc_horizon` (line 23)
- `src/analysis/spectrum.py` imports `auroc_from_groups`, `compute_auroc_curve` from `src.analysis.auroc_horizon` (line 16)
- `src/analysis/pr_curves.py` imports `auroc_from_groups` from `src.analysis.auroc_horizon` (line 11)
- `src/analysis/calibration.py` imports `auroc_from_groups` from `src.analysis.auroc_horizon` (line 12)
- `src/analysis/null_model.py` contains no reference to any AUROC computation function (confirmed by test and grep)
- `src/analysis/signal_concentration.py` contains no reference to any AUROC computation function (confirmed by test)

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| AUROC-01 | 20-01-PLAN.md | AUROC computation from violation/control groups uses correct rank-based probability P(X_violated > X_followed) | SATISFIED | 14 passing tests in test_auroc_computation.py: sklearn match within 1e-10, Mann-Whitney U equivalence within 1e-10, analytic Gaussian within 0.02 of theoretical, all edge cases handled |
| AUROC-02 | 20-01-PLAN.md | Lookback distance j correctly indexes metric values at step (t-j) relative to resolution step t | SATISFIED | 15 passing tests in test_lookback_indexing.py: planted-signal fence-post tests confirm j=1->index resolution_step-1, fence-post sensitivity catches +/-1 errors, boundary conditions verified |
| AUROC-03 | 20-02-PLAN.md | Predictive horizon definition (max j where AUROC > 0.75) is consistently applied across all analysis paths | SATISFIED | 12 passing tests in test_horizon_consistency.py: all 4 AUROC consumers use auroc_horizon.py (AST-verified), null_model.py confirmed AUROC-free, 0.75 threshold consistent via inspect.signature and AST defaults, living catalog finds no uncataloged 0.75 occurrences |
| AUROC-04 | 20-02-PLAN.md | Event extraction correctly identifies resolution steps from behavioral labels | SATISFIED | 16 passing tests in test_event_extraction.py: only FOLLOWED/VIOLATED yielded, resolution_step=encounter_step+r verified, is_first_violation per-walk tracking verified, contamination asymmetry confirmed, cross-module seam passes |

No orphaned requirements: all AUROC-01 through AUROC-04 are claimed by plans 20-01 and 20-02, covered by tests, and marked Complete in REQUIREMENTS.md traceability table.

---

## Anti-Patterns Found

No anti-patterns detected. Scan results:

- No TODO/FIXME/HACK/PLACEHOLDER comments in any of the 4 test files
- No empty implementations (return null/return {}/return [])
- No stub handlers (console.log-only, preventDefault-only)
- No placeholder assertions
- All 57 test methods contain substantive mathematical assertions with documented reasoning

---

## Test Execution Results

**Phase 20 audit tests:** 57 passed in 8.04s (0 failures, 0 errors)

**Full regression suite:** 705 passed, 1 skipped, 6 warnings in 81.86s (0 failures)

The 6 warnings are pre-existing RuntimeWarnings from `nanmean` and `nanvar` on empty slices in visualization tests; unrelated to Phase 20.

---

## Commit Verification

All commits documented in SUMMARY files exist and are valid in git history:

| Commit | Description | Status |
|--------|-------------|--------|
| `ffa4efb` | test(20-01): audit AUROC formula against sklearn, Mann-Whitney U, and analytic distributions | VERIFIED |
| `08f86d6` | test(20-01): audit lookback fence-post indexing and predictive horizon logic | VERIFIED |
| `e910df7` | test(20-02): cross-path horizon consistency audit (AUROC-03) | VERIFIED |
| `23c4785` | test(20-02): event extraction boundary audit (AUROC-04) | VERIFIED |

---

## Human Verification Required

None. All phase 20 goals are verifiable programmatically:

- Mathematical correctness: verified by oracle comparison (sklearn, scipy)
- Indexing correctness: verified by planted-signal construction
- Code-path consistency: verified by AST inspection
- Event extraction: verified by synthetic numpy arrays with known outcomes

---

## Gaps Summary

No gaps. All 11 truths verified, all 4 artifacts pass all three levels (exists, substantive, wired), all 4 key links confirmed wired, all 4 requirements satisfied with evidence, no anti-patterns found, no regressions introduced.

---

_Verified: 2026-03-06T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
