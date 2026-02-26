---
phase: 07-predictive-horizon-and-statistical-analysis
verified: 2026-02-26T00:40:00Z
status: passed
score: 12/12 must-haves verified
re_verification: false
---

# Phase 7: Predictive Horizon and Statistical Analysis Verification Report

**Phase Goal:** For each SVD metric, the system measures how far in advance it can predict rule violations (AUROC at each lookback distance), with position-matched baselines and rigorous statistical controls
**Verified:** 2026-02-26T00:40:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (Plan 07-01)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Jumper encounters are correctly extracted from generated sequences with walk index, encounter step, resolution step, r value, and outcome | VERIFIED | `AnalysisEvent` dataclass has all 6 fields; `extract_events` scans generated array, cross-references `rule_outcome` at `resolution_step - 1`; 4 tests pass covering basic extraction, first-violation flag, empty case, and off-by-one alignment |
| 2 | Contamination filtering excludes encounters whose countdown window overlaps a preceding violation window, with exclusion count recorded | VERIFIED | `filter_contaminated_events` tracks `last_violation_end` per walk, excludes events where `encounter_step < last_violation_end`; audit dict records `total_encounters`, `excluded_encounters`, `exclusion_rate`, `flagged`, `per_r`; 4 tests pass including overlap, non-overlap, success-does-not-contaminate, and >30% flagging |
| 3 | AUROC is computed at each lookback distance j (1 to r) for each SVD metric, stratified by r value | VERIFIED | `compute_auroc_curve` iterates j=1..r, extracts metric values at `resolution_step - j`, calls `auroc_from_groups` (rank-based via `scipy.stats.rankdata`); returns shape `(r_value,)` array; `stratify_by_r` groups events by r_value; 5 tests cover shape, known signal, NaN handling, min events, and stratification |
| 4 | Predictive horizon (furthest j where AUROC > 0.75) is calculated per metric per r-value | VERIFIED | `compute_predictive_horizon` scans from largest j to smallest, returns largest j where `auroc_curve[j-1] > threshold`; returns 0 if none exceed; 2 tests: basic case (horizon=5) and none-exceeds case (horizon=0) |
| 5 | Shuffle controls flag metrics where permuted-label AUROC exceeds 0.6 | VERIFIED | `run_shuffle_control` performs 10,000 permutations, computes max AUROC per shuffle, sets `shuffle_flag = (p95 > 0.6)`; 2 tests: no-signal (flag=False) and positional artifact (flag=True) |
| 6 | Per-metric AUROC curves are stored in a structured results dict matching result.json schema | VERIFIED | `run_auroc_analysis` returns `{config, contamination_audit, by_r_value}`; `by_r_value[r]` contains `{n_violations, n_controls, event_tier, by_metric}`; each metric entry has `auroc_by_lookback`, `horizon`, `max_auroc`, `shuffle_flag`, `n_valid_by_lookback`, `is_primary`; integration test verifies all fields |

### Observable Truths (Plan 07-02)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 7 | Holm-Bonferroni correction is applied only to the 5 pre-registered primary metrics, not all 21 | VERIFIED | `holm_bonferroni` applies correction factor `(m - i)` where `m = len(p_values)`; `apply_statistical_controls` passes only primary metric p-values; test explicitly verifies smallest p-value multiplied by 5 (not 21); 4 tests pass |
| 8 | BCa bootstrap confidence intervals are computed on AUROC estimates with percentile fallback | VERIFIED | `auroc_with_bootstrap_ci` tries `method="BCa"` first then falls back to `method="percentile"` in a loop; uses `scipy.stats.bootstrap` with vectorized AUROC statistic; 4 tests pass including deterministic, separable, no-signal, and fallback cases |
| 9 | Cohen's d effect sizes are computed per metric per lookback distance | VERIFIED | `cohens_d` uses pooled std; `compute_cohens_d_by_lookback` returns shape `(r_value,)` array; returns NaN for pooled_std < 1e-12 or insufficient samples; 4 tests pass |
| 10 | Two correlation matrices are computed: measurement redundancy (raw values) and predictive redundancy (AUROC values) | VERIFIED | `compute_correlation_matrix(mode="measurement")` pools raw metric values at event positions; `compute_correlation_matrix(mode="predictive")` computes AUROC curves then correlates; both flag pairs with `|r| > 0.9`; 3 tests pass including high-correlation detection and redundancy flagging |
| 11 | Metric importance ranking orders metrics by max AUROC per layer, annotated with redundancy flags | VERIFIED | `compute_metric_ranking` sorts by `max_auroc` descending, annotates `redundant_with` and `is_primary`; `apply_statistical_controls` groups by layer key; 3 tests pass covering ordering, redundancy annotation, and per-layer separation |
| 12 | Headline comparison reports QK^T vs AVWo predictive horizon per r value with bootstrap CIs | VERIFIED | `compute_headline_comparison` extracts max horizon across primary qkt.*/avwo.* metrics per r-value, reports `qkt_max_horizon`, `avwo_max_horizon`, `qkt_leads`, `gap`; 2 tests pass for QK^T-leads and AVWo-leads cases |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/analysis/__init__.py` | Public API for analysis module | VERIFIED | 49 lines; exports all 17 public functions via `__all__`; imports from all 3 submodules |
| `src/analysis/event_extraction.py` | AnalysisEvent dataclass, extract_events, filter_contaminated_events, stratify_by_r | VERIFIED | 216 lines; frozen dataclass with 6 fields; 3 complete functions with proper indexing (resolution_step - 1 convention) |
| `src/analysis/auroc_horizon.py` | AUROC computation, predictive horizon, shuffle controls | VERIFIED | 399 lines; 7 functions including `PRIMARY_METRICS` frozenset of 5 metrics; full pipeline orchestrator |
| `src/analysis/statistical_controls.py` | Holm-Bonferroni, BCa bootstrap, Cohen's d, correlation, ranking, headline comparison | VERIFIED | 683 lines; 8 functions + `apply_statistical_controls` orchestrator |
| `tests/test_event_extraction.py` | Tests for event extraction and contamination filtering | VERIFIED | 9 test functions, all passing |
| `tests/test_auroc_horizon.py` | Tests for AUROC computation, horizon, shuffle controls | VERIFIED | 14 test functions (plan said 13; integration test split into 2 classes), all passing |
| `tests/test_statistical_controls.py` | Tests for all statistical control functions | VERIFIED | 21 test functions, all passing |

**Note on test count:** Plan 07-01 said 9 + 13 tests; actual is 9 + 14 (the integration test includes an extra `test_primary_metrics_constant`). This is an acceptable deviation — more coverage, not less.

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/analysis/event_extraction.py` | `src/evaluation/behavioral.py` | `from src.evaluation.behavioral import RuleOutcome` | WIRED | Exact pattern found at line 21; used throughout extract_events and filter_contaminated_events |
| `src/analysis/auroc_horizon.py` | `src/analysis/event_extraction.py` | `from src.analysis.event_extraction import AnalysisEvent` | WIRED | Import block at lines 13-18; AnalysisEvent used in all curve/shuffle functions |
| `src/analysis/auroc_horizon.py` | `scipy.stats` | `from scipy.stats import rankdata` | WIRED | Line 11; rankdata used in `auroc_from_groups` for rank-based AUROC |
| `src/analysis/statistical_controls.py` | `scipy.stats` | `from scipy.stats import bootstrap` | WIRED | Line 18; bootstrap used in `auroc_with_bootstrap_ci` for BCa and percentile methods |
| `src/analysis/statistical_controls.py` | `src/analysis/auroc_horizon.py` | `from src.analysis.auroc_horizon import auroc_from_groups` | WIRED | Lines 20-28; auroc_from_groups used as the statistic inside bootstrap |
| `src/evaluation/pipeline.py` | `result.generated` | `npz_data["generated"] = result.generated` | WIRED | Line 410; generated array saved to NPZ for standalone re-analysis |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PRED-01 | 07-01 | Compute AUROC at each lookback distance j (1 to r) for each SVD metric | SATISFIED | `compute_auroc_curve` iterates j=1..r; `run_auroc_analysis` applies to all metric_keys |
| PRED-02 | 07-01 | Calculate predictive horizon as furthest j where AUROC exceeds 0.75 | SATISFIED | `compute_predictive_horizon` with configurable threshold (default 0.75); stored as `horizon` in result dict |
| PRED-03 | 07-01 | Position-matched baselines to control for positional confounds | SATISFIED | Control events are FOLLOWED encounters at same absolute positions in the same walks (same generated array); contamination filter prevents temporal leakage |
| PRED-04 | 07-01 | Shuffle controls (permuted labels) to verify signal is not positional artifact | SATISFIED | `run_shuffle_control` with 10,000 permutations; `shuffle_flag` set when p95 shuffled AUROC > 0.6; verified against positional artifact test |
| PRED-05 | 07-01 | Store per-metric AUROC curves in result.json metrics block | SATISFIED | `auroc_by_lookback` list in each metric's result; `run_auroc_analysis` returns structured dict matching result.json schema |
| STAT-01 | 07-02 | Holm-Bonferroni correction for multiple comparisons across pre-registered primary metrics | SATISFIED | `holm_bonferroni` step-down procedure; applied to exactly 5 primary metrics in `apply_statistical_controls`; correction factor at most 5 |
| STAT-02 | 07-02 | Bootstrap confidence intervals on AUROC and predictive horizon estimates | SATISFIED | `auroc_with_bootstrap_ci` with BCa-then-percentile fallback; applied to metrics with 10+ events per class in `apply_statistical_controls` |
| STAT-03 | 07-02 | Effect sizes (Cohen's d) for pre-failure vs post-failure metric distributions | SATISFIED | `cohens_d` and `compute_cohens_d_by_lookback`; stored as `cohens_d_by_lookback` per metric in enriched result |
| STAT-04 | 07-02 | SVD metric correlation matrix to identify redundant metrics | SATISFIED | Two matrices: `measurement_redundancy` (raw values) and `predictive_redundancy` (AUROC curves); both flag `|r| > 0.9` pairs as `redundant_pairs` |
| STAT-05 | 07-02 | Metric importance ranking by max AUROC across j values | SATISFIED | `compute_metric_ranking` ordered by max AUROC descending per layer; annotated with primary/exploratory and redundancy flags; stored in `metric_ranking` by layer |

All 10 requirements (PRED-01 through PRED-05, STAT-01 through STAT-05) are satisfied. No orphaned requirements found — all IDs declared in plan frontmatter match the REQUIREMENTS.md Phase 7 entries and all are marked Complete.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| `src/analysis/event_extraction.py:138` | `return [], {...}` | Info | Legitimate early return for empty-input case, not a stub |

No blockers or warnings found. The one flagged line is a proper empty-input guard with a fully populated audit dict.

### Human Verification Required

No items require human verification. All behaviors under test are algorithmic and verifiable programmatically:

- AUROC values against known distributions (perfect/no/reversed separation) are mathematical invariants
- Holm-Bonferroni adjusted p-values match hand-calculated expected values
- Cohen's d matches hand-calculated pooled-std formula
- JSON serializability confirmed programmatically in integration test

### Summary

Phase 7 goal is fully achieved. The system measures predictive horizon for SVD metrics by:

1. Extracting jumper encounters with correct resolution_step = encounter_step + r indexing convention aligned to behavioral.py
2. Applying contamination filtering that excludes post-violation encounters (but not post-success) with full audit metrics
3. Computing rank-based AUROC at each lookback distance j=1..r, stratified by r-value, with NaN handling and minimum event count enforcement
4. Calculating predictive horizon as the furthest j exceeding the 0.75 threshold
5. Validating with 10,000-permutation shuffle controls that flag positional artifacts
6. Adding BCa bootstrap CIs (with percentile fallback), Holm-Bonferroni correction on the 5 pre-registered primary metrics, Cohen's d per lookback, two correlation matrices (measurement + predictive redundancy), per-layer metric rankings, and headline QK^T vs AVWo comparison

All 44 phase 7 tests pass. 236 existing tests pass with no regressions (280 total). The analysis module is fully importable and produces JSON-serializable output matching the result.json predictive_horizon schema.

---

_Verified: 2026-02-26T00:40:00Z_
_Verifier: Claude (gsd-verifier)_
