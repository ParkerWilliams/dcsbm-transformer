---
phase: 21-statistical-controls
verified: 2026-03-10T03:15:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
gaps: []
human_verification: []
---

# Phase 21: Statistical Controls Verification Report

**Phase Goal:** All statistical testing machinery produces mathematically correct results — permutation tests, confidence intervals, multiple comparison corrections, effect sizes, and study design splits
**Verified:** 2026-03-10T03:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Shuffle permutation null correctly permutes event labels (not metric values) while preserving group sizes, and the resulting p-value distribution is uniform under H0 | VERIFIED | `test_shuffle_permutation.py`: H0 KS uniformity test passes (100 independent trials), immutability confirmed, group size mask invariant confirmed over 50 iterations, signal detection positive control passes |
| 2 | Bootstrap BCa confidence intervals apply correct bias correction (z0) and acceleration (a) per Efron's original method | VERIFIED | `test_bootstrap_bca.py`: `scipy.stats.bootstrap` called with `method='BCa'`, `vectorized=True`, `confidence_level` propagated; known-answer perfect separation (AUROC=1.0) and overlapping groups (CI width > 0, point inside CI) confirmed |
| 3 | Holm-Bonferroni step-down correction applies sorted p-values against thresholds alpha/(m-k+1) for k=1..m, rejecting in correct order | VERIFIED | `test_holm_bonferroni.py`: textbook 5-hypothesis Holm (1979) example matches exactly; monotonicity enforced on random inputs (20 trials); 0-based (m-i) equals 1-based (m-k+1) formula proven for all m in [1,10] |
| 4 | Cohen's d uses the pooled standard deviation formula s_p = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2)) | VERIFIED | `test_cohens_d.py`: hand-calculated textbook example matches within 1e-10; independent manual-loop pooled std computation agrees; exact unit-pooled-std case returns exactly 1.0; NaN guards for n<2 and zero pooled_std confirmed |
| 5 | Exploratory/confirmatory split uses reproducible random assignment with correct proportions, and Spearman \|r\| > 0.9 redundancy threshold is applied correctly | VERIFIED | `test_correlation_redundancy.py`: measurement mode confirmed Spearman (y=x^3 monotone test gives r=1.0, not ~0.95 Pearson); threshold boundary at 0.89 not flagged, 0.91 flagged, 0.90 exactly not flagged (strict >); `test_exploratory_confirmatory_split.py`: 40 viol + 60 non-viol split exactly 20/20 and 30/30; SPLIT_SEED=2026 determinism verified |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Min Lines Required | Actual Lines | Status | Details |
|----------|--------------------|--------------|--------|---------|
| `tests/audit/test_shuffle_permutation.py` | 60 | 214 | VERIFIED | Exists, substantive (214 lines), imported and executed |
| `tests/audit/test_bootstrap_bca.py` | 40 | 188 | VERIFIED | Exists, substantive (188 lines), imported and executed |
| `tests/audit/test_holm_bonferroni.py` | 50 | 205 | VERIFIED | Exists, substantive (205 lines), imported and executed |
| `tests/audit/test_cohens_d.py` | 40 | 196 | VERIFIED | Exists, substantive (196 lines), imported and executed |
| `tests/audit/test_correlation_redundancy.py` | 50 | 473 | VERIFIED | Exists, substantive (473 lines), imported and executed |
| `tests/audit/test_exploratory_confirmatory_split.py` | 60 | 282 | VERIFIED | Exists, substantive (282 lines), imported and executed |

All six artifacts exceed their minimum line requirements. Total: 1,558 lines across 6 files.

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/audit/test_shuffle_permutation.py` | `src/analysis/auroc_horizon.py:run_shuffle_control` | `from src.analysis.auroc_horizon import run_shuffle_control` | WIRED | Import present and function invoked in 4 tests |
| `tests/audit/test_bootstrap_bca.py` | `src/analysis/statistical_controls.py:auroc_with_bootstrap_ci` | `from src.analysis.statistical_controls import auroc_with_bootstrap_ci` | WIRED | Import present; function invoked in TestBCaDelegation, TestKnownAnswerEndToEnd, TestNaNHandling |
| `tests/audit/test_holm_bonferroni.py` | `src/analysis/statistical_controls.py:holm_bonferroni` | `from src.analysis.statistical_controls import holm_bonferroni` | WIRED | Import present; function invoked across all 4 test classes (14 tests) |
| `tests/audit/test_cohens_d.py` | `src/analysis/statistical_controls.py:cohens_d` | `from src.analysis.statistical_controls import cohens_d` | WIRED | Import present; function invoked across all 4 test classes (13 tests) |
| `tests/audit/test_correlation_redundancy.py` | `src/analysis/statistical_controls.py:compute_correlation_matrix` | `from src.analysis.statistical_controls import compute_correlation_matrix` | WIRED | Import present; function invoked in all 5 test classes (15 tests) |
| `tests/audit/test_exploratory_confirmatory_split.py` | `src/evaluation/split.py:assign_split` | `from src.evaluation.split import assign_split` | WIRED | Import present; function invoked in all 6 test classes (16 tests) |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| STAT-01 | 21-01-PLAN.md | Shuffle permutation null correctly permutes event labels while preserving group sizes | SATISFIED | 5 tests in `test_shuffle_permutation.py` pass: H0 KS uniformity, immutability, group size preservation, signal detection, no-signal control |
| STAT-02 | 21-01-PLAN.md | Bootstrap CIs use correct BCa method with proper bias correction and acceleration | SATISFIED | 9 tests in `test_bootstrap_bca.py` pass: BCa delegation verified via mock, confidence_level propagated, vectorized=True confirmed, 3 known-answer end-to-end tests, 3 NaN handling tests |
| STAT-03 | 21-01-PLAN.md | Holm-Bonferroni FWER correction applies sorted p-values with correct step-down threshold formula | SATISFIED | 14 tests in `test_holm_bonferroni.py` pass: textbook example, all/none significant, monotonicity enforcement, 7 edge cases, formula equivalence |
| STAT-04 | 21-01-PLAN.md | Cohen's d effect size uses correct pooled standard deviation formula | SATISFIED | 13 tests in `test_cohens_d.py` pass: hand calculation within 1e-10, exact unit case, equal groups, sign convention (3 tests), NaN guards (5 tests), independent pooled std (2 tests) |
| STAT-05 | 21-02-PLAN.md | Spearman correlation for redundancy analysis correctly computed and threshold (\|r\| > 0.9) justified | SATISFIED | Production code fixed from `np.corrcoef` (Pearson) to `scipy.stats.spearmanr`; 15 tests in `test_correlation_redundancy.py` pass: Spearman method identified, Pearson confirmed absent for measurement mode, boundary tests (0.89/0.90/0.91), Pearson correctness for predictive mode |
| STAT-06 | 21-02-PLAN.md | Exploratory/confirmatory split is methodologically sound (random, reproducible, correct proportions) | SATISFIED | 16 tests in `test_exploratory_confirmatory_split.py` pass: proportion verification, stratification independence, determinism (3 tests), all edge cases (7 tests), value validation (2 tests), SPLIT_SEED=2026 documented |

No orphaned requirements found. All 6 STAT requirements are claimed by plans 21-01 and 21-02 and independently verified.

---

### Anti-Patterns Found

None detected. Scan of all 6 new test files and the one modified production file (`src/analysis/statistical_controls.py`) found no TODOs, FIXMEs, empty implementations, placeholder returns, or stub patterns.

---

### Test Execution Results

**Phase 21 audit tests only (72 tests):**
```
72 passed in 121.98s
```

**Full audit suite regression check (234 tests including all prior phases):**
```
234 passed in 112.80s
```
No regressions introduced.

**Commit verification:** All 4 task commits documented in SUMMARYs confirmed present in git log:
- `db2285a` — test(21-01): audit shuffle permutation null and Holm-Bonferroni correction (STAT-01, STAT-03)
- `68a80ef` — test(21-01): audit BCa bootstrap CIs and Cohen's d effect size (STAT-02, STAT-04)
- `da68493` — fix(21-02): audit and fix Spearman correlation redundancy analysis (STAT-05)
- `5a28b92` — test(21-02): audit exploratory/confirmatory split assignment (STAT-06)

---

### Human Verification Required

None. All phase 21 deliverables are audit test files and one production code fix — all verifiable programmatically. The mathematical correctness claims (textbook examples, hand calculations, mock-based delegation checks) are confirmed by the passing test suite.

---

### Production Code Changes Note

Plan 21-02 identified and fixed a bug: `src/analysis/statistical_controls.py` measurement mode used `np.corrcoef` (Pearson) instead of `scipy.stats.spearmanr` (Spearman) for redundancy analysis. The fix is present in commit `da68493`. The corrected code at line 284 reads:
```python
corr_result, _ = spearmanr(data_matrix, axis=1)
```
with 2-variable scalar return handled at lines 286-290. This is a material correction satisfying STAT-05.

---

## Summary

Phase 21 goal is fully achieved. All five success criteria from ROADMAP.md are verified against the actual codebase. The six audit test files exist, are substantive (well above minimum line counts), are wired to their production functions via direct imports, and all 72 tests pass. All six STAT requirements are satisfied with no orphaned requirements. One production bug (Pearson vs Spearman) was found and fixed during the phase as expected.

---

_Verified: 2026-03-10T03:15:00Z_
_Verifier: Claude (gsd-verifier)_
