# Phase 6 Verification: Behavioral Evaluation and SVD Collection

**Date:** 2026-02-25
**Verifier:** Automated verification against ROADMAP success criteria
**Test suite:** 236 tests passing, 0 regressions

## Phase Goal

> A single evaluation pass through generated sequences produces both behavioral labels (4-class outcomes) and SVD metrics across three targets (QK^T routing, WvWo OV circuit, AVWo net residual update) with numerical stability guarantees

**Verdict: ACHIEVED** -- `fused_evaluate()` performs a single autoregressive generation loop that simultaneously collects SVD metrics on all three targets and produces behavioral labels. Numerical guards prevent NaN/Inf in stored metrics.

## Success Criteria Verification

### Criterion 1: 4-class behavioral classification with failure_index
> Each generation step is classified into the 4-class outcome (edge valid/invalid x rule followed/violated/not-applicable) and sequences are annotated with failure_index

**Status: PASS**

Evidence:
- `src/evaluation/behavioral.py`: `classify_steps()` returns `edge_valid` (bool), `rule_outcome` (RuleOutcome: NOT_APPLICABLE=0, FOLLOWED=1, VIOLATED=2), `failure_index` (first violation step, -1 if none)
- 4-class outcome: edge_valid (2 states) x rule_outcome (3 states including NA) with failure_index annotation
- `tests/test_behavioral.py`: 18 unit tests cover edge validity, rule compliance, failure_index detection, batched classification, and continuation after violation
- Integration: `fused_evaluate()` calls `classify_steps()` after generation and stores results in `EvaluationResult`

Requirements completed: EVAL-01, EVAL-02, EVAL-03, EVAL-04

### Criterion 2: SVD on three targets per layer at every step
> SVD is computed on three targets per layer at every token step: QK^T (routing stability), WvWo (OV circuit stability), AVWo (net residual update stability), using batched torch.linalg.svd with full_matrices=False on GPU

**Status: PASS**

Evidence:
- `src/evaluation/pipeline.py` lines 250-252: `torch.linalg.svd(qkt_clean, full_matrices=False)`
- `src/evaluation/pipeline.py` lines 301-303: `torch.linalg.svd(avwo_clean, full_matrices=False)`
- `src/evaluation/pipeline.py` line 154: `torch.linalg.svd(wvwo_clean, full_matrices=False)` (computed once, broadcast)
- `_compute_avwo_for_layer()`: AVWo = (A @ V) @ W_o.weight.T matching actual residual stream contribution
- `ExtractionMode.SVD_TARGETS` extracts QK^T, attention weights A, and values V
- `tests/test_evaluation_pipeline.py` TestSVDTargets: verifies QK^T produces finite metrics, WvWo is static across steps, AVWo differs from QK^T

Requirements completed: SVD-01, EVAL-05

### Criterion 3: Seven scalar metrics stored in token_metrics.npz
> Seven scalar metrics per target per step (stable rank, spectral entropy, spectral gap + generalized k=2,4, condition number, rank-1 residual norm, read-write alignment for WvWo) are computed and stored as token-level time series in token_metrics.npz keyed by target.metric

**Status: PASS**

Evidence:
- `src/evaluation/svd_metrics.py` `compute_all_metrics()`: returns stable_rank, spectral_entropy, spectral_gap_1_2, spectral_gap_2_3, spectral_gap_4_5, condition_number, rank1_residual_norm, read_write_alignment (for square matrices only)
- Plus grassmannian_distance computed separately for subspace rotation tracking
- NPZ key convention: `target.layer_N.metric_name` (e.g., `qkt.layer_0.stable_rank`)
- `save_evaluation_results()` writes compressed NPZ with all SVD and behavioral arrays
- `tests/test_evaluation_pipeline.py` TestNPZOutput: verifies save/load round-trip, key naming convention

Requirements completed: SVD-04

### Criterion 4: Warmup skip and numerical guards
> SVD metrics are collected only for positions >= w (context window warmup), and numerical guards (NaN/Inf clamping, epsilon in entropy, condition number cap at 1e6) prevent any NaN or Inf in stored metrics

**Status: PASS**

Evidence:
- `src/evaluation/pipeline.py` line 238: `if step >= w - 1` gates SVD collection
- Pre-allocated arrays are NaN-filled; positions < w remain NaN by design
- `guard_matrix_for_svd()`: clamps NaN->0.0, posInf->1e6, negInf->-1e6
- EPS=1e-12 in all denominator guards (stable_rank, spectral_entropy, condition_number, rank1_residual_norm)
- CONDITION_CAP=1e6 in condition_number
- `spectral_entropy`: clamped min=0.0 to prevent negative entropy from floating point
- `tests/test_evaluation_pipeline.py` TestWarmupSkip: verifies warmup positions are NaN, post-warmup positions are finite
- `tests/test_svd_metrics.py` TestNumericalGuards: verifies guard behavior on zero/inf matrices

Requirements completed: SVD-06

### Criterion 5: Unit tests against analytically known matrices
> Each SVD metric function has a unit test that verifies its output against an analytically known matrix decomposition for each target type

**Status: PASS**

Evidence:
- `tests/test_svd_metrics.py`: 34 unit tests using analytically known matrices:
  - Identity matrices (stable_rank=n, entropy=log(n), condition=1)
  - Diagonal matrices with known singular values
  - Rank-1 matrices (stable_rank=1, condition=cap, residual=0)
  - Uniform singular value matrices
  - Orthogonal subspaces for Grassmannian distance
  - torch.float64 precision throughout for numerical accuracy
- Test classes: TestStableRank (4), TestSpectralEntropy (4), TestSpectralGaps (3), TestConditionNumber (4), TestRank1ResidualNorm (3), TestReadWriteAlignment (2), TestGrassmannianDistance (3), TestNumericalGuards (4), TestBatchedSVD (3), TestComputeAllMetrics (4)

Requirements completed: SVD-02, SVD-03, SVD-05, SVD-07

## Requirements Completion

All 12 Phase 6 requirements verified complete:

| Requirement | Description | Plan | Status |
|-------------|-------------|------|--------|
| EVAL-01 | 4-class behavioral outcome | 06-02 | Complete |
| EVAL-02 | Edge validity via CSR adjacency | 06-02 | Complete |
| EVAL-03 | Rule compliance via jumper constraints | 06-02 | Complete |
| EVAL-04 | failure_index annotation | 06-02 | Complete |
| EVAL-05 | Fused evaluation pipeline | 06-03 | Complete |
| SVD-01 | SVD on 3 targets per layer | 06-03 | Complete |
| SVD-02 | Metric unit tests on known matrices | 06-01 | Complete |
| SVD-03 | 8+1 SVD metric functions | 06-01 | Complete |
| SVD-04 | token_metrics.npz output format | 06-03 | Complete |
| SVD-05 | Numerical guards (EPS, cap) | 06-01 | Complete |
| SVD-06 | Warmup skip for positions < w | 06-03 | Complete |
| SVD-07 | guard_matrix_for_svd NaN/Inf clamping | 06-01 | Complete |

## Test Summary

- **SVD metrics unit tests:** 34 tests (test_svd_metrics.py)
- **Behavioral classification tests:** 18 tests (test_behavioral.py)
- **Integration pipeline tests:** 16 tests (test_evaluation_pipeline.py)
- **Phase 6 total:** 68 new tests
- **Full suite:** 236 tests passing, 0 failures, 0 regressions

## Verdict

**PHASE 6 VERIFICATION: PASSED**

All 5 success criteria met. All 12 requirements complete. 236/236 tests passing.
