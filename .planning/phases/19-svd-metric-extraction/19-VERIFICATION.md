---
phase: 19-svd-metric-extraction
verified: 2026-03-05T21:00:00Z
status: passed
score: 12/12 must-haves verified
re_verification: false
---

# Phase 19: SVD Metric Extraction Verification Report

**Phase Goal:** Every SVD-related metric formula and matrix construction is verified correct, including numerical fidelity of spectrum storage
**Verified:** 2026-03-05
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | QK^T matrix equals (x@Wq) @ (x@Wk)^T / sqrt(d_k) with zero-fill causal mask for SVD target and -inf for softmax path | VERIFIED | 8 passing tests in test_qkt_construction.py; formula, dual mask, multi-head, and scale-factor each tested with known weights against manual computation |
| 2 | WvWo per-head OV circuit equals Wv_h.T @ Wo_h.T with correct weight slicing for 1h and 4h configs | VERIFIED | 2 passing tests in test_wvwo_avwo_construction.py; single-head and per-head slicing verified with known weights against manual matrix product |
| 3 | AVWo = (A_h @ V_h) @ Wo_h^T with correct per-head W_o slicing | VERIFIED | 3 passing tests in test_wvwo_avwo_construction.py; single-head and multi-head (all 4 heads) verified with known attention and value tensors |
| 4 | All 5 singular-value metrics produce textbook-correct values on matrices with analytically known singular values | VERIFIED | 20 passing tests in test_sv_metrics.py covering stable_rank, spectral_entropy, spectral_gap (1-2, 2-3, 4-5), condition_number, rank1_residual_norm, and read_write_alignment |
| 5 | Grassmannian distance uses geodesic d = sqrt(sum(arccos(sigma_i)^2)) matching Edelman et al. (1998) | VERIFIED | 16 passing tests in test_grassmannian_distance.py; identical subspaces d=0, orthogonal d=pi/2*sqrt(k), known-angle rotations, k=1/2/3, clipping, batched |
| 6 | Grassmannian distance returns 0 for identical subspaces, pi/2*sqrt(k) for orthogonal subspaces | VERIFIED | Covered by TestGrassmannianIdenticalSubspaces and TestGrassmannianOrthogonalSubspaces classes, all pass |
| 7 | Grassmannian distance formula generalizes correctly for k=1, k=2, k=3 | VERIFIED | TestGrassmannianVaryingK::test_k1, test_k2, test_k3 all pass |
| 8 | Float16 vs float32 spectrum storage impact on curvature/torsion is quantified with clear recommendation | VERIFIED | 5 passing tests in test_float16_fidelity.py document 1130% curvature error and 702M% torsion error; pipeline.py updated to float32 |
| 9 | Discrete curvature on a circle of radius r converges to 1/r with O(h) error as step size decreases | VERIFIED | TestCurvatureConvergence passes: monotonically decreasing error across N=100/1000/10000, actual convergence is O(h^2) (ratio ~100x) |
| 10 | Discrete torsion on a circle is zero (planar curve) | VERIFIED | TestCircleTorsion::test_circle_torsion_near_zero passes; values within atol=0.05 for N=1000 |
| 11 | Discrete curvature and torsion on a helix converge to analytically known values | VERIFIED | TestHelixCurvatureTorsion passes; curvature and torsion match analytic formulas kappa=r/(r^2+c^2), tau=c/(r^2+c^2) within 10% |
| 12 | Index mapping is correct: curvature at a[t] maps to orig_idx=t+1, torsion at j[t] maps to orig_idx=t+2 | VERIFIED | TestCurvatureIndexMapping and TestTorsionIndexMapping both pass; peak curvature and torsion found within tolerance of expected original index |

**Score:** 12/12 truths verified

### Required Artifacts

| Artifact | min_lines | Actual Lines | Status | Details |
|----------|-----------|--------------|--------|---------|
| `tests/audit/test_qkt_construction.py` | 100 | 278 | VERIFIED | 8 tests across 4 classes; imports and calls CausalSelfAttention with extract=True |
| `tests/audit/test_wvwo_avwo_construction.py` | 100 | 226 | VERIFIED | 5 tests across 4 classes; imports TransformerLM.get_wvwo() and _compute_avwo_for_layer |
| `tests/audit/test_sv_metrics.py` | 150 | 257 | VERIFIED | 20 tests across 6 classes; imports all metric functions from svd_metrics.py |
| `tests/audit/test_grassmannian_distance.py` | 120 | 314 | VERIFIED | 16 tests across 6 classes; imports grassmannian_distance |
| `tests/audit/test_float16_fidelity.py` | 80 | 245 | VERIFIED | 5 tests across 3 classes; imports spectral_curvature and spectral_torsion |
| `tests/audit/test_curvature_torsion.py` | 200 | 420 | VERIFIED | 13 tests across 7 classes; imports spectral_curvature and spectral_torsion |
| `src/evaluation/pipeline.py` (float32 fix) | n/a | 26470 bytes | VERIFIED | dtype=torch.float32 at line 235 and 340; docstring updated with upgrade note |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| test_qkt_construction.py | src/model/attention.py | Creates CausalSelfAttention, calls `attn(x, extract=True)` | WIRED | `from src.model.attention import CausalSelfAttention`; extract=True present in every test |
| test_wvwo_avwo_construction.py | src/model/transformer.py | Calls `model.get_wvwo()` | WIRED | `from src.model.transformer import TransformerLM`; `model.get_wvwo()` called in single and multi-head tests |
| test_wvwo_avwo_construction.py | src/evaluation/pipeline.py | Calls `_compute_avwo_for_layer` | WIRED | `from src.evaluation.pipeline import _compute_avwo_for_layer`; called in TestAVWoSingleHead and TestAVWoMultiHead |
| test_sv_metrics.py | src/evaluation/svd_metrics.py | Feeds analytically known singular values, verifies each metric formula | WIRED | Imports stable_rank, spectral_entropy, spectral_gap_1_2/2_3/4_5, condition_number, rank1_residual_norm, read_write_alignment |
| test_grassmannian_distance.py | src/evaluation/svd_metrics.py | Calls grassmannian_distance with rotation matrices | WIRED | `from src.evaluation.svd_metrics import grassmannian_distance`; 16 tests call function directly |
| test_float16_fidelity.py | src/analysis/spectrum.py | Computes curvature/torsion in float16 vs float32 | WIRED | `from src.analysis.spectrum import spectral_curvature, spectral_torsion`; both called in fidelity tests |
| src/evaluation/pipeline.py | spectrum storage | Line 235 dtype=torch.float32, line 340 .to(torch.float32) | WIRED | Both float16 locations replaced with float32 as confirmed by grep |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SVD-01 | 19-01-PLAN.md | QK^T attention matrix correctly constructed | SATISFIED | 8 tests pass verifying formula, dual mask, multi-head, scale factor |
| SVD-02 | 19-01-PLAN.md | WvWo and AVWo matrix constructions match mathematical definitions | SATISFIED | 5 tests pass verifying OV circuit and net residual update for 1h and 4h |
| SVD-03 | 19-01-PLAN.md | All SV-derived metrics use correct formulas | SATISFIED | 20 tests pass verifying all 5 metric families with analytically known SVs |
| SVD-04 | 19-02-PLAN.md | Grassmannian distance computed correctly | SATISFIED | 16 tests pass verifying geodesic formula, edge cases, k=1/2/3, batched |
| SVD-05 | 19-02-PLAN.md | Spectrum storage preserves numerical fidelity | SATISFIED | 5 tests quantify float16 error (>1000%); pipeline.py upgraded to float32; 648 tests pass after fix |
| SVD-06 | 19-03-PLAN.md | Frenet-Serret curvature/torsion uses correct discrete formulas | SATISFIED | 13 tests pass; convergence to analytic values on circle and helix verified; index mapping confirmed |

**Orphaned requirements check:** REQUIREMENTS.md Traceability table maps SVD-01 through SVD-06 exclusively to Phase 19. All 6 IDs are claimed by plans 19-01, 19-02, and 19-03. No orphaned requirements.

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
|------|---------|----------|--------|
| None | — | — | — |

Scan of all 6 new test files and pipeline.py: no TODO/FIXME/PLACEHOLDER/placeholder comments found. No empty return stubs. No console.log-only implementations. Test files contain substantive assertions with `torch.testing.assert_close` and documented mathematical reasoning.

### Human Verification Required

None. All verifications are fully automated:

- Metric formula correctness: verifiable algebraically against analytically constructed inputs
- Convergence rate: verified by comparing error magnitudes across resolutions
- Float16 vs float32 fidelity: verified by round-trip cast and numerical error measurement
- Index mapping: verified by finding argmax of curvature/torsion in synthetic curves with known peak location
- Pipeline production fix: verified by grep and by the full regression suite (648 tests pass)

### Commit Verification

All 6 commit hashes cited in summaries exist in git log:

| Commit | Task | Plan |
|--------|------|------|
| 740875b | QK^T audit (SVD-01) | 19-01 |
| f4c2351 | WvWo/AVWo audit (SVD-02) | 19-01 |
| a373c83 | SV metrics audit (SVD-03) | 19-01 |
| c3d5fa4 | Grassmannian distance audit (SVD-04) | 19-02 |
| 47a493c | Float16 fidelity audit + pipeline fix (SVD-05) | 19-02 |
| 5bf7791 | Curvature/torsion audit (SVD-06) | 19-03 |

### Regression Suite

Full test suite result: **648 passed, 1 skipped, 0 failed** (97.94s). The 1 skipped test is pre-existing and unrelated to Phase 19. No regressions introduced by the pipeline.py float16 -> float32 change.

---

## Summary

Phase 19 goal is fully achieved. All 67 new audit tests pass across 6 test files. Every SVD-related metric formula (QK^T, WvWo, AVWo, stable_rank, spectral_entropy, spectral_gap, condition_number, rank1_residual_norm, read_write_alignment, Grassmannian distance, spectral_curvature, spectral_torsion) is verified correct against analytically known inputs or manual matrix products. Spectrum storage numerical fidelity was quantified, a catastrophic float16 error (>1000%) was found and fixed in pipeline.py, and the fix was validated by 648 regression tests.

Requirements SVD-01 through SVD-06 are all satisfied. No production code defects remain in scope.

---

_Verified: 2026-03-05_
_Verifier: Claude (gsd-verifier)_
