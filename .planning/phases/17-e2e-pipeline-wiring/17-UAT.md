---
status: complete
phase: v1.1-all-phases
source: 11-01-SUMMARY.md, 12-01-SUMMARY.md, 12-02-SUMMARY.md, 13-01-SUMMARY.md, 13-02-SUMMARY.md, 13-03-SUMMARY.md, 14-01-SUMMARY.md, 14-02-SUMMARY.md, 15-01-PLAN.md, 15-02-PLAN.md, 15-03-PLAN.md, 16-01-PLAN.md, 16-02-PLAN.md, 16-03-PLAN.md, 17-01-PLAN.md
started: 2026-02-27T12:00:00Z
updated: 2026-02-28T10:30:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Pre-registration document exists with required sections
expected: `docs/pre-registration.md` exists and contains: primary hypothesis, primary metric, alpha level (0.05), Holm-Bonferroni correction, three-outcome decision criterion, and held-out protocol.
result: pass

### 2. Held-out split is deterministic and stratified
expected: `src/evaluation/split.py` provides `assign_split()` that produces deterministic 50/50 exploratory/confirmatory labels, stratified by violation status, using fixed seed 2026.
result: pass

### 3. Null walk generator produces jumper-free walks
expected: `src/analysis/null_model.py` provides `generate_null_walks()` using column-filtered adjacency that zeros out jumper vertex columns. Discard fallback for pathological graphs.
result: pass

### 4. Null model statistical comparison with Mann-Whitney U
expected: `compare_null_vs_violation()` computes per-lookback Mann-Whitney U, Cohen's d, with Holm-Bonferroni correction as a separate family. `run_null_analysis()` orchestrates the full null pipeline.
result: pass

### 5. Null overlay and MP histogram visualizations
expected: Event-aligned plots can show gray 95% CI band + median line for null distribution. Marchenko-Pastur histogram overlays theoretical density with KS annotation.
result: pass

### 6. PR curves with AUPRC computation
expected: `src/analysis/pr_curves.py` computes precision-recall curves per metric per lookback with auto-detected score direction. Visualization shows AUPRC vs lookback with no-skill baseline.
result: pass

### 7. Calibration diagnostics with reliability diagrams
expected: `src/analysis/calibration.py` computes ECE with rank-based pseudo-probability conversion. Reliability diagram shows lookback-colored lines with perfect-calibration diagonal and histogram below.
result: pass

### 8. SVD overhead benchmarking
expected: `src/analysis/svd_benchmark.py` benchmarks full, randomized, and values-only SVD with wall-clock timing. Grouped bar chart and accuracy-cost tradeoff scatter plot generated.
result: pass

### 9. LaTeX softmax filtering bound derivation
expected: `docs/softmax_bound.tex` contains self-contained derivation with three-stage perturbation chain, Theorem 6.1 (end-to-end bound), Corollary 6.2 (Weyl's inequality), softmax Lipschitz constant 1/2, and tightness analysis.
result: pass

### 10. Empirical perturbation bound verification
expected: `src/analysis/perturbation_bound.py` injects controlled perturbations into scaled QK^T, computes theoretical vs empirical bound ratios. Visualization shows scatter + envelope with bound line.
result: pass
note: "User noted: <5% violation tolerance is generous for a theoretical upper bound. A correct bound should be exceeded 0 times. Keep 5% as automated gate but log every violation with magnitude, direction type, and ratio. Violations indicate assumption mismatches between theorem and implementation, not noise. Future refinement item."

### 11. Multi-head attention architecture
expected: `CausalSelfAttention` accepts n_heads parameter, produces per-head QK^T [B, n_heads, T, T]. Scaling uses 1/sqrt(d_head). Config validates n_heads in (1, 2, 4). Single-head backward compatible.
result: pass

### 12. Per-head SVD evaluation with dual-key emission
expected: `fused_evaluate` computes per-head SVD metrics with keys `target.layer_N.head_H.metric_name`. When n_heads=1, emits BOTH legacy 3-part AND v1.1 4-part keys for backward compatibility.
result: pass

### 13. Signal concentration analysis (entropy/Gini)
expected: `src/analysis/signal_concentration.py` computes per-head AUROC, normalized entropy, and Gini coefficient of AUROC distribution across heads. Single-head produces trivial (entropy=0, Gini=0).
result: pass

### 14. Spectrum trajectory storage and curvature analysis
expected: `fused_evaluate` stores top-k singular values per step. `src/analysis/spectrum.py` computes spectral curvature (Savitzky-Golay + Frenet-Serret) and torsion with NaN at zero-velocity steps.
result: pass
note: "User noted: float16 storage for spectrum trajectories is risky for third-derivative computation (torsion). Float16 has ~3 decimal digits of precision — quantization noise amplified through three rounds of differencing. Recommendation: use float32, or keep float16 for archival but cast to float32 before derivative computation. Document in analysis code."

### 15. Compliance curve analysis
expected: `src/analysis/compliance_curve.py` loads multiple result.json files, extracts (r/w ratio, compliance, horizon) tuples, aggregates with mean +/- std. Dual-axis plot shows compliance + horizon vs r/w ratio.
result: pass

### 16. E2E pipeline runs from config to report
expected: `python run_experiment.py --config config.json` chains all stages: seeding -> graph -> walks -> training -> eval -> analysis -> viz -> reporting. Prints stage banners with elapsed times and final summary.
result: pass

### 17. set_seed called at pipeline start
expected: `set_seed(config.seed)` is the first operation in the pipeline, before any stochastic code. Seed logged to console and recorded in result.json metadata.
result: pass

### 18. Predictive horizon written to result.json
expected: After AUROC analysis, `result.json` contains `metrics.predictive_horizon` block with config and by_r_value sub-keys. Statistical controls also persisted.
result: pass

### 19. --dry-run shows full pipeline plan
expected: `python run_experiment.py --config config.json --dry-run` shows config summary plus all 9 pipeline stages and expected output paths (result.json, NPZ, figures, report, config copy).
result: pass

### 20. --verbose enables DEBUG logging
expected: `python run_experiment.py --config config.json --verbose` shows DEBUG-level messages including per-stage details.
result: pass

### 21. Visualization __init__.py exports public API
expected: `from src.visualization import render_all, load_result_data, apply_style, save_figure` all work. `__all__` is defined.
result: pass

### 22. Makefile pdf and report targets wired
expected: `make pdf` calls `generate_math_pdf`. `make report` runs the full pipeline via `run_experiment.py`.
result: pass

### 23. All Phase 13 report sections render
expected: HTML report template has collapsible sections for PR curves (AUPRC table), Calibration (ECE table + reliability diagrams), and SVD benchmark (cost table + figures).
result: pass

### 24. Schema validation backward compatible for all new blocks
expected: `validate_result()` accepts optional blocks for null_model, pr_curves, calibration, svd_benchmark, perturbation_bound, spectrum_analysis, compliance_curve. Passes when absent.
result: pass

### 25. Full test suite passes with no regressions
expected: `pytest tests/` passes 536+ tests with 0 failures.
result: pass

## Summary

total: 25
passed: 25
issues: 0
pending: 0
skipped: 0

## Gaps

[none]

## Deferred Ideas

- Post-hoc detection threshold table for exploratory metrics: for each secondary metric, compute backward-looking "this metric would have passed Gate 1+2 with threshold X at lookback j=Y, effect size Z". For talks/presentations, not the final paper.
- Perturbation bound violation logging: log every single violation with magnitude, direction type, and ratio. Investigate assumption mismatches rather than treating as noise. A correct upper bound should never be exceeded.
- Spectrum trajectory float32 storage: upgrade from float16 to float32 to avoid quantization noise amplification in third-derivative (torsion) computation. Or cast to float32 before derivative computation and document.
