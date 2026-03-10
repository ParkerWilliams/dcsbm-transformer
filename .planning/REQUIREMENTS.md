# Requirements: DCSBM Transformer v1.2 Mathematical Audit

**Defined:** 2026-03-05
**Core Value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.

## v1.2 Requirements

Requirements for mathematical correctness audit. Each requirement verifies a specific mathematical claim or formula against its implementation, and fixes any issues found.

### Graph Theory & Data Generation

- [x] **GRAPH-01**: DCSBM edge probability matrix P_{ij} = theta_i * theta_j * B_{z_i, z_j} is correctly implemented (degree correction, block assignment, symmetry)
- [x] **GRAPH-02**: Walk sampling produces uniform random neighbor selection at each step (no bias toward high/low degree vertices beyond what DCSBM specifies)
- [x] **GRAPH-03**: Block jumper designation correctly assigns jump distance r and target block according to specification
- [x] **GRAPH-04**: Behavioral classification (4-class: followed, violated, unconstrained, pending) correctly identifies rule outcomes at each step
- [x] **GRAPH-05**: Walk compliance rate calculation matches the mathematical definition (violations / constrained steps)

### SVD Metrics

- [x] **SVD-01**: QK^T attention matrix is correctly constructed from Q and K projections before SVD extraction
- [x] **SVD-02**: WvWo and AVWo matrix constructions match their mathematical definitions
- [x] **SVD-03**: All singular-value-derived metrics (condition number, spectral gap, entropy, effective rank, stable rank) use correct formulas
- [x] **SVD-04**: Grassmannian distance (subspace angle) between consecutive steps is computed correctly using the standard definition
- [x] **SVD-05**: Spectrum trajectory storage preserves numerical fidelity (verify float16 vs float32 impact on downstream curvature/torsion)
- [x] **SVD-06**: Frenet-Serret curvature and torsion computation uses correct discrete differential geometry formulas

### AUROC & Predictive Horizon

- [x] **AUROC-01**: AUROC computation from violation/control groups uses correct rank-based probability P(X_violated > X_followed)
- [x] **AUROC-02**: Lookback distance j correctly indexes metric values at step (t-j) relative to resolution step t
- [x] **AUROC-03**: Predictive horizon definition (max j where AUROC > 0.75) is consistently applied across all analysis paths
- [x] **AUROC-04**: Event extraction (violation events, control events) correctly identifies resolution steps from behavioral labels

### Statistical Controls

- [x] **STAT-01**: Shuffle permutation null distribution correctly permutes event labels while preserving group sizes
- [x] **STAT-02**: Bootstrap confidence intervals use correct BCa method with proper bias correction and acceleration
- [x] **STAT-03**: Holm-Bonferroni family-wise error correction applies sorted p-values with correct step-down threshold formula
- [x] **STAT-04**: Cohen's d effect size uses correct pooled standard deviation formula
- [x] **STAT-05**: Spearman correlation for redundancy analysis is correctly computed and threshold (|r| > 0.9) is justified
- [x] **STAT-06**: Exploratory/confirmatory split assignment is methodologically sound (random, reproducible, correct proportions)

### Softmax Filtering Bound

- [x] **SFTX-01**: LaTeX derivation of QK^T → AVWo epsilon-bound is mathematically correct (each step follows from the previous)
- [x] **SFTX-02**: Empirical verification code correctly tests the bound's predictions against observed data
- [x] **SFTX-03**: Bound assumptions are explicitly stated and implementation respects them

### Null Model

- [x] **NULL-01**: Grassmannian drift on jumper-free sequences is computed identically to primary analysis (same SVD extraction, same metrics)
- [x] **NULL-02**: Mann-Whitney U comparison uses correct test statistic and p-value computation
- [x] **NULL-03**: Column-filtered adjacency correctly removes all jumper paths (zero contamination)
- [x] **NULL-04**: Null model and primary model use separate Holm-Bonferroni families (no cross-contamination of p-values)

### Audit Report

- [x] **REPT-01**: Generate HTML audit report linking every mathematical formula to its code implementation with LaTeX rendering
- [x] **REPT-02**: Each report entry includes: formula in LaTeX, code location (file:line), correctness verdict, and fix description if applicable
- [ ] **REPT-03**: Report is self-contained (inline CSS/JS, MathJax for LaTeX) and navigable by audit category

## Out of Scope

| Feature | Reason |
|---------|--------|
| Training loop mathematics (loss, gradients, LR schedule) | Verified by convergence; standard PyTorch autograd |
| Parameter sweep infrastructure | Deferred to v2 |
| New metric development | Audit existing, don't add new |
| Performance optimization | Already addressed in quick tasks 3-4 |
| Test coverage expansion | Audit math, not test infrastructure |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| GRAPH-01 | Phase 18 | Complete |
| GRAPH-02 | Phase 18 | Complete |
| GRAPH-03 | Phase 18 | Complete |
| GRAPH-04 | Phase 18 | Complete |
| GRAPH-05 | Phase 18 | Complete |
| SVD-01 | Phase 19 | Complete |
| SVD-02 | Phase 19 | Complete |
| SVD-03 | Phase 19 | Complete |
| SVD-04 | Phase 19 | Complete |
| SVD-05 | Phase 19 | Complete |
| SVD-06 | Phase 19 | Complete |
| AUROC-01 | Phase 20 | Complete |
| AUROC-02 | Phase 20 | Complete |
| AUROC-03 | Phase 20 | Complete |
| AUROC-04 | Phase 20 | Complete |
| STAT-01 | Phase 21 | Complete |
| STAT-02 | Phase 21 | Complete |
| STAT-03 | Phase 21 | Complete |
| STAT-04 | Phase 21 | Complete |
| STAT-05 | Phase 21 | Complete |
| STAT-06 | Phase 21 | Complete |
| SFTX-01 | Phase 22 | Complete |
| SFTX-02 | Phase 22 | Complete |
| SFTX-03 | Phase 22 | Complete |
| NULL-01 | Phase 22 | Complete |
| NULL-02 | Phase 22 | Complete |
| NULL-03 | Phase 22 | Complete |
| NULL-04 | Phase 22 | Complete |
| REPT-01 | Phase 23 | Complete |
| REPT-02 | Phase 23 | Complete |
| REPT-03 | Phase 23 | Pending |

**Coverage:**
- v1.2 requirements: 31 total
- Mapped to phases: 31
- Unmapped: 0

---
*Requirements defined: 2026-03-05*
*Last updated: 2026-03-05 after roadmap creation*
