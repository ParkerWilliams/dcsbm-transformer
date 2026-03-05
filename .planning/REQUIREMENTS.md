# Requirements: DCSBM Transformer v1.2 Mathematical Audit

**Defined:** 2026-03-05
**Core Value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.

## v1.2 Requirements

Requirements for mathematical correctness audit. Each requirement verifies a specific mathematical claim or formula against its implementation, and fixes any issues found.

### Graph Theory & Data Generation

- [ ] **GRAPH-01**: DCSBM edge probability matrix P_{ij} = theta_i * theta_j * B_{z_i, z_j} is correctly implemented (degree correction, block assignment, symmetry)
- [ ] **GRAPH-02**: Walk sampling produces uniform random neighbor selection at each step (no bias toward high/low degree vertices beyond what DCSBM specifies)
- [ ] **GRAPH-03**: Block jumper designation correctly assigns jump distance r and target block according to specification
- [ ] **GRAPH-04**: Behavioral classification (4-class: followed, violated, unconstrained, pending) correctly identifies rule outcomes at each step
- [ ] **GRAPH-05**: Walk compliance rate calculation matches the mathematical definition (violations / constrained steps)

### SVD Metrics

- [ ] **SVD-01**: QK^T attention matrix is correctly constructed from Q and K projections before SVD extraction
- [ ] **SVD-02**: WvWo and AVWo matrix constructions match their mathematical definitions
- [ ] **SVD-03**: All singular-value-derived metrics (condition number, spectral gap, entropy, effective rank, stable rank) use correct formulas
- [ ] **SVD-04**: Grassmannian distance (subspace angle) between consecutive steps is computed correctly using the standard definition
- [ ] **SVD-05**: Spectrum trajectory storage preserves numerical fidelity (verify float16 vs float32 impact on downstream curvature/torsion)
- [ ] **SVD-06**: Frenet-Serret curvature and torsion computation uses correct discrete differential geometry formulas

### AUROC & Predictive Horizon

- [ ] **AUROC-01**: AUROC computation from violation/control groups uses correct rank-based probability P(X_violated > X_followed)
- [ ] **AUROC-02**: Lookback distance j correctly indexes metric values at step (t-j) relative to resolution step t
- [ ] **AUROC-03**: Predictive horizon definition (max j where AUROC > 0.75) is consistently applied across all analysis paths
- [ ] **AUROC-04**: Event extraction (violation events, control events) correctly identifies resolution steps from behavioral labels

### Statistical Controls

- [ ] **STAT-01**: Shuffle permutation null distribution correctly permutes event labels while preserving group sizes
- [ ] **STAT-02**: Bootstrap confidence intervals use correct BCa method with proper bias correction and acceleration
- [ ] **STAT-03**: Holm-Bonferroni family-wise error correction applies sorted p-values with correct step-down threshold formula
- [ ] **STAT-04**: Cohen's d effect size uses correct pooled standard deviation formula
- [ ] **STAT-05**: Spearman correlation for redundancy analysis is correctly computed and threshold (|r| > 0.9) is justified
- [ ] **STAT-06**: Exploratory/confirmatory split assignment is methodologically sound (random, reproducible, correct proportions)

### Softmax Filtering Bound

- [ ] **SFTX-01**: LaTeX derivation of QK^T → AVWo epsilon-bound is mathematically correct (each step follows from the previous)
- [ ] **SFTX-02**: Empirical verification code correctly tests the bound's predictions against observed data
- [ ] **SFTX-03**: Bound assumptions are explicitly stated and implementation respects them

### Null Model

- [ ] **NULL-01**: Grassmannian drift on jumper-free sequences is computed identically to primary analysis (same SVD extraction, same metrics)
- [ ] **NULL-02**: Mann-Whitney U comparison uses correct test statistic and p-value computation
- [ ] **NULL-03**: Column-filtered adjacency correctly removes all jumper paths (zero contamination)
- [ ] **NULL-04**: Null model and primary model use separate Holm-Bonferroni families (no cross-contamination of p-values)

### Audit Report

- [ ] **REPT-01**: Generate HTML audit report linking every mathematical formula to its code implementation with LaTeX rendering
- [ ] **REPT-02**: Each report entry includes: formula in LaTeX, code location (file:line), correctness verdict, and fix description if applicable
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
| GRAPH-01 | — | Pending |
| GRAPH-02 | — | Pending |
| GRAPH-03 | — | Pending |
| GRAPH-04 | — | Pending |
| GRAPH-05 | — | Pending |
| SVD-01 | — | Pending |
| SVD-02 | — | Pending |
| SVD-03 | — | Pending |
| SVD-04 | — | Pending |
| SVD-05 | — | Pending |
| SVD-06 | — | Pending |
| AUROC-01 | — | Pending |
| AUROC-02 | — | Pending |
| AUROC-03 | — | Pending |
| AUROC-04 | — | Pending |
| STAT-01 | — | Pending |
| STAT-02 | — | Pending |
| STAT-03 | — | Pending |
| STAT-04 | — | Pending |
| STAT-05 | — | Pending |
| STAT-06 | — | Pending |
| SFTX-01 | — | Pending |
| SFTX-02 | — | Pending |
| SFTX-03 | — | Pending |
| NULL-01 | — | Pending |
| NULL-02 | — | Pending |
| NULL-03 | — | Pending |
| NULL-04 | — | Pending |
| REPT-01 | — | Pending |
| REPT-02 | — | Pending |
| REPT-03 | — | Pending |

**Coverage:**
- v1.2 requirements: 31 total
- Mapped to phases: 0
- Unmapped: 31 (pending roadmap creation)

---
*Requirements defined: 2026-03-05*
*Last updated: 2026-03-05 after initial definition*
