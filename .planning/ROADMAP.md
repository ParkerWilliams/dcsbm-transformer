# Roadmap: DCSBM Transformer v1.1 -- Journal Feedback

## Overview

This roadmap addresses convergent reviewer concerns about the DCSBM transformer SVD hallucination prediction paper. The six phases progress through methodological prerequisites (pre-registration), core signal validation (null model baseline), additive evaluation enrichments (PR curves, calibration, SVD benchmarks), theoretical formalization (softmax filtering bound), advanced analysis features (spectrum trajectory, compliance curve), and finally the most invasive architectural change (multi-head ablation). Pre-registration is committed before any confirmatory analysis. The null model validates the core signal claim that all subsequent work depends on. All additive features are built and validated on the single-head architecture before multi-head support touches 10+ files in the final phase.

## Phases

**Phase Numbering:**
- v1.0 phases (1-10): Archived -- see git history
- v1.1 phases (11-16): Current milestone
- Decimal phases (e.g., 12.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 11: Pre-Registration Framework** - Lock primary hypothesis, held-out protocol, and deviation log before any v1.1 confirmatory analysis (completed 2026-02-26)
- [x] **Phase 12: Null Model Baseline** - Validate the core SVD signal claim with jumper-free null distribution and statistical comparison (completed 2026-02-26)
- [x] **Phase 13: Evaluation Enrichment** - Add precision-recall curves, calibration diagnostics, and SVD computational overhead benchmarks (completed 2026-02-26)
- [x] **Phase 14: Softmax Filtering Bound** - Derive and empirically verify the epsilon-bound from QK^T perturbation through softmax to AVWo spectral change (completed 2026-02-26)
- [ ] **Phase 15: Advanced Analysis** - Full spectrum trajectory with curvature/torsion and sharp compliance curve across r/w ratio sweep
- [ ] **Phase 16: Multi-Head Ablation** - Extend transformer to 2h/4h with per-head SVD extraction and signal concentration analysis

## Phase Details

### Phase 11: Pre-Registration Framework
**Goal**: The primary hypothesis, analysis plan, and held-out evaluation protocol are locked in git history before any v1.1 confirmatory analysis runs
**Depends on**: Nothing (first v1.1 phase; methodological prerequisite for all subsequent phases)
**Requirements**: PREG-01, PREG-02, PREG-03
**Success Criteria** (what must be TRUE):
  1. A pre-registration document exists in git specifying Grassmannian distance of QK^T as the primary hypothesis, the primary metric, alpha level (0.05), Holm-Bonferroni correction method, and the decision criterion for confirming/rejecting the hypothesis
  2. The evaluation pipeline splits walks into exploratory (50%) and confirmatory (50%) sets, and result.json tags each analysis result with its split membership (exploratory vs confirmatory)
  3. A deviation log file exists and is referenced from the pre-registration document, ready to record any changes to the analysis plan with timestamped rationale
**Plans**: TBD

Plans:
- [ ] 11-01: Pre-registration document, held-out split implementation, and deviation log

### Phase 12: Null Model Baseline
**Goal**: The SVD Grassmannian drift signal is demonstrated to be a real response to block jumper events, not an artifact of normal attention dynamics, through statistical comparison against a null distribution from jumper-free sequences
**Depends on**: Phase 11
**Requirements**: NULL-01, NULL-02, NULL-03, NULL-04
**Success Criteria** (what must be TRUE):
  1. The system generates evaluation walks with zero block jumpers (same graph, same trained model) and computes a Grassmannian drift distribution from these null sequences
  2. Position-matched Mann-Whitney U tests and Cohen's d effect sizes compare null vs violation Grassmannian drift at each lookback distance, and the results quantify whether the signal exceeds background noise
  3. A Marchenko-Pastur reference distribution for QK^T singular values at the anchor config matrix dimensions (w x w, aspect ratio from d_k) is computed and available for comparison
  4. Null model results are stored in a `null_model` block in result.json, and event-aligned plots render a null distribution overlay alongside the violation signal
**Plans**: TBD

Plans:
- [ ] 12-01: Null walk generation, null Grassmannian distribution, and Marchenko-Pastur reference
- [ ] 12-02: Statistical comparison (Mann-Whitney U, Cohen's d), result.json storage, and null overlay visualization

### Phase 13: Evaluation Enrichment
**Goal**: Violation prediction quality is assessed beyond AUROC with precision-recall curves and calibration diagnostics, and SVD computational cost is benchmarked with cheaper approximation candidates identified
**Depends on**: Phase 12
**Requirements**: PRCL-01, PRCL-02, PRCL-03, OVHD-01, OVHD-02, OVHD-03
**Success Criteria** (what must be TRUE):
  1. Precision-recall curves and AUPRC are computed per metric per lookback distance using the same event extraction as existing AUROC, and results are stored alongside AUROC in result.json
  2. Reliability diagrams (calibration curves) with Expected Calibration Error (ECE) are generated for violation prediction, showing whether predicted probabilities match observed frequencies
  3. PR curves and reliability diagrams are rendered in HTML reports alongside existing AUROC plots
  4. Wall-clock SVD cost per step is benchmarked by target (QK^T, WvWo, AVWo) and matrix dimension using CUDA events with warmup, and full SVD vs randomized SVD (torch.svd_lowrank) vs values-only SVD (torch.linalg.svdvals) are compared with accuracy-cost tradeoff reported
  5. A cost summary table (matrix size, time per step, percentage of total evaluation time) is included in HTML reports
**Plans**: TBD

Plans:
- [ ] 13-01: Precision-recall curves, AUPRC computation, and report integration
- [ ] 13-02: Calibration diagnostics (reliability diagrams, ECE) and report integration
- [ ] 13-03: SVD overhead benchmarks (full vs randomized vs values-only) and cost summary table

### Phase 14: Softmax Filtering Bound
**Goal**: The theoretical relationship between QK^T perturbation and downstream AVWo spectral change is formalized as an epsilon-bound with a LaTeX derivation, and the bound is empirically verified against controlled perturbation experiments
**Depends on**: Phase 12
**Requirements**: SFTX-01, SFTX-02, SFTX-03
**Success Criteria** (what must be TRUE):
  1. A LaTeX derivation exists showing the epsilon-bound from QK^T perturbation through softmax (Lipschitz constant 1/2) to AVWo spectral change, incorporating the 1/sqrt(d_k) scaling factor and the correct chain through value projection and output projection
  2. The bound is empirically verified by injecting controlled perturbations (random and adversarial directions) into QK^T at specific steps and measuring actual AVWo spectral change vs the theoretical bound, with fewer than 5% of perturbations exceeding the bound
  3. A bound tightness visualization is generated showing the theoretical envelope vs empirical measurements, and the tightness ratio (median empirical / theoretical bound) is reported
**Plans**: TBD

Plans:
- [ ] 14-01: LaTeX derivation of softmax filtering epsilon-bound
- [ ] 14-02: Empirical bound verification and tightness visualization

### Phase 15: Advanced Analysis
**Goal**: The spectral trajectory is tracked as full singular value vectors with geometric curvature/torsion analysis feeding into the AUROC pipeline, and the compliance phase transition from near-perfect to failure is characterized across fine-grained r/w ratio sweep
**Depends on**: Phase 13
**Requirements**: SPEC-01, SPEC-02, SPEC-03, COMP-01, COMP-02
**Success Criteria** (what must be TRUE):
  1. Full singular value vectors (sigma_1 through sigma_k) are stored per step in NPZ alongside existing scalar metrics, configurable per target (QK^T by default), and storage overhead is managed (float16, selective targets)
  2. Discrete Frenet-Serret curvature and torsion are computed on the spectral trajectory curve in R^k with appropriate numerical smoothing, and the resulting time series are stored as additional metrics
  3. Curvature and torsion time series are fed into the AUROC pipeline as secondary predictive metrics, and their predictive power is compared against existing scalar metrics
  4. The r/w ratio is swept with at least 8 values spanning r << w through r >> w, with 3 seeds per value, and a composite publication figure shows compliance rate and predictive horizon as a function of r/w ratio with dual y-axes
**Plans**: TBD

Plans:
- [ ] 15-01: Full spectrum storage in NPZ and discrete curvature/torsion computation
- [ ] 15-02: Curvature/torsion as AUROC predictive metrics
- [ ] 15-03: Compliance curve sweep and dual-axis publication figure

### Phase 16: Multi-Head Ablation
**Goal**: The transformer supports multi-head attention (1h/2h/4h) with per-head SVD extraction, and an ablation study demonstrates whether the predictive signal concentrates in specific heads or distributes across all heads
**Depends on**: Phase 15
**Requirements**: MHAD-01, MHAD-02, MHAD-03, MHAD-04
**Success Criteria** (what must be TRUE):
  1. The transformer accepts n_heads = 1, 2, or 4 with d_k held constant at 128 (d_model scales as n_heads * d_k: 1h=128, 2h=256, 4h=512), and per-head QK^T matrices are extractable for SVD analysis
  2. SVD metrics are computed per-head with NPZ keys in format `target.layer_N.head_H.metric_name`, and single-head runs emit backward-compatible dual keys (both legacy flat format and new per-head format)
  3. Per-head AUROC is computed and signal concentration analysis (entropy and Gini coefficient of AUROC distribution across heads) identifies which heads carry predictive signal
  4. An ablation comparison runs on matched configs (same graph, same walks -- 1h d_model=128, 2h d_model=256, 4h d_model=512, all d_k=128) and reports per-head vs aggregate signal strength, with results determining whether single-head multiplexing is an artifact
**Plans**: TBD

Plans:
- [ ] 16-01: Multi-head CausalSelfAttention with per-head QK^T extraction
- [ ] 16-02: Per-head SVD metric computation with dual key emission
- [ ] 16-03: Per-head AUROC, signal concentration analysis, and ablation comparison

## Progress

**Execution Order:**
Phases execute in numeric order: 11 -> 12 -> 13 -> 14 -> 15 -> 16
Note: Phase 14 depends on Phase 12 (not 13), so Phases 13 and 14 could theoretically be parallelized after Phase 12 completes.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 11. Pre-Registration Framework | 1/1 | Complete    | 2026-02-26 |
| 12. Null Model Baseline | 2/2 | Complete    | 2026-02-26 |
| 13. Evaluation Enrichment | 3/3 | Complete    | 2026-02-26 |
| 14. Softmax Filtering Bound | 2/2 | Complete    | 2026-02-26 |
| 15. Advanced Analysis | 0/3 | Not started | - |
| 16. Multi-Head Ablation | 0/3 | Not started | - |
