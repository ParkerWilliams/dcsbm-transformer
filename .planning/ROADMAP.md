# Roadmap: DCSBM Transformer

## Milestones

- ✅ **v1.0 MVP** — Phases 1-9 (shipped 2026-02-25)
- ✅ **v1.1 Journal Feedback** — Phases 11-17 (shipped 2026-02-28)
- 🚧 **v1.2 Mathematical Audit** — Phases 18-23 (in progress)

## Phases

<details>
<summary>✅ v1.0 MVP (Phases 1-9) — SHIPPED 2026-02-25</summary>

- [x] Phase 1: Config Schema & Reproducibility Foundation (2/2 plans) — completed
- [x] Phase 2: DCSBM Graph Generation (2/2 plans) — completed
- [x] Phase 3: Walk Generation (2/2 plans) — completed
- [x] Phase 4: Transformer Model (3/3 plans) — completed
- [x] Phase 5: Training Pipeline (2/2 plans) — completed
- [x] Phase 6: Behavioral Evaluation & SVD Collection (3/3 plans) — completed
- [x] Phase 7: Predictive Horizon & Statistical Analysis (2/2 plans) — completed
- [x] Phase 8: Visualization (2/2 plans) — completed
- [x] Phase 9: Reporting & Math Verification (2/2 plans) — completed

See `.planning/milestones/` for archived details.

</details>

<details>
<summary>✅ v1.1 Journal Feedback (Phases 11-17) — SHIPPED 2026-02-28</summary>

- [x] Phase 11: Pre-Registration Framework (1/1 plan) — completed 2026-02-26
- [x] Phase 12: Null Model Baseline (2/2 plans) — completed 2026-02-26
- [x] Phase 13: Evaluation Enrichment (3/3 plans) — completed 2026-02-26
- [x] Phase 14: Softmax Filtering Bound (2/2 plans) — completed 2026-02-26
- [x] Phase 15: Advanced Analysis (3/3 plans) — completed 2026-02-27
- [x] Phase 16: Multi-Head Ablation (3/3 plans) — completed 2026-02-27
- [x] Phase 17: E2E Pipeline Wiring (1/1 plan) — completed 2026-02-27

See `.planning/milestones/v1.1-ROADMAP.md` for archived details.

</details>

### 🚧 v1.2 Mathematical Audit (In Progress)

**Milestone Goal:** Exhaustive mathematical correctness review of every formula, derivation, and implementation in the codebase — audit and fix all issues found.

- [ ] **Phase 18: Graph & Walk Foundations** - Verify DCSBM edge probabilities, walk sampling, jumper designation, behavioral classification, and compliance rates
- [ ] **Phase 19: SVD Metric Extraction** - Verify QK^T construction, WvWo/AVWo matrices, all singular-value-derived metrics, Grassmannian distance, float16 fidelity, and Frenet-Serret curvature/torsion
- [ ] **Phase 20: AUROC & Predictive Horizon** - Verify AUROC rank-based computation, lookback indexing, horizon definition consistency, and event extraction logic
- [ ] **Phase 21: Statistical Controls** - Verify shuffle permutation null, bootstrap BCa intervals, Holm-Bonferroni correction, Cohen's d, Spearman redundancy threshold, and exploratory/confirmatory split
- [ ] **Phase 22: Softmax Bound & Null Model** - Verify LaTeX derivation correctness, empirical bound verification code, bound assumptions, Grassmannian drift parity, Mann-Whitney U, column-filtered adjacency, and Holm-Bonferroni family separation
- [ ] **Phase 23: Audit Report Generation** - Generate self-contained HTML report linking every formula to its implementation with correctness verdicts

## Phase Details

### Phase 18: Graph & Walk Foundations
**Goal**: Every graph-theoretic formula and walk-generation algorithm is verified correct against its mathematical definition, with all issues fixed
**Depends on**: Nothing (first audit phase; reads existing code only)
**Requirements**: GRAPH-01, GRAPH-02, GRAPH-03, GRAPH-04, GRAPH-05
**Success Criteria** (what must be TRUE):
  1. DCSBM edge probability P_ij = theta_i * theta_j * B_{z_i, z_j} matches implementation (degree correction applied correctly, matrix is symmetric, no off-by-one in block indexing)
  2. Walk sampling draws neighbors uniformly from the adjacency list without artificial bias (verified by comparing empirical neighbor frequencies to expected distribution)
  3. Block jumper designation assigns correct jump distance r and target block per the specification, and behavioral classification (followed/violated/unconstrained/pending) correctly labels every step
  4. Walk compliance rate formula (violations / constrained steps) matches the code computation exactly
**Plans:** 2 plans

Plans:
- [ ] 18-01-PLAN.md — Audit DCSBM probability, walk sampling, jumper designation, and compliance rate math
- [ ] 18-02-PLAN.md — Expand RuleOutcome to 4-class behavioral classification and update consumers

### Phase 19: SVD Metric Extraction
**Goal**: Every SVD-related metric formula and matrix construction is verified correct, including numerical fidelity of spectrum storage
**Depends on**: Phase 18 (graph/walk correctness is prerequisite for meaningful SVD analysis)
**Requirements**: SVD-01, SVD-02, SVD-03, SVD-04, SVD-05, SVD-06
**Success Criteria** (what must be TRUE):
  1. QK^T attention matrix is constructed as Q @ K.transpose(-2, -1) from the correct projection outputs (not post-softmax, not post-dropout), and WvWo / AVWo matrices match their documented definitions
  2. All five singular-value-derived metrics (condition number, spectral gap, entropy, effective rank, stable rank) use textbook-correct formulas verified against reference implementations
  3. Grassmannian distance between consecutive steps uses the canonical principal-angle definition (arccos of clipped singular values of U1^T @ U2)
  4. Float16 vs float32 impact on downstream curvature/torsion is quantified, with a clear recommendation documented
  5. Frenet-Serret curvature and torsion use correct discrete differential geometry formulas (finite differences of tangent, normal, binormal vectors on the spectrum trajectory curve)
**Plans**: TBD

Plans:
- [ ] 19-01: TBD
- [ ] 19-02: TBD

### Phase 20: AUROC & Predictive Horizon
**Goal**: The AUROC predictive horizon pipeline is verified correct from event extraction through lookback indexing to horizon determination
**Depends on**: Phase 19 (SVD metrics must be correct before verifying the analysis that consumes them)
**Requirements**: AUROC-01, AUROC-02, AUROC-03, AUROC-04
**Success Criteria** (what must be TRUE):
  1. AUROC computation uses rank-based P(X_violated > X_followed) and matches sklearn.metrics.roc_auc_score on identical inputs
  2. Lookback distance j correctly retrieves the metric value at step (t-j) relative to resolution step t, with no fence-post error in indexing
  3. Predictive horizon (max j where AUROC > 0.75) is applied identically in primary analysis, null model comparison, and multi-head ablation code paths
  4. Event extraction correctly identifies resolution steps from behavioral labels, matching the 4-class classification verified in Phase 18
**Plans**: TBD

Plans:
- [ ] 20-01: TBD

### Phase 21: Statistical Controls
**Goal**: All statistical testing machinery produces mathematically correct results — permutation tests, confidence intervals, multiple comparison corrections, effect sizes, and study design splits
**Depends on**: Phase 20 (statistical controls operate on AUROC and metric outputs)
**Requirements**: STAT-01, STAT-02, STAT-03, STAT-04, STAT-05, STAT-06
**Success Criteria** (what must be TRUE):
  1. Shuffle permutation null correctly permutes event labels (not metric values) while preserving group sizes, and the resulting p-value distribution is uniform under H0
  2. Bootstrap BCa confidence intervals apply correct bias correction (z0) and acceleration (a) per Efron's original method
  3. Holm-Bonferroni step-down correction applies sorted p-values against thresholds alpha/(m-k+1) for k=1..m, rejecting in correct order
  4. Cohen's d uses the pooled standard deviation formula s_p = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
  5. Exploratory/confirmatory split uses reproducible random assignment with correct proportions, and Spearman |r| > 0.9 redundancy threshold is applied correctly
**Plans**: TBD

Plans:
- [ ] 21-01: TBD
- [ ] 21-02: TBD

### Phase 22: Softmax Bound & Null Model
**Goal**: The softmax filtering bound derivation and null model baseline methodology are mathematically sound and correctly implemented
**Depends on**: Phase 19 (SVD extraction correctness), Phase 20 (AUROC correctness)
**Requirements**: SFTX-01, SFTX-02, SFTX-03, NULL-01, NULL-02, NULL-03, NULL-04
**Success Criteria** (what must be TRUE):
  1. LaTeX derivation of QK^T to AVWo epsilon-bound is step-by-step correct (each inequality follows from the previous, matrix norm properties applied correctly, bound assumptions explicitly stated)
  2. Empirical verification code tests the bound predictions against observed data using the same matrix constructions verified in Phase 19, and implementation respects stated assumptions
  3. Null model Grassmannian drift uses identical SVD extraction code path as primary analysis (same function calls, same parameters), differing only in input walks
  4. Mann-Whitney U test statistic and p-value computation match scipy.stats.mannwhitneyu on identical inputs
  5. Column-filtered adjacency provably removes all jumper paths (zero contamination), and null model uses a separate Holm-Bonferroni family from primary analysis
**Plans**: TBD

Plans:
- [ ] 22-01: TBD
- [ ] 22-02: TBD

### Phase 23: Audit Report Generation
**Goal**: A self-contained HTML audit report documents every formula-to-code mapping with correctness verdicts
**Depends on**: Phases 18-22 (all audit findings must be complete before report generation)
**Requirements**: REPT-01, REPT-02, REPT-03
**Success Criteria** (what must be TRUE):
  1. HTML report renders every mathematical formula in LaTeX (via MathJax) alongside its code location (file:line), with a correctness verdict (correct / fixed / concern) and fix description where applicable
  2. Report is organized by audit category (Graph, SVD, AUROC, Statistical, Softmax, Null Model) with clickable navigation
  3. Report is fully self-contained (inline CSS/JS, bundled MathJax) and opens correctly in a browser without network access
**Plans**: TBD

Plans:
- [ ] 23-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 18 -> 19 -> 20 -> 21 -> 22 -> 23

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1-9   | v1.0      | 20/20          | Complete | 2026-02-25 |
| 11-17 | v1.1      | 15/15          | Complete | 2026-02-28 |
| 18. Graph & Walk Foundations | v1.2 | 0/2 | Planned | - |
| 19. SVD Metric Extraction | v1.2 | 0/TBD | Not started | - |
| 20. AUROC & Predictive Horizon | v1.2 | 0/TBD | Not started | - |
| 21. Statistical Controls | v1.2 | 0/TBD | Not started | - |
| 22. Softmax Bound & Null Model | v1.2 | 0/TBD | Not started | - |
| 23. Audit Report Generation | v1.2 | 0/TBD | Not started | - |
