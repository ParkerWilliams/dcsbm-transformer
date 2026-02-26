# Requirements: DCSBM Transformer SVD Hallucination Prediction

**Defined:** 2026-02-24 (v1.0), updated 2026-02-26 (v1.1)
**Core Value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.

## v1.0 Requirements (Validated)

Shipped and verified in v1.0 milestone (phases 1-9). Listed for reference.

### Graph Generation

- [x] **GRPH-01**: System generates DCSBM directed graphs with configurable number of vertices (n), blocks (K), in-group probability (p_in), out-group probability (p_out), and degree correction parameters per Karrer & Newman 2011
- [x] **GRPH-02**: System designates block jumper vertices with configurable jump length r and target block, where the rule "after r steps from jumper v_i in block b, the walk must land in a specific target block different from b" is enforced
- [x] **GRPH-03**: System validates graph connectivity (strongly connected), minimum expected degree >= 3, and edge density matching expected p_in/p_out ratios
- [x] **GRPH-04**: System verifies non-triviality of block jumper rules: valid paths of length r from jumper to target block exist but are not the only paths at that length
- [x] **GRPH-05**: System caches generated graphs by config hash to avoid redundant regeneration across sweep configs sharing the same graph parameters

### Walk Generation

- [x] **WALK-01**: System generates directed random walks on the DCSBM graph with configurable walk length l (swept at 2w, 4w, 8w)
- [x] **WALK-02**: System validates corpus size is at least 2 orders of magnitude larger than n (t >= 100n)
- [x] **WALK-03**: System produces separate train and evaluation walk sets with different seeds
- [x] **WALK-04**: System tracks block jumper encounter metadata during walk generation (which jumper was hit, at which step, expected target block at step+r)
- [x] **WALK-05**: System caches generated walks by config hash to avoid redundant regeneration

### Model

- [x] **MODL-01**: System implements a NanoGPT-scale transformer with configurable d_model (64, 128, 256), n_layers (2, 4, 6), and exactly 1 attention head
- [x] **MODL-02**: Model's single attention head exposes internal components for SVD analysis via a `return_internals=True` flag: raw QK^T matrix (causal-masked, zero-filled), attention weights A, value matrix V, and access to Wv/Wo weight parameters — enabling three SVD targets (QK^T routing, WvWo OV circuit, AVWo net residual update)
- [x] **MODL-03**: Model vocabulary equals the number of graph vertices (tokens are vertex IDs)

### Training

- [x] **TRNG-01**: System trains the transformer using cross-entropy next-token prediction with AdamW optimizer and cosine learning rate schedule
- [x] **TRNG-02**: System controls all random seeds (torch, numpy, python random, CUDA deterministic) from a single master seed for reproducibility
- [x] **TRNG-03**: System checkpoints model weights, optimizer state, and training step periodically and on gate pass
- [x] **TRNG-04**: System logs training loss and compliance curves per step, stored in result.json curves block
- [x] **TRNG-05**: System enforces training sufficiency gate: edge compliance >95% and rule compliance >80% on held-out walks, evaluated periodically during training
- [x] **TRNG-06**: Configurations that fail the sufficiency gate after allocated training budget are flagged and excluded from SVD analysis, with failure metadata recorded in result.json
- [x] **TRNG-07**: System tracks git code hash (short SHA) and stores it with results for reproducibility

### Behavioral Evaluation

- [x] **EVAL-01**: System classifies each generation step into 4-class outcome: edge valid/invalid crossed with rule followed/violated/not-applicable
- [x] **EVAL-02**: System checks edge validity (chosen next token corresponds to a valid directed edge in DCSBM) at each step
- [x] **EVAL-03**: System checks rule compliance (at step r from a block jumper vertex, the walk lands in the required target block) at each step
- [x] **EVAL-04**: System annotates each generated sequence with failure_index (index of first rule violation), or null for correct sequences
- [x] **EVAL-05**: Behavioral evaluation and SVD collection happen in a single fused forward pass (not separate inference runs)

### SVD Metrics

- [x] **SVD-01**: System computes SVD on three targets per attention layer at every token step during evaluation: (1) QK^T — routing stability, (2) WvWo — OV circuit stability (input-agnostic), (3) AVWo — net residual stream update stability
- [x] **SVD-02**: System computes SVD using torch.linalg.svd with full_matrices=False, batched for efficiency on GPU
- [x] **SVD-03**: System computes 7 scalar metrics per SVD target per token step: stable rank (||M||^2_F/||M||^2_2), spectral entropy (-Sum p_i log p_i where p_i=sigma_i/Sum sigma), spectral gap (sigma_1-sigma_2 and generalized sigma_k-sigma_{k+1} for k=2,4), condition number (sigma_1/sigma_n), rank-1 residual norm (||M-sigma_1 u_1 v_1^T||_F/||M||_F), and read-write subspace alignment (WvWo only: cosine angle between top left and right singular vectors in d_model space)
- [x] **SVD-04**: System stores all SVD metrics as token-level time series in result.json/token_metrics.npz keyed by target and metric name (e.g., qkt.stable_rank, wvwo.spectral_entropy, avwo.condition_number)
- [x] **SVD-05**: System includes numerical guards: NaN/Inf clamping, epsilon in entropy computation, condition number capped at 1e6, Grassmannian distance for subspace tracking
- [x] **SVD-06**: System collects SVD metrics only for positions >= w (context window warmup) to avoid padding artifacts
- [x] **SVD-07**: Each SVD metric function has unit tests against analytically known matrix decompositions for each target type

### Predictive Horizon Analysis

- [x] **PRED-01**: System computes AUROC at each lookback distance j (from 1 to r) for each SVD metric, comparing metric values at step (t-j) for violation vs non-violation events
- [x] **PRED-02**: System calculates predictive horizon as the furthest j at which AUROC exceeds 0.75 for each metric
- [x] **PRED-03**: System uses position-matched baselines (control events sampled at same absolute position in non-jumper walks) to control for positional confounds
- [x] **PRED-04**: System runs shuffle controls (permuted labels) to verify signal is not positional artifact (AUROC > 0.6 on shuffled = flag)
- [x] **PRED-05**: System stores per-metric AUROC curves in result.json metrics block

### Experiment Management

- [x] **MGMT-01**: System defines experiment configuration as a frozen, serializable, hashable dataclass with all governing parameters
- [ ] **MGMT-02**: System implements parameter sweep with declarative definition of parameter ranges matching the spec sweep ranges
- [ ] **MGMT-03**: System implements a priority-ordered job queue: anchor config first, then core r-vs-w sweep (Tier 1), then architecture/w sweeps (Tier 2), then secondary sweeps (Tier 3)
- [ ] **MGMT-04**: System runs 3 random seeds per configuration
- [x] **MGMT-05**: System writes result.json per configuration conforming to the project schema (schema_version, experiment_id, timestamp, description, tags, config, metrics, sequences, metadata)
- [ ] **MGMT-06**: System persists sweep state for resume after RunPod preemption

### Visualization

- [x] **PLOT-01**: System generates event-aligned SVD metric plots (position 0 = failure event, negative = before, positive = after) with confidence bands and correct-sequence baseline overlay
- [x] **PLOT-02**: System generates training convergence curves (loss and compliance over steps)
- [x] **PLOT-03**: System generates AUROC vs lookback distance j curves per SVD metric
- [x] **PLOT-04**: System generates confusion matrix for 4-class behavioral outcomes
- [x] **PLOT-05**: System generates pre/post failure distribution comparison plots
- [x] **PLOT-06**: System generates predictive horizon heatmap across (r, w) parameter grid
- [x] **PLOT-07**: All plots follow the project style baseline (seaborn whitegrid, consistent palette, no default matplotlib style)
- [x] **PLOT-08**: All figures saved as both PNG (300 dpi) and SVG to results/{experiment_id}/figures/

### Reporting

- [x] **REPT-01**: System generates self-contained single-experiment HTML report with base64-embedded figures, covering: header, configuration, scalar metrics, curves, confusion matrix, statistical tests, sequence analysis, and reproduction command
- [x] **REPT-02**: System generates comparison HTML report across multiple experiments with: scalar metrics comparison table, curve overlays, config diff table, and aligned sequence plot overlays
- [x] **REPT-03**: Every report includes a reproduction block with git checkout command and full CLI arguments

### Statistical Rigor

- [x] **STAT-01**: System applies Holm-Bonferroni correction for multiple comparisons across pre-registered primary metrics (3-5 metrics selected before sweep)
- [x] **STAT-02**: System computes bootstrap confidence intervals on AUROC and predictive horizon estimates
- [x] **STAT-03**: System reports effect sizes (Cohen's d) for pre-failure vs post-failure metric distributions
- [x] **STAT-04**: System computes SVD metric correlation matrix to identify redundant metrics
- [x] **STAT-05**: System produces metric importance ranking by max AUROC across j values

### Math Verification

- [x] **MATH-01**: System generates a peer-review PDF containing: title page, table of contents, one section per math-heavy source file with plain-language summary, full code block, LaTeX representation of implemented mathematics, and appendix listing all other source files
- [x] **MATH-02**: PDF title page notes clearly that LaTeX was AI-generated and requires researcher sign-off

## v1.1 Requirements

Requirements for journal feedback milestone. Each maps to roadmap phases.

### Null Model Baseline

- [ ] **NULL-01**: System generates evaluation walks with zero block jumpers (same graph, same trained model) to produce null Grassmannian drift distribution
- [ ] **NULL-02**: System computes position-matched statistical comparison (Mann-Whitney U, Cohen's d) of Grassmannian drift between null and violation sequences at each lookback distance
- [ ] **NULL-03**: System computes Marchenko-Pastur reference distribution for QK^T singular values at the anchor config matrix dimensions
- [ ] **NULL-04**: System stores null model results in result.json `null_model` block and renders null overlay on event-aligned plots

### Softmax Filtering Bound

- [ ] **SFTX-01**: System includes a LaTeX derivation of the epsilon-bound from QK^T perturbation through softmax to AVWo spectral change, using the correct Lipschitz constant (1/2) and 1/sqrt(d_k) scaling
- [ ] **SFTX-02**: System empirically verifies the bound by injecting controlled perturbations into QK^T at specific steps and measuring actual AVWo spectral change vs theoretical bound
- [ ] **SFTX-03**: System generates bound tightness visualization (theoretical envelope vs empirical measurements) and reports tightness ratio

### Multi-Head Ablation

- [ ] **MHAD-01**: Transformer supports configurable n_heads (1, 2, 4) with per-head QK^T extraction; d_k is held constant (d_model scales as n_heads * d_k) to ensure equal per-head dimensionality across ablation configs
- [ ] **MHAD-02**: SVD metrics are computed per-head with NPZ keys in format `target.layer_N.head_H.metric_name`, with backward-compatible dual key emission for single-head runs
- [ ] **MHAD-03**: System computes per-head AUROC and signal concentration analysis (entropy/Gini of AUROC distribution across heads) to identify which heads carry predictive signal
- [ ] **MHAD-04**: System runs ablation comparison on matched configs (same graph, walks — 1h d_model=128, 2h d_model=256, 4h d_model=512, all d_k=128) and reports per-head vs aggregate signal strength

### Precision-Recall and Calibration

- [ ] **PRCL-01**: System computes precision-recall curves and AUPRC per metric per lookback distance, using the same event extraction as existing AUROC
- [ ] **PRCL-02**: System generates reliability diagrams (calibration curves) with Expected Calibration Error (ECE) for violation prediction
- [ ] **PRCL-03**: PR curves and reliability diagrams are integrated into HTML reports alongside existing AUROC plots

### Pre-Registration Framework

- [ ] **PREG-01**: Pre-registration document specifying primary hypothesis (Grassmannian distance of QK^T), primary metric, alpha level, correction method, and decision criterion is committed to git before any v1.1 confirmatory analysis runs
- [ ] **PREG-02**: System implements held-out evaluation split (exploratory / confirmatory walks) and tags results with split membership
- [ ] **PREG-03**: System maintains a deviation log tracking any changes to the pre-registered analysis plan with rationale

### Sharp Compliance Curve

- [ ] **COMP-01**: System sweeps r/w ratio with fine granularity (at least 8 values spanning r << w through r >> w) and 3 seeds per value to establish the compliance phase transition
- [ ] **COMP-02**: System generates composite publication figure showing compliance rate and predictive horizon as a function of r/w ratio with dual y-axes

### Full Spectrum Trajectory

- [ ] **SPEC-01**: System stores full singular value vectors sigma_1...sigma_k per step in NPZ alongside existing scalar metrics, configurable per target (QK^T by default)
- [ ] **SPEC-02**: System computes discrete Frenet-Serret curvature and torsion on the spectral trajectory curve in R^k with appropriate numerical smoothing
- [ ] **SPEC-03**: Curvature and torsion time series are fed into the AUROC pipeline as additional (secondary) predictive metrics

### SVD Computational Overhead

- [ ] **OVHD-01**: System benchmarks wall-clock SVD cost per step, broken down by target (QK^T, WvWo, AVWo) and matrix dimension, using proper GPU timing (CUDA events with warmup)
- [ ] **OVHD-02**: System compares full SVD vs randomized SVD (torch.svd_lowrank) vs values-only SVD (torch.linalg.svdvals) and reports accuracy-cost tradeoff
- [ ] **OVHD-03**: Cost summary table (matrix size, time per step, % of total evaluation time) is included in HTML reports

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Analysis

- **ADV-01**: Grassmannian trajectory visualization (low-dimensional embedding of subspace trajectory)
- **ADV-02**: Automated parameter sensitivity analysis (variance decomposition of config parameters with predictive horizon)
- **ADV-03**: Automated budget tracking dashboard with real-time cost display

### Generalization Studies

- **GENR-01**: Multiple rule types beyond block jumpers (parity constraints, subsequence bans, cycle return)
- **GENR-02**: Multi-head beyond 4 heads (requires larger d_model for meaningful d_k)
- **GENR-03**: Cross-task signal robustness analysis across rule type variations

### Sweep Infrastructure (deferred from v1.0)

- **MGMT-02**: System implements parameter sweep with declarative definition of parameter ranges
- **MGMT-03**: System implements a priority-ordered job queue
- **MGMT-04**: System runs 3 random seeds per configuration
- **MGMT-06**: System persists sweep state for resume after RunPod preemption

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-head beyond 4 heads | d_k too small at d_model=128; 1h/2h/4h demonstrates the principle |
| Full Bayesian calibration (Platt scaling) | Undermines pre-registration claims; use raw reliability diagrams |
| Real-time SVD monitoring | v1.1 is offline research, not deployment |
| Symbolic math verification (SymPy) | Manual LaTeX is faster for one derivation |
| Gradient-based attribution for SVD metrics | Different research question (why vs whether) |
| General-purpose spectral trajectory library | Premature generalization |
| Multiple rule types | Deferred to v2; focus on proving single-rule signal first |
| Full production sweep infrastructure | Deferred; v1.1 runs targeted ablation configs, not full grid |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| NULL-01 | TBD | Pending |
| NULL-02 | TBD | Pending |
| NULL-03 | TBD | Pending |
| NULL-04 | TBD | Pending |
| SFTX-01 | TBD | Pending |
| SFTX-02 | TBD | Pending |
| SFTX-03 | TBD | Pending |
| MHAD-01 | TBD | Pending |
| MHAD-02 | TBD | Pending |
| MHAD-03 | TBD | Pending |
| MHAD-04 | TBD | Pending |
| PRCL-01 | TBD | Pending |
| PRCL-02 | TBD | Pending |
| PRCL-03 | TBD | Pending |
| PREG-01 | TBD | Pending |
| PREG-02 | TBD | Pending |
| PREG-03 | TBD | Pending |
| COMP-01 | TBD | Pending |
| COMP-02 | TBD | Pending |
| SPEC-01 | TBD | Pending |
| SPEC-02 | TBD | Pending |
| SPEC-03 | TBD | Pending |
| OVHD-01 | TBD | Pending |
| OVHD-02 | TBD | Pending |
| OVHD-03 | TBD | Pending |

**Coverage:**
- v1.1 requirements: 25 total
- Mapped to phases: 0 (awaiting roadmap)
- Unmapped: 25

---
*Requirements defined: 2026-02-24 (v1.0)*
*Last updated: 2026-02-26 after v1.1 Journal Feedback milestone definition*
