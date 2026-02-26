# Requirements: DCSBM Transformer SVD Hallucination Prediction

**Defined:** 2026-02-24
**Core Value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

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
- [x] **SVD-03**: System computes 7 scalar metrics per SVD target per token step: stable rank (||M||²_F/||M||²_2), spectral entropy (-Σ pᵢ log pᵢ where pᵢ=σᵢ/Σσ), spectral gap (σ₁-σ₂ and generalized σₖ-σₖ₊₁ for k=2,4), condition number (σ₁/σₙ), rank-1 residual norm (||M-σ₁u₁v₁ᵀ||_F/||M||_F), and read-write subspace alignment (WvWo only: cosine angle between top left and right singular vectors in d_model space)
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
- [ ] **REPT-02**: System generates comparison HTML report across multiple experiments with: scalar metrics comparison table, curve overlays, config diff table, and aligned sequence plot overlays
- [x] **REPT-03**: Every report includes a reproduction block with git checkout command and full CLI arguments

### Statistical Rigor

- [x] **STAT-01**: System applies Holm-Bonferroni correction for multiple comparisons across pre-registered primary metrics (3-5 metrics selected before sweep)
- [x] **STAT-02**: System computes bootstrap confidence intervals on AUROC and predictive horizon estimates
- [x] **STAT-03**: System reports effect sizes (Cohen's d) for pre-failure vs post-failure metric distributions
- [x] **STAT-04**: System computes SVD metric correlation matrix to identify redundant metrics
- [x] **STAT-05**: System produces metric importance ranking by max AUROC across j values

### Math Verification

- [ ] **MATH-01**: System generates a peer-review PDF containing: title page, table of contents, one section per math-heavy source file with plain-language summary, full code block, LaTeX representation of implemented mathematics, and appendix listing all other source files
- [ ] **MATH-02**: PDF title page notes clearly that LaTeX was AI-generated and requires researcher sign-off

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Advanced Analysis

- **ADV-01**: Grassmannian trajectory visualization (low-dimensional embedding of subspace trajectory)
- **ADV-02**: Automated parameter sensitivity analysis (variance decomposition of config parameters with predictive horizon)
- **ADV-03**: Automated budget tracking dashboard with real-time cost display

## Out of Scope

| Feature | Reason |
|---------|--------|
| Multi-head attention support | Single head is intentional and essential for unambiguous QK^T analysis |
| Real-world / clinical data ingestion | Synthetic controlled-environment study only |
| HuggingFace Transformers integration | Excessive overhead for NanoGPT-scale; obscures QK^T extraction |
| Distributed / multi-GPU training | Single GPU sufficient at this scale; adds non-determinism |
| Web UI / dashboard | Research framework; static HTML reports are sufficient |
| Weights & Biases / MLflow | External dependency; result.json schema is comprehensive |
| Automatic hyperparameter optimization | Sweep grid is the experiment, not a means to optimize |
| GPU cluster orchestration (SLURM, K8s) | Single-machine workload within $100 budget |
| Streaming / real-time inference | Offline research framework |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MGMT-01 | Phase 1 | Complete |
| MGMT-05 | Phase 1 | Complete |
| TRNG-02 | Phase 1 | Complete |
| TRNG-07 | Phase 1 | Complete |
| GRPH-01 | Phase 2 | Complete |
| GRPH-02 | Phase 2 | Complete |
| GRPH-03 | Phase 2 | Complete |
| GRPH-04 | Phase 2 | Complete |
| GRPH-05 | Phase 2 | Complete |
| WALK-01 | Phase 3 | Complete |
| WALK-02 | Phase 3 | Complete |
| WALK-03 | Phase 3 | Complete |
| WALK-04 | Phase 3 | Complete |
| WALK-05 | Phase 3 | Complete |
| MODL-01 | Phase 4 | Complete |
| MODL-02 | Phase 4 | Complete |
| MODL-03 | Phase 4 | Complete |
| TRNG-01 | Phase 5 | Complete |
| TRNG-03 | Phase 5 | Complete |
| TRNG-04 | Phase 5 | Complete |
| TRNG-05 | Phase 5 | Complete |
| TRNG-06 | Phase 5 | Complete |
| EVAL-01 | Phase 6 | Complete |
| EVAL-02 | Phase 6 | Complete |
| EVAL-03 | Phase 6 | Complete |
| EVAL-04 | Phase 6 | Complete |
| EVAL-05 | Phase 6 | Complete |
| SVD-01 | Phase 6 | Complete |
| SVD-02 | Phase 6 | Complete |
| SVD-03 | Phase 6 | Complete |
| SVD-04 | Phase 6 | Complete |
| SVD-05 | Phase 6 | Complete |
| SVD-06 | Phase 6 | Complete |
| SVD-07 | Phase 6 | Complete |
| PRED-01 | Phase 7 | Complete |
| PRED-02 | Phase 7 | Complete |
| PRED-03 | Phase 7 | Complete |
| PRED-04 | Phase 7 | Complete |
| PRED-05 | Phase 7 | Complete |
| STAT-01 | Phase 7 | Complete |
| STAT-02 | Phase 7 | Complete |
| STAT-03 | Phase 7 | Complete |
| STAT-04 | Phase 7 | Complete |
| STAT-05 | Phase 7 | Complete |
| PLOT-01 | Phase 8 | Complete |
| PLOT-02 | Phase 8 | Complete |
| PLOT-03 | Phase 8 | Complete |
| PLOT-04 | Phase 8 | Complete |
| PLOT-05 | Phase 8 | Complete |
| PLOT-06 | Phase 8 | Complete |
| PLOT-07 | Phase 8 | Complete |
| PLOT-08 | Phase 8 | Complete |
| REPT-01 | Phase 9 | Complete |
| REPT-02 | Phase 9 | Pending |
| REPT-03 | Phase 9 | Complete |
| MATH-01 | Phase 9 | Pending |
| MATH-02 | Phase 9 | Pending |
| MGMT-02 | Phase 10 | Pending |
| MGMT-03 | Phase 10 | Pending |
| MGMT-04 | Phase 10 | Pending |
| MGMT-06 | Phase 10 | Pending |

**Coverage:**
- v1 requirements: 61 total
- Mapped to phases: 61
- Unmapped: 0

---
*Requirements defined: 2026-02-24*
*Last updated: 2026-02-24 after roadmap creation (10-phase comprehensive structure)*
