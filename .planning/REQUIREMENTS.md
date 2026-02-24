# Requirements: DCSBM Transformer SVD Hallucination Prediction

**Defined:** 2026-02-24
**Core Value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Graph Generation

- [x] **GRPH-01**: System generates DCSBM directed graphs with configurable number of vertices (n), blocks (K), in-group probability (p_in), out-group probability (p_out), and degree correction parameters per Karrer & Newman 2011
- [ ] **GRPH-02**: System designates block jumper vertices with configurable jump length r and target block, where the rule "after r steps from jumper v_i in block b, the walk must land in a specific target block different from b" is enforced
- [x] **GRPH-03**: System validates graph connectivity (strongly connected), minimum expected degree >= 3, and edge density matching expected p_in/p_out ratios
- [ ] **GRPH-04**: System verifies non-triviality of block jumper rules: valid paths of length r from jumper to target block exist but are not the only paths at that length
- [ ] **GRPH-05**: System caches generated graphs by config hash to avoid redundant regeneration across sweep configs sharing the same graph parameters

### Walk Generation

- [ ] **WALK-01**: System generates directed random walks on the DCSBM graph with configurable walk length l (swept at 2w, 4w, 8w)
- [ ] **WALK-02**: System validates corpus size is at least 2 orders of magnitude larger than n (t >= 100n)
- [ ] **WALK-03**: System produces separate train and evaluation walk sets with different seeds
- [ ] **WALK-04**: System tracks block jumper encounter metadata during walk generation (which jumper was hit, at which step, expected target block at step+r)
- [ ] **WALK-05**: System caches generated walks by config hash to avoid redundant regeneration

### Model

- [ ] **MODL-01**: System implements a NanoGPT-scale transformer with configurable d_model (64, 128, 256), n_layers (2, 4, 6), and exactly 1 attention head
- [ ] **MODL-02**: Model's single attention head returns QK^T matrix alongside logits via a `return_qkt=True` flag for SVD analysis
- [ ] **MODL-03**: Model vocabulary equals the number of graph vertices (tokens are vertex IDs)

### Training

- [ ] **TRNG-01**: System trains the transformer using cross-entropy next-token prediction with AdamW optimizer and cosine learning rate schedule
- [x] **TRNG-02**: System controls all random seeds (torch, numpy, python random, CUDA deterministic) from a single master seed for reproducibility
- [ ] **TRNG-03**: System checkpoints model weights, optimizer state, and training step periodically and on gate pass
- [ ] **TRNG-04**: System logs training loss and compliance curves per step, stored in result.json curves block
- [ ] **TRNG-05**: System enforces training sufficiency gate: edge compliance >95% and rule compliance >80% on held-out walks, evaluated periodically during training
- [ ] **TRNG-06**: Configurations that fail the sufficiency gate after allocated training budget are flagged and excluded from SVD analysis, with failure metadata recorded in result.json
- [x] **TRNG-07**: System tracks git code hash (short SHA) and stores it with results for reproducibility

### Behavioral Evaluation

- [ ] **EVAL-01**: System classifies each generation step into 4-class outcome: edge valid/invalid crossed with rule followed/violated/not-applicable
- [ ] **EVAL-02**: System checks edge validity (chosen next token corresponds to a valid directed edge in DCSBM) at each step
- [ ] **EVAL-03**: System checks rule compliance (at step r from a block jumper vertex, the walk lands in the required target block) at each step
- [ ] **EVAL-04**: System annotates each generated sequence with failure_index (index of first rule violation), or null for correct sequences
- [ ] **EVAL-05**: Behavioral evaluation and SVD collection happen in a single fused forward pass (not separate inference runs)

### SVD Metrics

- [ ] **SVD-01**: System extracts the full QK^T matrix from the single attention head at every token step during evaluation
- [ ] **SVD-02**: System computes SVD using torch.linalg.svd with full_matrices=False, batched for efficiency on GPU
- [ ] **SVD-03**: System computes all ~20 specified SVD metrics per token step: principal vector direction change, dominant subspace membership change, principal angles (Grassmannian distance), condition number, spectral gap, generalized gap (k=2,4,8), singular value entropy, stable rank, participation ratio, low-rank approximation error (k=2,4,8), angular velocity, subspace drift, singular value velocity, condition number velocity, left SV alignment with current token embedding, right SV alignment with predicted token embedding, dominant subspace coherence with embedding matrix, effective rank within context window, singular value variance
- [ ] **SVD-04**: System stores all SVD metrics as token-level time series in result.json/token_metrics.npz keyed by metric name
- [ ] **SVD-05**: System includes numerical guards: NaN/Inf clamping, epsilon in entropy computation, condition number capped at 1e6, Grassmannian distance for subspace tracking instead of single vector direction
- [ ] **SVD-06**: System collects SVD metrics only for positions >= w (context window warmup) to avoid padding artifacts
- [ ] **SVD-07**: Each SVD metric has unit tests against analytically known matrix decompositions

### Predictive Horizon Analysis

- [ ] **PRED-01**: System computes AUROC at each lookback distance j (from 1 to r) for each SVD metric, comparing metric values at step (t-j) for violation vs non-violation events
- [ ] **PRED-02**: System calculates predictive horizon as the furthest j at which AUROC exceeds 0.75 for each metric
- [ ] **PRED-03**: System uses position-matched baselines (control events sampled at same absolute position in non-jumper walks) to control for positional confounds
- [ ] **PRED-04**: System runs shuffle controls (permuted labels) to verify signal is not positional artifact (AUROC > 0.6 on shuffled = flag)
- [ ] **PRED-05**: System stores per-metric AUROC curves in result.json metrics block

### Experiment Management

- [x] **MGMT-01**: System defines experiment configuration as a frozen, serializable, hashable dataclass with all governing parameters
- [ ] **MGMT-02**: System implements parameter sweep with declarative definition of parameter ranges matching the spec sweep ranges
- [ ] **MGMT-03**: System implements a priority-ordered job queue: anchor config first, then core r-vs-w sweep (Tier 1), then architecture/w sweeps (Tier 2), then secondary sweeps (Tier 3)
- [ ] **MGMT-04**: System runs 3 random seeds per configuration
- [x] **MGMT-05**: System writes result.json per configuration conforming to the project schema (schema_version, experiment_id, timestamp, description, tags, config, metrics, sequences, metadata)
- [ ] **MGMT-06**: System persists sweep state for resume after RunPod preemption

### Visualization

- [ ] **PLOT-01**: System generates event-aligned SVD metric plots (position 0 = failure event, negative = before, positive = after) with confidence bands and correct-sequence baseline overlay
- [ ] **PLOT-02**: System generates training convergence curves (loss and compliance over steps)
- [ ] **PLOT-03**: System generates AUROC vs lookback distance j curves per SVD metric
- [ ] **PLOT-04**: System generates confusion matrix for 4-class behavioral outcomes
- [ ] **PLOT-05**: System generates pre/post failure distribution comparison plots
- [ ] **PLOT-06**: System generates predictive horizon heatmap across (r, w) parameter grid
- [ ] **PLOT-07**: All plots follow the project style baseline (seaborn whitegrid, consistent palette, no default matplotlib style)
- [ ] **PLOT-08**: All figures saved as both PNG (300 dpi) and SVG to results/{experiment_id}/figures/

### Reporting

- [ ] **REPT-01**: System generates self-contained single-experiment HTML report with base64-embedded figures, covering: header, configuration, scalar metrics, curves, confusion matrix, statistical tests, sequence analysis, and reproduction command
- [ ] **REPT-02**: System generates comparison HTML report across multiple experiments with: scalar metrics comparison table, curve overlays, config diff table, and aligned sequence plot overlays
- [ ] **REPT-03**: Every report includes a reproduction block with git checkout command and full CLI arguments

### Statistical Rigor

- [ ] **STAT-01**: System applies Holm-Bonferroni correction for multiple comparisons across pre-registered primary metrics (3-5 metrics selected before sweep)
- [ ] **STAT-02**: System computes bootstrap confidence intervals on AUROC and predictive horizon estimates
- [ ] **STAT-03**: System reports effect sizes (Cohen's d) for pre-failure vs post-failure metric distributions
- [ ] **STAT-04**: System computes SVD metric correlation matrix to identify redundant metrics
- [ ] **STAT-05**: System produces metric importance ranking by max AUROC across j values

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
| GRPH-02 | Phase 2 | Pending |
| GRPH-03 | Phase 2 | Complete |
| GRPH-04 | Phase 2 | Pending |
| GRPH-05 | Phase 2 | Pending |
| WALK-01 | Phase 3 | Pending |
| WALK-02 | Phase 3 | Pending |
| WALK-03 | Phase 3 | Pending |
| WALK-04 | Phase 3 | Pending |
| WALK-05 | Phase 3 | Pending |
| MODL-01 | Phase 4 | Pending |
| MODL-02 | Phase 4 | Pending |
| MODL-03 | Phase 4 | Pending |
| TRNG-01 | Phase 5 | Pending |
| TRNG-03 | Phase 5 | Pending |
| TRNG-04 | Phase 5 | Pending |
| TRNG-05 | Phase 5 | Pending |
| TRNG-06 | Phase 5 | Pending |
| EVAL-01 | Phase 6 | Pending |
| EVAL-02 | Phase 6 | Pending |
| EVAL-03 | Phase 6 | Pending |
| EVAL-04 | Phase 6 | Pending |
| EVAL-05 | Phase 6 | Pending |
| SVD-01 | Phase 6 | Pending |
| SVD-02 | Phase 6 | Pending |
| SVD-03 | Phase 6 | Pending |
| SVD-04 | Phase 6 | Pending |
| SVD-05 | Phase 6 | Pending |
| SVD-06 | Phase 6 | Pending |
| SVD-07 | Phase 6 | Pending |
| PRED-01 | Phase 7 | Pending |
| PRED-02 | Phase 7 | Pending |
| PRED-03 | Phase 7 | Pending |
| PRED-04 | Phase 7 | Pending |
| PRED-05 | Phase 7 | Pending |
| STAT-01 | Phase 7 | Pending |
| STAT-02 | Phase 7 | Pending |
| STAT-03 | Phase 7 | Pending |
| STAT-04 | Phase 7 | Pending |
| STAT-05 | Phase 7 | Pending |
| PLOT-01 | Phase 8 | Pending |
| PLOT-02 | Phase 8 | Pending |
| PLOT-03 | Phase 8 | Pending |
| PLOT-04 | Phase 8 | Pending |
| PLOT-05 | Phase 8 | Pending |
| PLOT-06 | Phase 8 | Pending |
| PLOT-07 | Phase 8 | Pending |
| PLOT-08 | Phase 8 | Pending |
| REPT-01 | Phase 9 | Pending |
| REPT-02 | Phase 9 | Pending |
| REPT-03 | Phase 9 | Pending |
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
