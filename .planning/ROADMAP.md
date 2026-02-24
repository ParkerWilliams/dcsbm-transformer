# Roadmap: DCSBM Transformer SVD Hallucination Prediction

## Overview

This roadmap delivers a research framework that generates synthetic token sequences from DCSBM graphs with known ground-truth rules, trains a NanoGPT-scale transformer on them, and analyzes whether SVD instability in the QK^T attention matrix predicts rule violations before they occur. The pipeline progresses from foundational infrastructure (config, schema, reproducibility) through data generation (graphs, walks), model training, measurement (behavioral evaluation and SVD collection), analysis (predictive horizon, statistical rigor), visualization, reporting, and finally sweep execution across the full parameter grid. Each phase delivers a verifiable capability; the anchor configuration runs end-to-end through Phases 1-9 before the sweep (Phase 10) consumes GPU budget.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Config, Schema, and Reproducibility Foundation** - Experiment configuration, result.json schema, seed management, and code hash tracking
- [ ] **Phase 2: DCSBM Graph Generation** - Custom degree-corrected stochastic block model with block jumper rules, validation gates, and caching
- [ ] **Phase 3: Walk Generation** - Directed random walks on DCSBM graphs with corpus validation, jumper metadata, and caching
- [ ] **Phase 4: Transformer Model** - NanoGPT-scale single-head transformer with QK^T extraction capability
- [ ] **Phase 5: Training Pipeline** - Cross-entropy training loop with sufficiency gate, checkpointing, and training curve logging
- [ ] **Phase 6: Behavioral Evaluation and SVD Collection** - Fused forward pass producing 4-class behavioral labels and all SVD metrics with numerical guards
- [ ] **Phase 7: Predictive Horizon and Statistical Analysis** - AUROC at each lookback distance, position-matched baselines, multiple comparison correction, and effect sizes
- [ ] **Phase 8: Visualization** - All publication-quality plot types from event-aligned metrics to predictive horizon heatmaps
- [ ] **Phase 9: Reporting and Math Verification** - Single-experiment and comparison HTML reports, reproduction blocks, and math verification PDF
- [ ] **Phase 10: Sweep Infrastructure and Execution** - Priority-ordered job queue, multi-seed execution, budget tracking, preemption recovery, and full parameter sweep

## Phase Details

### Phase 1: Config, Schema, and Reproducibility Foundation
**Goal**: Every experiment is fully specified by a frozen, hashable configuration object; results conform to a validated schema; and all sources of randomness are controlled from a single master seed
**Depends on**: Nothing (first phase)
**Requirements**: MGMT-01, MGMT-05, TRNG-02, TRNG-07
**Success Criteria** (what must be TRUE):
  1. An ExperimentConfig can be instantiated with the anchor parameters (n=500, w=64, t=200k, d_model=128, n_layers=4, 1 head), serialized to JSON, deserialized back, and the hash matches
  2. A result.json file can be created conforming to the project schema (schema_version, experiment_id, timestamp, description, tags, config, metrics, sequences, metadata) and validated against the schema
  3. Setting the master seed produces identical random number sequences across torch, numpy, and python random on repeated runs
  4. The current git short SHA is captured and stored in a result.json metadata block
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — ExperimentConfig dataclass system, result.json schema validation, and project scaffolding
- [ ] 01-02-PLAN.md — Seed management, git hash tracking, and reproducibility integration tests

### Phase 2: DCSBM Graph Generation
**Goal**: The system generates valid, non-trivial DCSBM graphs with block jumper rules that are ready to serve as training data foundations
**Depends on**: Phase 1
**Requirements**: GRPH-01, GRPH-02, GRPH-03, GRPH-04, GRPH-05
**Success Criteria** (what must be TRUE):
  1. A DCSBM graph with the anchor config parameters (n=500, K blocks, configurable p_in/p_out) is generated and is strongly connected with minimum expected degree >= 3
  2. Block jumper vertices are designated with jump length r, and for each jumper vertex there exist valid paths of length r to the target block that are not the only paths at that length (non-triviality verified)
  3. Edge density of the generated graph matches expected p_in/p_out ratios within statistical tolerance
  4. Regenerating with the same config hash loads from cache instead of recomputing
  5. Degree correction produces heterogeneous degree distributions following the configured parameters
**Plans**: TBD

Plans:
- [ ] 02-01: Custom DCSBM generator with degree correction
- [ ] 02-02: Block jumper designation and graph validation gates
- [ ] 02-03: Graph caching by config hash

### Phase 3: Walk Generation
**Goal**: The system produces correctly structured walk corpora with complete jumper-event metadata, ready to serve as transformer training and evaluation data
**Depends on**: Phase 2
**Requirements**: WALK-01, WALK-02, WALK-03, WALK-04, WALK-05
**Success Criteria** (what must be TRUE):
  1. Directed random walks of configurable length (2w, 4w, 8w) are generated on the DCSBM graph, and each walk follows only valid directed edges
  2. The corpus size is validated as at least 100x n, and generation fails with a clear error if this threshold is not met
  3. Train and evaluation walk sets use different seeds and contain no overlapping walks
  4. Every block jumper encounter during walk generation is recorded with the jumper vertex ID, encounter step, and expected target block at step+r
  5. Regenerating walks with the same config hash loads from cache instead of recomputing
**Plans**: TBD

Plans:
- [ ] 03-01: Walk generator with corpus validation
- [ ] 03-02: Jumper event metadata tracking and walk caching

### Phase 4: Transformer Model
**Goal**: A minimal, fully transparent NanoGPT-scale transformer exists that can process token sequences and expose its internal QK^T attention matrix for analysis
**Depends on**: Phase 1
**Requirements**: MODL-01, MODL-02, MODL-03
**Success Criteria** (what must be TRUE):
  1. The transformer accepts configurable d_model (64, 128, 256), n_layers (2, 4, 6), and enforces exactly 1 attention head
  2. With return_qkt=True, the forward pass returns both logits and the QK^T matrix, and the QK^T matrix has the expected shape (seq_len, seq_len)
  3. The vocabulary size equals the number of graph vertices (token IDs are vertex IDs), and the model handles the anchor config vocabulary (n=500) correctly
**Plans**: TBD

Plans:
- [ ] 04-01: NanoGPT transformer with single-head attention and QK^T extraction

### Phase 5: Training Pipeline
**Goal**: The transformer can be trained to learn edge structure and block jumper rules from walk data, with a hard sufficiency gate that must pass before any downstream SVD analysis
**Depends on**: Phase 3, Phase 4
**Requirements**: TRNG-01, TRNG-03, TRNG-04, TRNG-05, TRNG-06
**Success Criteria** (what must be TRUE):
  1. The model trains with cross-entropy next-token prediction using AdamW optimizer and cosine learning rate schedule, and training loss decreases monotonically (on average)
  2. Model weights, optimizer state, and training step are checkpointed periodically and can be resumed from checkpoint
  3. Training loss and compliance curves (edge compliance, rule compliance) are logged per evaluation interval and stored in result.json curves block
  4. The anchor configuration passes the sufficiency gate (edge compliance >95%, rule compliance >80%) on held-out evaluation walks
  5. A configuration that fails the sufficiency gate is flagged with failure metadata in result.json and excluded from SVD analysis
**Plans**: TBD

Plans:
- [ ] 05-01: Training loop with AdamW and cosine schedule
- [ ] 05-02: Sufficiency gate and checkpoint management

### Phase 6: Behavioral Evaluation and SVD Collection
**Goal**: A single evaluation pass through generated sequences produces both behavioral labels (4-class outcomes) and all SVD metrics from the QK^T matrix, with numerical stability guarantees
**Depends on**: Phase 5
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, SVD-01, SVD-02, SVD-03, SVD-04, SVD-05, SVD-06, SVD-07
**Success Criteria** (what must be TRUE):
  1. Each generation step is classified into the 4-class outcome (edge valid/invalid x rule followed/violated/not-applicable) and sequences are annotated with failure_index
  2. The QK^T matrix is extracted at every token step and SVD is computed using batched torch.linalg.svd with full_matrices=False on GPU
  3. All ~20 specified SVD metrics are computed per token step and stored as token-level time series in token_metrics.npz
  4. SVD metrics are collected only for positions >= w (context window warmup), and numerical guards (NaN/Inf clamping, epsilon in entropy, condition number cap at 1e6, Grassmannian distance for subspace tracking) prevent any NaN or Inf in stored metrics
  5. Each SVD metric function has a unit test that verifies its output against an analytically known matrix decomposition
**Plans**: TBD

Plans:
- [ ] 06-01: Fused behavioral evaluation with 4-class classification
- [ ] 06-02: SVD metric computation functions with unit tests
- [ ] 06-03: Batched SVD collection pipeline with numerical guards and storage

### Phase 7: Predictive Horizon and Statistical Analysis
**Goal**: For each SVD metric, the system measures how far in advance it can predict rule violations (AUROC at each lookback distance), with position-matched baselines and rigorous statistical controls
**Depends on**: Phase 6
**Requirements**: PRED-01, PRED-02, PRED-03, PRED-04, PRED-05, STAT-01, STAT-02, STAT-03, STAT-04, STAT-05
**Success Criteria** (what must be TRUE):
  1. AUROC is computed at each lookback distance j (from 1 to r) for each SVD metric, comparing metric values at step (t-j) for violation vs non-violation events
  2. Predictive horizon (furthest j where AUROC > 0.75) is calculated for each metric and stored in result.json
  3. Position-matched baselines (control events at same absolute position in non-jumper walks) are used, and shuffle controls with permuted labels flag any metric where shuffled AUROC > 0.6
  4. Holm-Bonferroni correction is applied across pre-registered primary metrics, bootstrap confidence intervals are computed on AUROC estimates, and effect sizes (Cohen's d) are reported
  5. The SVD metric correlation matrix identifies redundant metrics, and a metric importance ranking by max AUROC produces a clear ordering
**Plans**: TBD

Plans:
- [ ] 07-01: AUROC computation with position-matched baselines and shuffle controls
- [ ] 07-02: Statistical rigor (correction, bootstrap CIs, effect sizes, correlation, ranking)

### Phase 8: Visualization
**Goal**: All analysis results can be rendered as publication-quality static figures that follow a consistent visual style
**Depends on**: Phase 7
**Requirements**: PLOT-01, PLOT-02, PLOT-03, PLOT-04, PLOT-05, PLOT-06, PLOT-07, PLOT-08
**Success Criteria** (what must be TRUE):
  1. Event-aligned SVD metric plots show position 0 = failure event with negative positions before and positive after, including confidence bands and correct-sequence baseline overlay
  2. Training convergence curves, AUROC vs lookback distance curves, confusion matrices, and pre/post failure distribution plots are all generated from result.json data
  3. The predictive horizon heatmap across the (r, w) parameter grid renders correctly with at least the anchor config data point
  4. All plots use seaborn whitegrid style with a consistent palette, and every figure is saved as both PNG (300 dpi) and SVG
**Plans**: TBD

Plans:
- [ ] 08-01: Core plot types (event-aligned, training curves, AUROC curves, confusion matrix)
- [ ] 08-02: Distribution plots, heatmaps, and style standardization

### Phase 9: Reporting and Math Verification
**Goal**: A complete experiment produces a self-contained HTML report with all figures and reproduction instructions, and the mathematical implementations are documented for peer review
**Depends on**: Phase 8
**Requirements**: REPT-01, REPT-02, REPT-03, MATH-01, MATH-02
**Success Criteria** (what must be TRUE):
  1. A single-experiment HTML report is generated with base64-embedded figures covering header, configuration, scalar metrics, curves, confusion matrix, statistical tests, sequence analysis, and reproduction command
  2. A comparison HTML report is generated across multiple experiments with scalar metrics comparison table, curve overlays, config diff table, and aligned sequence plot overlays
  3. Every report includes a reproduction block with git checkout command and full CLI arguments
  4. A math verification PDF is generated with title page (noting AI-generated LaTeX requiring researcher sign-off), table of contents, and per-source-file sections with code blocks and LaTeX math
**Plans**: TBD

Plans:
- [ ] 09-01: Single-experiment HTML report with Jinja2 templates
- [ ] 09-02: Comparison HTML report and reproduction blocks
- [ ] 09-03: Math verification PDF generation

### Phase 10: Sweep Infrastructure and Execution
**Goal**: The full parameter sweep runs in priority order across the $100 GPU budget, with automatic caching, multi-seed replication, and crash recovery
**Depends on**: Phase 9
**Requirements**: MGMT-02, MGMT-03, MGMT-04, MGMT-06
**Success Criteria** (what must be TRUE):
  1. The parameter sweep is defined declaratively matching the spec sweep ranges, and the job queue executes in priority order (Tier 1: anchor r-sweep first, Tier 2: architecture/w sweeps, Tier 3: secondary sweeps)
  2. Each configuration runs with 3 random seeds, and graph/walk caching prevents redundant regeneration across configs sharing the same graph parameters
  3. Sweep state is persisted to disk and execution resumes correctly after RunPod preemption without re-running completed configurations
  4. Budget tracking halts execution at the $10 reserve threshold
**Plans**: TBD

Plans:
- [ ] 10-01: Parameter sweep definition and priority-ordered job queue
- [ ] 10-02: Multi-seed execution, caching, and preemption recovery

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10
Note: Phase 4 depends only on Phase 1 (not 2 or 3), so Phases 2-3 and Phase 4 could theoretically be parallelized.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Config, Schema, and Reproducibility Foundation | 0/2 | Not started | - |
| 2. DCSBM Graph Generation | 0/3 | Not started | - |
| 3. Walk Generation | 0/2 | Not started | - |
| 4. Transformer Model | 0/1 | Not started | - |
| 5. Training Pipeline | 0/2 | Not started | - |
| 6. Behavioral Evaluation and SVD Collection | 0/3 | Not started | - |
| 7. Predictive Horizon and Statistical Analysis | 0/2 | Not started | - |
| 8. Visualization | 0/2 | Not started | - |
| 9. Reporting and Math Verification | 0/3 | Not started | - |
| 10. Sweep Infrastructure and Execution | 0/2 | Not started | - |
