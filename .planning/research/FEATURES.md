# Feature Landscape

**Domain:** Synthetic-data transformer research framework for SVD-based hallucination prediction
**Researched:** 2026-02-24
**Mode:** Ecosystem (What features does this type of research framework need?)

---

## Table Stakes

Features users (researchers, reviewers, co-authors) expect. Missing = results are not publishable.

### Graph Generation and Validation

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| DCSBM graph generator with configurable blocks, p_in, p_out, degree correction | Core synthetic data source; without it there is no experiment | Medium | Use networkx for graph primitives, custom DCSBM sampling on top. Do NOT use networkx's built-in SBM generators -- they lack degree correction and the block jumper logic. |
| Block jumper vertex designation with configurable r and target block | Defines the "hallucination" event; this IS the experiment | Medium | Must validate that valid paths of length r exist from jumper to target block but are not the only paths (non-triviality check) |
| Graph validation: connectivity, non-triviality, edge density checks | Reviewers will ask "how do you know your graph is well-formed?" | Low | Assert connected components, verify p_in/p_out ratios produce expected density, check path existence |
| Walk generation on directed graph with configurable walk length | Training corpus; must be random walks respecting directed edges | Low | Simple random walk sampler; ensure walk length >= 2w as specified |
| Corpus size validation (>= 100x n) | Spec requirement; insufficient corpus means undertrained model | Low | Hard assert at generation time |

### Training Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| NanoGPT-scale transformer with configurable d_model, n_layers, 1 head | The model under study; architecture must match spec exactly | Medium | Single attention head is non-negotiable. Use standard PyTorch nn.Module, not HuggingFace -- too much overhead for this scale |
| Cross-entropy next-token prediction training loop | Standard training objective | Low | Vanilla training loop with AdamW, learning rate scheduling |
| Training sufficiency gate (edge >95%, rule >80%) | Hard gate before SVD analysis; without this, SVD results are noise from undertrained models | Medium | Must evaluate on held-out walks periodically during training, not just at end. Gate failure means the config is excluded, not that training continues forever |
| Checkpointing (model weights, optimizer state, training step) | Reproducibility and crash recovery on RunPod (spot instances can be preempted) | Low | torch.save/torch.load; checkpoint every N steps + on gate pass |
| Training loss and compliance curves logged per step | Reviewers expect convergence evidence for every reported config | Low | Store in result.json curves block per the schema |
| Seed control (torch, numpy, python random, CUDA deterministic) | Reproducibility requirement; 3 seeds per config means seeds must be controlable | Low | Set all seeds from one master seed; use torch.use_deterministic_algorithms(True) |
| Code hash tracking (git SHA stored with results) | Reproducibility; reviewer must be able to checkout exact code that produced results | Low | Already specified in schema; subprocess git rev-parse |

### Behavioral Evaluation

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| 4-class outcome classification per step (edge valid/invalid x rule followed/violated/N-A) | Core evaluation taxonomy; defines what counts as hallucination | Medium | Must be computed at generation time, not post-hoc. Each step gets exactly one label |
| Edge validity check (does chosen token correspond to valid directed edge?) | Fundamental correctness check | Low | Lookup in adjacency matrix |
| Rule compliance check (at step r from block jumper, is walk in target block?) | Defines the hallucination event | Medium | Must track block jumper encounters and count steps precisely |
| failure_index annotation on sequences | Alignment anchor for ALL event-aligned analysis | Low | First rule violation in each generated walk |
| Held-out walk generation for evaluation (separate from training corpus) | Cannot evaluate on training data | Low | Generate separate evaluation walks with different seed |

### SVD / Spectral Analysis

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Full QK^T matrix extraction at each token step | The measurement instrument; without this there is no experiment | Medium | Hook into attention layer forward pass; extract Q, K, compute QK^T |
| torch.linalg.svd with full_matrices=False | Performance-critical; O(d^3) per step demands efficient implementation | Low | Standard torch call, but must be batched properly |
| All ~20 SVD metrics computed per step | Spec lists specific metrics; reviewers will ask why any were omitted | High | This is the most complex feature. Each metric is individually simple but there are ~20, they must all be correct, and they must be computed efficiently in a single pass over the SVD output |
| Token-level time series storage in result.json | Required by schema; feeds all downstream analysis | Medium | Each metric becomes a float array keyed by name in token_metrics |
| Metric correctness validation (unit tests with known matrices) | Spectral metrics are easy to get subtly wrong; one bug invalidates all results | Medium | Must have test cases with analytically known SVD decompositions |

### Predictive Horizon Analysis

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| AUROC at each lookback distance j for each SVD metric | THE core result metric; this is what the paper reports | Medium | For each confirmed rule violation, look back j steps, compute AUROC of metric values at (t-j) for violation vs non-violation events |
| Predictive horizon calculation (furthest j where AUROC > 0.75) | Headline number per metric per configuration | Low | Simple threshold over AUROC curve |
| Sweep j from 1 to r | Must cover full range to find signal decay curve | Low | Loop over j values |
| Per-metric AUROC curves stored in result.json | Must be reproducible and plottable from stored data | Low | Store as curves in metrics block |

### Experiment Management

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Configuration specification (all governing parameters) | Defines what experiment was run; must be complete for reproducibility | Low | Dataclass or dict with all parameters; stored in result.json config block |
| Result.json per configuration (per the schema) | Contract that makes all downstream analysis work | Low | Already fully specified in combined-spec.md |
| Parameter sweep definition (which params to vary, ranges) | Core experimental design; must match spec sweep ranges | Medium | Define sweep space declaratively, enumerate combinations |
| Job queue with priority ordering | Budget constraint ($100) means most important configs run first; budget can be cut at any point | Medium | Priority: anchor config first, then r-vs-w sweep, then secondary sweeps |
| 3 random seeds per configuration | Statistical validity; single seed results are not publishable | Low | Outer loop over seeds |

### Visualization

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Event-aligned metric plots (position 0 = failure event) | Primary visualization of whether SVD metrics predict failures | Medium | Already specified in plotting guide; align on failure_index, show pre/post with confidence bands |
| Training convergence curves | Reviewer evidence that model is sufficiently trained | Low | Loss over steps, compliance over steps |
| AUROC vs lookback distance j curves | Core result visualization | Low | One curve per metric, x=j, y=AUROC |
| Confusion matrix for 4-class behavioral outcomes | Shows model behavior distribution | Low | Already in plotting guide |
| Pre/post failure distribution comparison | Shows whether metric distributions actually separate | Low | Already in plotting guide |

### Reporting

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Single-experiment HTML report | Self-contained documentation of each run | Medium | Already specified; base64-embedded figures |
| Comparison report across configurations | Primary deliverable; must support filtering by parameter subsets | High | Most complex reporting feature; must overlay metrics across filtered configs |
| Reproduction command in every report | Reproducibility; reviewer must be able to re-run | Low | Auto-generated from config + code hash |

---

## Differentiators

Features that strengthen the paper but are not strictly required for publishability. These elevate the work from "technically correct" to "impressive and thorough."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Non-triviality verification for block jumper paths | Proves the experiment is actually testing learning, not just graph structure memorization. Strengthens the "controlled environment" claim significantly | Medium | For each block jumper, verify that valid paths of length r to target block exist but alternative (non-compliant) paths also exist at length r. Compute path entropy as a non-triviality score |
| Training sufficiency gate with early stopping and budget accounting | Prevents wasting compute on configs that will never converge, and tracks $ spent vs remaining | Medium | Monitor compliance curves; if rate of improvement drops below threshold, stop early and mark as gate-failed. Track GPU-hours per config against budget |
| SVD metric correlation matrix across metrics | Shows which of the ~20 metrics are redundant vs independent; reviewers will appreciate dimensionality reduction of the metric space | Low | Pearson/Spearman correlation across all metric time series; identify clusters of redundant metrics |
| Metric importance ranking (which SVD metrics are most predictive) | Tells the story of which spectral properties matter most; natural "main result" table | Low | Rank metrics by max AUROC across j values or by predictive horizon length |
| Effect size reporting (Cohen's d, not just p-values) | Modern statistical practice demands effect sizes, not just significance. Reviewers increasingly reject p-value-only analysis | Low | Cohen's d for pre-failure vs post-failure metric distributions |
| Bootstrap confidence intervals on AUROC and predictive horizon | Uncertainty quantification on the core results; strengthens claims substantially | Medium | Bootstrap over sequences (not steps) to respect dependence structure |
| Wall-time profiling per pipeline stage | Proves $100 budget is realistic; helps others reproduce on similar hardware | Low | Time each stage: graph gen, walk gen, training, SVD collection, analysis |
| Automated parameter sensitivity analysis | Shows which parameters matter most for the core result (r/w ratio vs graph density vs model size) | Medium | Variance decomposition or partial correlation of config parameters with predictive horizon |
| LaTeX math verification PDF | Specified in project context; peer-reviewable math documentation | Medium | Extract all math-heavy modules, generate LaTeX, compile PDF |
| Heatmap of predictive horizon across (r, w) grid | The "money figure" for the paper; instantly communicates the r/w interaction | Low | 2D heatmap where cell color = predictive horizon, axes = r and w values |
| Grassmannian distance visualization between consecutive subspaces | Makes the abstract "subspace drift" concept visually intuitive | Medium | Plot subspace trajectory on a low-dimensional embedding of the Grassmannian |
| Config diff in comparison reports | Immediately shows what changed between runs; essential for iteration velocity | Low | Already specified in combined-spec; table of parameters that differ |

---

## Anti-Features

Features to explicitly NOT build. Building these would waste time, add complexity, or hurt the project.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Multi-head attention support | Single head is the entire point -- it makes QK^T analysis unambiguous. Adding multi-head support invites confusion, doubles testing surface, and solves no problem for this paper | Hard-code single head. Do not parameterize number of heads beyond asserting it equals 1 |
| Real-world / clinical data ingestion | This is a synthetic controlled-environment study. Adding real data pipelines conflates two very different research questions and adds months of work (IRB, data cleaning, label noise) | Keep the data pipeline strictly: DCSBM -> walks -> tokens. No data loading from external sources |
| HuggingFace Transformers integration | Massive dependency for a NanoGPT-scale model. Brings tokenizer complexity, config inheritance, model hub overhead -- none of which is needed when vocabulary IS graph vertices | Write a clean 200-line transformer in raw PyTorch. It will be faster, more debuggable, and the attention hook will be trivial |
| Distributed training / multi-GPU | RTX 3090/4090 single GPU is sufficient for NanoGPT-scale. Distributed training adds NCCL complexity, debugging difficulty, and non-determinism | Single GPU, single process. Keep it simple |
| Web UI / dashboard | Research framework, not a product. Jupyter notebooks and HTML reports are sufficient. A dashboard adds frontend dependencies, maintenance burden, and zero scientific value | Generate static HTML reports. Use Jupyter for interactive exploration |
| Weights & Biases / MLflow integration | Adds external service dependency, API key management, and network requirements (RunPod may have spotty connectivity). The result.json schema already covers logging needs comprehensively | Use the project's own result.json schema. It is more than sufficient. If W&B is desired later, it is trivial to add a write-through layer |
| Automatic hyperparameter optimization (Optuna, Ray Tune) | The sweep space is fully specified in the research design. Auto-tuning would optimize for the wrong objective (loss) rather than the right one (predictive horizon). The sweep IS the experiment, not a means to an end | Implement the specified sweep grid. Priority ordering is more important than search strategy |
| GPU cluster orchestration | Runs on single RunPod instances within $100. Cluster orchestration (SLURM, Kubernetes) is massive overhead for a single-machine workload | Simple sequential job queue with priority ordering. Save/resume between RunPod sessions |
| Streaming / real-time inference | Research framework runs offline experiments, generates results, and produces reports. There is no inference serving use case | Batch generation, batch evaluation, batch SVD collection |
| Abstract base classes and plugin architecture | This is a single-purpose research framework with a known, fixed pipeline. Extensibility through abstraction adds cognitive overhead without payoff | Concrete implementations. Functions, not frameworks. If something needs to change, change it directly |

---

## Feature Dependencies

```
DCSBM Graph Generator
  -> Block Jumper Designation
    -> Non-Triviality Verification (differentiator)
  -> Walk Generator
    -> Corpus Size Validation
    -> Training Corpus (training set)
    -> Evaluation Walks (held-out set)

NanoGPT Transformer (single head)
  -> Training Loop (cross-entropy)
    -> Seed Control
    -> Checkpointing
    -> Training Loss/Compliance Logging
    -> Training Sufficiency Gate
      -> [GATE PASS required for all downstream]
      -> Behavioral Evaluation (4-class)
        -> failure_index Annotation
      -> QK^T Extraction
        -> SVD Computation (torch.linalg.svd)
          -> All ~20 SVD Metrics
            -> Token-Level Time Series Storage
              -> Predictive Horizon Analysis (AUROC @ j)
                -> Metric Importance Ranking (differentiator)
                -> Bootstrap CIs (differentiator)
              -> SVD Metric Correlation Matrix (differentiator)

Configuration Specification
  -> Parameter Sweep Definition
    -> Job Queue with Priority Ordering
      -> 3 Seeds Per Config
        -> Result.json Per Config
          -> Single-Experiment Report
          -> Comparison Report (across configs)
            -> Config Diff
            -> Parameter Sensitivity Analysis (differentiator)

Result.json (stored)
  -> All Visualization (event-aligned plots, AUROC curves, confusion matrices, etc.)
  -> All Reports (single and comparison)
  -> Reproduction Command
```

Key dependency insight: The training sufficiency gate is a hard dependency for all SVD analysis. The pipeline must be structured so that gate-failed configs produce a result.json that records the failure but skips SVD entirely. This prevents wasted compute and noisy results.

---

## MVP Recommendation

### Phase 1: Core Pipeline (must work end-to-end on anchor config)

Prioritize in this order:

1. **DCSBM graph generator** with block jumper designation -- the data source
2. **Walk generator** with validation -- the training corpus
3. **NanoGPT transformer** (single head, d_model=128, n_layers=4) -- the model
4. **Training loop** with loss logging, seed control, checkpointing -- train the model
5. **Training sufficiency gate** -- the go/no-go decision
6. **4-class behavioral evaluation** with failure_index -- define hallucination events
7. **QK^T extraction and SVD metrics** (start with top 5 most interpretable metrics, not all 20) -- the measurement
8. **Predictive horizon AUROC** at each j -- the core result
9. **Result.json writer** -- store everything
10. **Basic event-aligned plot** -- visualize the result

This gives a complete end-to-end pipeline for one configuration. If this works on the anchor config (n=500, w=64, t=200k, d_model=128, n_layers=4), the experiment is viable.

### Phase 2: Full Metric Suite and Sweep

11. **All ~20 SVD metrics** -- complete the measurement battery
12. **Parameter sweep infrastructure** with job queue and priority ordering
13. **Full visualization suite** (all plot types from plotting guide)
14. **Single-experiment and comparison HTML reports**

### Phase 3: Statistical Rigor and Polish

15. **Bootstrap confidence intervals** on AUROC and predictive horizon
16. **Effect size reporting** (Cohen's d)
17. **Metric correlation and importance analysis**
18. **Parameter sensitivity analysis**
19. **Non-triviality verification** for block jumper paths
20. **Math verification PDF**

### Defer

- **Grassmannian trajectory visualization**: Conceptually interesting but not required for the paper's claims. Build only if time permits after core results are in.
- **Automated budget tracking**: Nice for operations but does not affect scientific results. Manual tracking in a spreadsheet is fine.

---

## Complexity Budget

Total estimated complexity for table stakes: **HIGH** (individually most are Low-Medium, but there are many features and the SVD metric suite is substantial).

The critical complexity concentration is in two areas:

1. **SVD metrics implementation (~20 metrics, each must be correct)**: This is where bugs hide. Each metric is simple in isolation but getting all 20 right, tested, and efficient in aggregate is the hardest single feature. Budget 30% of implementation effort here.

2. **Comparison reporting with parameter filtering**: The comparison report must handle arbitrary subsets of the sweep space. This is a combinatorial UI problem that can easily balloon. Keep it simple: filter by exact parameter values, no complex query language.

---

## Sources

- Project specification (combined-spec.md) -- PRIMARY source for all feature requirements
- PROJECT.md -- Requirements and constraints
- Training data knowledge of: NanoGPT (Karpathy), mechanistic interpretability research (Elhage et al., Olsson et al.), spectral methods in machine learning, stochastic block models (Holland et al., Karrer & Newman 2011), experiment management patterns
- Confidence: HIGH for feature categorization (derived directly from project spec), MEDIUM for complexity estimates (based on training data experience with similar systems, not verified against this specific codebase)
