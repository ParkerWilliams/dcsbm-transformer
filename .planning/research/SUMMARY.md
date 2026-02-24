# Project Research Summary

**Project:** DCSBM Transformer -- SVD Predictive Signals for LLM Hallucination
**Domain:** Synthetic-data research framework for spectral analysis of transformer attention
**Researched:** 2026-02-24
**Confidence:** HIGH

## Executive Summary

This project is a scientific research framework, not a product. Experts build this type of system as a tightly-controlled pipeline: a synthetic data source (DCSBM graphs) feeds a minimal model (NanoGPT-scale transformer), whose internal state (QK^T attention matrices) is measured throughout autoregressive generation, and the measurements are subjected to statistical analysis to detect predictive signals. The canonical approach is to write minimal, purpose-built code with no framework overhead -- custom DCSBM generator, custom transformer (~200 lines), custom training loop, custom job queue -- because every abstraction layer between the researcher and the QK^T matrix is a liability. The entire pipeline after walk generation lives on GPU in PyTorch tensors; moving data to CPU for any intermediate step (SVD, metrics) is the primary performance antipattern.

The recommended technical approach is Python 3.11 with PyTorch 2.5+ as the single computational framework, NetworkX as the graph primitive layer with a custom DCSBM implementation on top (NetworkX's built-in SBM generator lacks degree correction -- this is a critical gap requiring ~50-80 lines of custom sampling code), and a seven-module architecture (config, graph, walks, model, training, evaluation, analysis) communicating through file-based contracts (`.pt`, `.npz`, `.json`). Experiment orchestration uses a priority-ordered job queue implemented in ~200 lines of custom dataclass code -- no Hydra, no wandb, no Optuna. The job queue is critical because the $100 GPU budget forces priority-ordered execution: anchor config first, then the core r-sweep, then secondary parameter sweeps. Budget can be cut at any point without losing the most important results.

The dominant risks are: (1) degenerate DCSBM graphs (disconnected blocks, trivially-learnable block jumper rules) that must be caught before training begins; (2) SVD memory and compute scaling (O(w^3) per step) that becomes unaffordable for w=256 without batching and streaming; (3) statistical confounds in the predictive horizon analysis (position effects, class imbalance, multiple comparisons inflation) that could make spurious signals appear significant. All three risks have known mitigations -- graph validation gates, batched GPU SVD, and position-matched baselines -- but they must be built into the pipeline from the start, not retrofitted after the sweep is running.

---

## Key Findings

### Recommended Stack

The full pipeline stays in PyTorch on GPU. After graph generation (NetworkX on CPU), every computation from tokenized walks through training, attention extraction, SVD, and metric derivation runs on GPU using `torch.linalg.svd` (batched, on CUDA tensors). The only CPU-side operations are graph generation, disk I/O, and final JSON serialization. This eliminates GPU-to-CPU transfer latency, which would otherwise dominate the SVD collection loop.

The model is a NanoGPT-scale transformer written from scratch in raw PyTorch (~200 lines), not HuggingFace Transformers. This is non-negotiable: HuggingFace adds massive dependency weight, obscures attention internals, and makes QK^T extraction require fighting abstraction layers. At vocabulary size n (up to 2000 vertices) and d_model up to 256 with a single attention head, the model fits in under 1MB and trains in minutes on an RTX 3090. Results are stored as JSON (per the project's defined schema) plus companion `.npz` files for per-step SVD metric tensors. Visualization is matplotlib + seaborn (static figures only). HTML reports use Jinja2 templates.

**Core technologies:**
- **Python 3.11 + venv**: runtime (3.11 is the stable sweet spot; 3.12+ has intermittent C extension issues with scientific stack)
- **PyTorch >=2.5**: everything -- model, training loop, `torch.linalg.svd`, metric computation on GPU
- **NetworkX >=3.2 + custom DCSBM**: graph generation (NetworkX SBM generator lacks degree correction; ~50-80 lines of custom sampling code required per Karrer & Newman 2011)
- **NumPy >=1.26 + SciPy >=1.12**: CPU-side graph adjacency storage (sparse), statistical tests (Mann-Whitney U, Wilson intervals)
- **matplotlib >=3.8 + seaborn >=0.13**: static publication-quality figures
- **Jinja2 >=3.1**: HTML report templating
- **pylatex >=1.4 + system pdflatex**: math verification PDF generation (pylatex can be fragile; have a raw Jinja2+LaTeX template fallback ready)
- **pytest >=8.0 + ruff >=0.4**: testing and linting (SVD metric correctness depends entirely on unit tests against known analytical solutions)

See `.planning/research/STACK.md` for full `pyproject.toml`, installation commands, and complete alternatives analysis.

### Expected Features

The project spec (combined-spec.md) fully defines the feature set. The complexity concentration is in two areas: the ~20 SVD metrics (individually simple but collectively requiring ~30% of implementation effort, and easy to get subtly wrong) and comparison reporting (combinatorial config filtering that can balloon without discipline).

**Must have (table stakes):**
- Custom DCSBM generator with degree correction, configurable p_in/p_out, and block jumper designation
- Graph validation: connectivity, non-triviality check (block jumper rules must not be topologically trivial -- require 0.1-0.5 fraction of length-r paths reach target block), minimum expected degree, edge density checks
- Walk generator producing directed random walks with jumper-event metadata tracked; corpus size validated at >= 100x n
- NanoGPT transformer (single attention head, configurable d_model/n_layers) with explicit QK^T extraction via `return_qkt=True` flag
- Training loop with cross-entropy, AdamW, cosine schedule, seed control, and checkpointing
- Training sufficiency gate (edge compliance >95%, rule compliance >80%) -- hard gate before any SVD analysis
- 4-class behavioral outcome classification (edge valid/invalid x rule followed/violated/NA) with `failure_index` annotation
- All ~20 SVD metrics computed per token step during fused evaluation pass
- Predictive horizon AUROC at each lookback distance j for each metric
- Priority-ordered job queue with 3 seeds per config and result.json output per run
- Single-experiment and comparison HTML reports with embedded figures and reproduction commands

**Should have (differentiators):**
- Non-triviality verification using path fraction analysis (builds reviewer confidence in experimental validity)
- Bootstrap confidence intervals on AUROC and predictive horizon estimates
- Effect size reporting (Cohen's d, not just p-values -- required by modern reviewers)
- SVD metric correlation matrix (identifies which of the ~20 metrics are redundant)
- Metric importance ranking by max AUROC (natural "main result" table for the paper)
- Wall-time profiling per pipeline stage (proves $100 budget is reproducible on similar hardware)
- Heatmap of predictive horizon across (r, w) grid (the "money figure" for the paper)

**Defer to v2+:**
- Grassmannian trajectory visualization (conceptually interesting; defer until core results are confirmed)
- Automated budget tracking dashboard (manual tracking in a spreadsheet is sufficient for a single researcher)
- Automated parameter sensitivity analysis (can be done after core data is in hand)

See `.planning/research/FEATURES.md` for the full feature dependency graph and per-feature complexity estimates.

### Architecture Approach

The system is a seven-module linear pipeline wrapped by a sweep orchestrator. Each module has a single responsibility and communicates via well-defined file contracts -- every stage writes to disk before the next stage reads from disk. This is non-negotiable: an in-memory pipeline prevents caching, crash recovery, and stage-level profiling. The most critical architectural decision is that behavioral evaluation and SVD collection happen in a single fused forward pass (`return_qkt=True`), not as separate inference runs. The model returns logits AND QK^T simultaneously; running inference twice would double wall time for no benefit.

**Major components:**
1. **config** -- `ExperimentConfig` dataclass (frozen, serializable, hashable); cross-parameter validation; deterministic hashing for cache keying
2. **graph** -- DCSBM generation, block jumper designation, path validation, adjacency persistence (scipy sparse + JSON metadata); cached by config hash
3. **walks** -- directed random walk sampling, train/eval split, jumper-event metadata; cached by config hash (many sweep configs share the same graph)
4. **model** -- NanoGPT transformer with `SingleHeadAttention(return_qkt=True)`; ~200 lines total; single head is hard-coded, not parameterizable
5. **training** -- cross-entropy training loop, sufficiency gate, checkpointing, training curve logging
6. **evaluation** -- fused behavioral + SVD collection pass; writes `result.json` (summary scalars + curves) + `token_metrics.npz` (full per-step metric arrays referenced by sequence_id)
7. **analysis** -- predictive horizon AUROC, position-matched baselines, statistical tests, all visualization, HTML report generation
8. **sweep** -- priority-ordered job queue (3 tiers), budget tracking, resume-from-checkpoint state persistence

**Key patterns to follow:**
- Config-driven everything: no hardcoded parameters in any module
- `return_qkt=True` flag for QK^T extraction (preferred over forward hooks since model code is fully controlled)
- Batched SVD: group same-size QK^T matrices (all positions >= w have identical shape) into a batch tensor and call `torch.linalg.svd` once per group
- Hybrid storage: `result.json` holds summary scalars and curves; `token_metrics.npz` holds full per-step arrays (loading multi-GB JSON for a single plot is the primary reporting antipattern)
- Graph and walk caching by config hash: many sweep configs vary only model architecture while sharing the same graph -- do not regenerate

See `.planning/research/ARCHITECTURE.md` for full data flow, module dependency graph, build sequence, interface contracts, and scalability projections across anchor vs. large-n configs.

### Critical Pitfalls

1. **Degenerate DCSBM graphs** -- With small n, many blocks, and low p_out, blocks become disconnected and block jumper rules become unsatisfiable before training even begins. Budget is burned on configs that were dead on arrival. Prevention: validate strong connectivity, minimum expected degree >= 3, and valid path existence for every block jumper immediately after graph generation. Pre-filter the entire sweep grid on CPU before any GPU time is allocated. (PITFALLS.md, Pitfall 1)

2. **SVD memory and compute scaling** -- Collecting full per-step SVD metrics for all evaluation walks produces GB-scale data that overwhelms RunPod storage (typically 20-50 GB). The O(w^3) cost per step makes w=256 configs 64x more expensive than w=64. Prevention: collect SVD only during evaluation (not training), use streaming writes to `.npz`, store only scalar metrics (never U/S/Vh tensors), evaluate only walks containing jumper events, batch same-size QK^T matrices for GPU efficiency. Profile the anchor config before launching the sweep. (PITFALLS.md, Pitfalls 3 and 8)

3. **Position confounds masquerading as predictive signal** -- SVD metrics have inherent positional trends (condition number grows with context length); if jumper events cluster at specific walk positions, AUROC appears significant by coincidence. This would make the paper's central claim wrong. Prevention: use position-matched baselines (control events sampled at the same absolute position in non-jumper walks), run shuffle controls (AUROC > 0.6 with permuted labels means the signal is positional, not predictive), detrend SVD metrics by subtracting position-wise means from non-jumper walks. Design these baselines into the evaluation schema from the start. (PITFALLS.md, Pitfall 7)

4. **Budget exhaustion before core results** -- The sweep space has thousands of configs; $100 at $0.44/hr buys ~227 GPU-hours. Without priority ordering, the budget can be consumed before the r-vs-w interaction (the headline scientific result) is fully characterized. Prevention: anchor config always runs first (calibrates timing and verifies the pipeline), three-tier job queue (Tier 1: core r-sweep at anchor config, 24 runs; Tier 2: architecture and w sweeps, ~63 runs; Tier 3: everything else), hard budget halt at $10 reserve. (PITFALLS.md, Pitfall 5)

5. **SVD numerical instability** -- Near-zero singular values produce condition number = inf and NaN in entropy; nearly-equal singular values cause the principal singular vector to flip 180 degrees between steps, creating false spikes that correlate with early-context positions and inflate AUROC. Prevention: clamp condition number at 1e6, use epsilon in entropy computation, track k-subspace (Grassmannian distance) instead of single principal vector direction, scan every evaluation sequence for NaN/Inf and replace with sentinels. These guards must be built into metric computation functions, not added as post-processing. (PITFALLS.md, Pitfall 4)

---

## Implications for Roadmap

Based on research, the module dependency chain and pipeline structure strongly dictate phase order. The anchor config must be demonstrated end-to-end before sweep infrastructure is invested in. Statistical rigor features (position-matched baselines, multiple comparison corrections) affect data collection and analysis schema -- they must be designed in from the start, not retrofitted after results accumulate.

### Phase 1: Reproducible Infrastructure and Graph Foundation

**Rationale:** Everything else depends on correct config serialization, graph generation, seed management, and the result.json schema. These modules have no upstream dependencies and must be demonstrably correct before any GPU time is spent. The schema and metadata format must be finalized here -- changing result.json schema after 50 runs requires re-running completed experiments.

**Delivers:** `ExperimentConfig` dataclass hierarchy with validation and deterministic hashing; custom DCSBM generator with degree-correction sampling (Uniform[0.5, 2.0] theta distribution recommended); graph validation gates (connectivity, path existence, non-triviality); `result.json` schema finalized and validated; seed management with separate graph/walk/model seed streams; RunPod checkpointing strategy documented.

**Addresses features:** Config specification, DCSBM graph generation, block jumper designation, graph validation, result schema.

**Avoids pitfalls:** Degenerate graphs (Pitfall 1), trivial block jumper rules (Pitfall 2), seed conflation (Pitfall 15), schema mismatch (Pitfall 17), missing metadata (Pitfall 20), RunPod preemption (Pitfall 16).

**Research flag:** Standard patterns -- no additional research needed.

### Phase 2: Walk Generation and Training Pipeline

**Rationale:** The transformer and training loop must be built and proven on the anchor config before the sweep is designed. The sufficiency gate is a hard dependency for all downstream phases; it must exist and be enforced before SVD infrastructure is invested in. Discovering a fundamental architecture issue (model cannot learn the rule) at this phase costs far less than discovering it in Phase 3.

**Delivers:** Walk generator with corpus validation, coverage report (every block jumper appears >= 50 times), and config-hash caching; NanoGPT transformer with `SingleHeadAttention(return_qkt=True)` and unit-tested output shapes; training loop with AdamW, cosine schedule with warmup, checkpointing every N steps, loss and compliance curve logging; sufficiency gate evaluated at 50% and 80% of training budget with early abort; verified anchor config passes the gate.

**Addresses features:** Walk generator, corpus validation, NanoGPT transformer, training loop, seed control, checkpointing, training curves, sufficiency gate.

**Avoids pitfalls:** Training never reaching gate (Pitfall 6), corpus inadequately covering jumper vertices (Pitfall 12), non-deterministic GPU ops breaking reproducibility (Pitfall 11), degree correction choice creating pathological graphs (Pitfall 18).

**Research flag:** Standard patterns for training loop. Light research needed on walk length requirements: for large-r configs (r=2w), walk_length >= r + w = 3w to ensure full rule expression within a walk.

### Phase 3: SVD Collection Pipeline

**Rationale:** This is the most computationally expensive and numerically sensitive phase. It must be built correctly once -- streaming storage, batched GPU SVD, numerical guards, context-window warmup, fused evaluation pass. Retrofitting streaming or numerical guards after the sweep is running is a rewrite. All ~20 SVD metrics must have unit tests against analytical solutions before any sweep run.

**Delivers:** Fused evaluation pass (behavioral labels + SVD metrics in a single forward pass); all ~20 SVD metric functions with unit tests (each metric tested against hand-computed known-matrix answers); streaming write to `token_metrics.npz` (never accumulate full per-step tensors in memory); context-window warmup (metrics collected only for positions >= w); numerical guards (NaN/Inf clamping, entropy epsilon, condition number cap at 1e6, Grassmannian distance for subspace tracking); anchor config profiled for wall time and storage size per phase.

**Addresses features:** 4-class behavioral evaluation, QK^T extraction, all ~20 SVD metrics, token-level time series storage, `failure_index` annotation.

**Avoids pitfalls:** SVD memory blowup (Pitfall 3), numerical instability (Pitfall 4), O(d^3) scaling cost (Pitfall 8), context padding poisoning metrics (Pitfall 19), recomputing metrics in analysis layer (Anti-pattern 5 in ARCHITECTURE.md).

**Research flag:** Needs deeper research before planning. Key questions: (1) what is the actual memory footprint for w=256 configs -- measure on anchor first; (2) float16 precision sufficiency for each metric type -- condition number likely needs float32; (3) how many evaluation walks are needed for 400 violation events at the expected jumper-event frequency. Run anchor profiling as the first task of this phase.

### Phase 4: Analysis, Statistical Rigor, and Reporting

**Rationale:** With a complete `result.json` from the anchor config, the analysis infrastructure can be built and validated against known results before the sweep expands. Discovering a flaw in the AUROC computation after 100 sweep runs is expensive. The position-matched baseline and multiple comparison correction must be validated on the anchor before they are trusted for the full sweep.

**Delivers:** Predictive horizon AUROC at each lookback j with position-matched baselines and shuffle controls; Holm-Bonferroni correction across pre-registered primary metrics; effect sizes (Cohen's d); bootstrap CIs on AUROC (DeLong's method or bootstrap over sequences); all plot types from plotting guide (event-aligned metric plots, AUROC vs j curves, confusion matrix, pre/post failure distributions, heatmaps); single-experiment HTML report with embedded base64 figures and reproduction command; comparison HTML report with config diff table.

**Addresses features:** Predictive horizon analysis, event-aligned plots, AUROC curves, confusion matrix, single-experiment and comparison reports.

**Avoids pitfalls:** Position confounds (Pitfall 7), multiple comparisons inflation (Pitfall 9), AUROC class imbalance (Pitfall 10), incorrect baselines (Pitfall 14).

**Research flag:** Light research needed on DeLong's method availability in scipy (confirm `scipy.stats` has equivalent, or identify the correct package). Statistical methods are well-established; the implementation choices need verification.

### Phase 5: Sweep Infrastructure and Full Execution

**Rationale:** Only after the full pipeline is verified end-to-end on the anchor config should sweep infrastructure be built. Sweep execution is the final phase because its cost is irreversible: GPU time spent on buggy or misconfigured runs cannot be recovered. The comparison report and secondary analyses (metric correlation, importance ranking, predictive horizon heatmap) complete the scientific deliverable.

**Delivers:** Three-tier priority-ordered job queue (Tier 1: anchor r-sweep 24 runs; Tier 2: architecture + w sweeps ~63 runs; Tier 3: graph parameter and corpus sweeps); graph and walk caching by config hash (prevent redundant regeneration across shared-graph configs); 3-seed-per-config outer loop; budget tracker with hard halt at $10 reserve; sweep state persisted as `sweep_state.json` for resume after preemption; secondary analyses (SVD metric correlation matrix, metric importance ranking, predictive horizon heatmap across r/w grid); math verification PDF (pylatex, with Jinja2+raw-LaTeX fallback).

**Addresses features:** Job queue with priority ordering, 3 seeds per config, parameter sweep enumeration, comparison reports with config diffs, metric importance ranking, predictive horizon heatmap, math verification PDF.

**Avoids pitfalls:** Budget exhaustion (Pitfall 5), monolithic sweep script with no resume (Anti-pattern 4 in ARCHITECTURE.md), preemption data loss (Pitfall 16).

**Research flag:** Standard patterns -- job queue and sweep orchestration are straightforward Python. pylatex stability on RunPod's TeXLive install needs a quick verification check before committing to it; have the Jinja2+LaTeX fallback ready.

### Phase Ordering Rationale

- **Infrastructure first**: Config schema, seed management, and result.json format must be frozen before any code writes to them. Changing the schema mid-sweep requires re-running completed experiments.
- **Gate enforced before measurement**: The sufficiency gate (Phase 2) must be verified passing on the anchor config before SVD infrastructure (Phase 3) is built. SVD machinery for a model that cannot learn the rule is wasted effort.
- **Anchor profiling gates sweep planning**: Phase 3 ends with a complete anchor config run profiled for wall time and storage. These measurements directly determine which Tier 2 and Tier 3 sweep configs are affordable within the $100 budget.
- **Analysis validated before sweep expansion**: Phase 4 validates that the analysis pipeline produces sensible results (AUROC ~0.5 on shuffle control, consistent with expectations on anchor) before the sweep is launched. Discovering a confound after 100 runs is expensive.
- **Sweep last**: Phase 5 consumes most of the $100 budget. Every bug must be caught in Phases 1-4.

### Research Flags

Phases needing deeper research during planning:
- **Phase 3 (SVD Collection):** Batching strategy specifics, float16 vs float32 per metric type, actual memory footprint for w=256. Run anchor profiling as the first task; let measurements drive planning decisions rather than estimates.
- **Phase 4 (Analysis):** Confirm DeLong's method or equivalent is available in scipy. Verify that position-matched baseline sampling produces statistically sufficient violation events at expected jumper-event frequency.

Phases with standard patterns (can skip research-phase):
- **Phase 1 (Infrastructure):** Config dataclasses, NetworkX usage, scipy sparse -- all well-documented with standard patterns.
- **Phase 2 (Walk Generation and Training):** PyTorch training loop, cosine schedule, checkpointing -- established patterns.
- **Phase 5 (Sweep):** Job queue, file-based caching, JSON state persistence -- straightforward Python.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | API choices (torch.linalg.svd, NetworkX SBM gap, custom DCSBM) are HIGH; specific version numbers (PyTorch 2.5, matplotlib 3.8) are MEDIUM -- based on training data through May 2025, not verified against live PyPI. Run `pip index versions torch` before finalizing pyproject.toml. |
| Features | HIGH | Feature set derived directly from project spec (combined-spec.md). Complexity estimates are MEDIUM -- based on experience with similar systems, not benchmarked against this specific codebase. |
| Architecture | HIGH | Module boundaries follow directly from spec; data flow is dictated by the pipeline's dependency chain; interface contracts are clearly specified. SVD batching speedup estimates are MEDIUM -- need profiling on actual RunPod hardware. |
| Pitfalls | HIGH | Pitfalls 1-7 draw on established numerical linear algebra and graph theory. Budget estimates (Pitfall 5) use RunPod pricing that may have changed since research was conducted (MEDIUM). |

**Overall confidence:** HIGH

### Gaps to Address

- **DCSBM degree correction distribution**: the spec does not specify which distribution to use for theta_i parameters. Research recommends Uniform[0.5, 2.0] (bounded, avoids isolated vertices and pathological hubs). This choice should be documented in the config, fixed across all runs (not swept), and validated by checking that max_degree / median_degree < 20 after graph generation.

- **Library version verification**: run `pip index versions torch networkx matplotlib seaborn jinja2 pylatex` on the actual RunPod instance before finalizing `pyproject.toml`. PyTorch 2.6 or 2.7 may be current stable; verify `torch.linalg.svd` API is unchanged.

- **Walk length at anchor config**: the spec sweeps walk_length at 2w, 4w, 8w. PITFALLS.md recommends l >= r + w to ensure full rule expression within any walk. For the anchor config (r=0.9w, w=64), l=2w=128 satisfies this (r+w = 1.9w = 121.6). But for large-r configs (r=2w), l=2w is insufficient. Use l=4w as the anchor value. Confirm against spec before locking in the anchor config definition.

- **Evaluation walk count for statistical power**: for AUROC CI half-width of 0.05, approximately 400 violation events per config are needed (per PITFALLS.md Pitfall 10). Given the expected frequency of block jumper events in random walks (depends on jumper vertex density and walk length), compute the required number of evaluation walks before locking in the eval corpus size. This affects Phase 3 storage estimates significantly.

- **pylatex stability on RunPod**: pylatex's API can be finicky with complex math environments. A quick test of pylatex + pdflatex on the RunPod base image before Phase 5 will determine whether the Jinja2+raw-LaTeX fallback is needed. Do not block any other phase on the math verification PDF.

---

## Sources

### Primary (HIGH confidence)
- `combined-spec.md` (repository root) -- PRIMARY source for all feature requirements, result schema, plotting guide, reporting guide
- `.planning/PROJECT.md` -- project constraints, scope, and anchor configuration definitions
- PyTorch `torch.linalg.svd` documentation -- batched GPU SVD API, full_matrices flag, CUDA tensor support
- NetworkX community generators documentation -- confirms `stochastic_block_model` exists, degree-corrected variant does not
- Karrer & Newman (2011) "Stochastic blockmodels and community structure" -- formal DCSBM definition and degree-correction sampling formula
- Golub & Van Loan "Matrix Computations" -- SVD numerical stability, condition number behavior, standard eps-clamping practice

### Secondary (MEDIUM confidence)
- Karpathy nanoGPT (github.com/karpathy/nanoGPT) -- architectural reference for NanoGPT-scale single-file transformer
- PyTorch deterministic algorithm documentation -- `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- Erdos-Renyi connectivity threshold analysis extended to SBM -- minimum expected degree for strong connectivity
- DeLong et al. (1988) -- AUROC confidence interval method
- Benjamini & Hochberg (1995) -- FDR control procedure for multiple comparisons

### Tertiary (LOW confidence)
- Library version ranges in `pyproject.toml` (STACK.md) -- based on training data through May 2025, not verified against live PyPI; verify before committing
- RunPod pricing (RTX 3090 ~$0.44/hr on-demand, ~$0.22/hr spot) -- pricing subject to change; verify current rates before budget planning
- NanoGPT training dynamics (3e-4 LR with cosine annealing) -- established community practice, not verified for this specific vocabulary size and architecture

---
*Research completed: 2026-02-24*
*Ready for roadmap: yes*
