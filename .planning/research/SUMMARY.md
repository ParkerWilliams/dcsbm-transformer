# Project Research Summary

**Project:** DCSBM Transformer v1.1 -- Journal Feedback Capabilities
**Domain:** Research software -- extending an SVD-based hallucination prediction framework with reviewer-requested features for journal resubmission
**Researched:** 2026-02-26
**Confidence:** HIGH

## Executive Summary

The v1.1 milestone adds 8 features to address journal reviewer feedback on the DCSBM transformer paper. These features fall into three categories: (A) validating the core signal claim (null model baseline, multi-head ablation, pre-registration), (B) enriching the analysis toolkit (PR curves, calibration, softmax filtering bound, spectrum trajectory with curvature/torsion), and (C) addressing practical concerns (SVD computational overhead benchmarking, cheaper approximations). The existing v1.0 stack -- PyTorch 2.10, NumPy 2.3, SciPy 1.17, Matplotlib 3.10 -- is fully sufficient. Zero new pip dependencies are needed. Every v1.1 capability maps to APIs already installed in the environment (`torch.svd_lowrank`, `torch.linalg.svdvals`, `torch.utils.benchmark`, `np.gradient`, `scipy.optimize.minimize`).

The recommended approach is to build features in strict dependency order, with multi-head ablation built last despite being the most reviewer-critical feature. Multi-head is the most invasive change (touching 10+ files across model, evaluation, analysis, and visualization layers), and building it first would force every subsequent feature to handle both single-head and multi-head codepaths simultaneously. Instead, build all additive features first (null model, PR curves, softmax bound, benchmarks, pre-registration, spectrum trajectory, compliance curve), validate them on the single-head architecture, and then introduce multi-head support as the final integration. Pre-registration must be committed before any v1.1 analysis runs -- this is not optional, it is a methodological requirement that cannot be backdated.

The key risks are: (1) confounding the null model baseline (using a different graph or unmatched temporal structure), which would invalidate the "real signal" claim all reviewers demand; (2) silent semantic breakage during multi-head integration, where tensor shape changes propagate through the pipeline without errors but produce meaningless metrics; and (3) mathematical errors in the softmax filtering bound derivation, particularly using the wrong Lipschitz constant (must be 1/2 per arXiv:2510.23012, not 1) or forgetting the 1/sqrt(d_k) scaling factor. All three risks have concrete prevention strategies with detection tests documented in the research.

## Key Findings

### Recommended Stack

The v1.0 stack requires no changes. Every v1.1 feature maps to existing PyTorch, NumPy, or SciPy APIs. The project's established pattern of implementing from first principles (e.g., AUROC via Mann-Whitney U rather than sklearn) extends naturally to PR curves (~25 lines of NumPy), calibration curves (~20 lines), and discrete curvature/torsion (~30 lines). Adding scikit-learn for three function calls would introduce a 50MB dependency contrary to the project's philosophy and adds opacity that reviewers cannot easily inspect.

**Core technologies (unchanged):**
- **PyTorch 2.10:** Model, training, SVD (`torch.linalg.svd`, `torch.svd_lowrank`, `torch.linalg.svdvals`), benchmarking (`torch.utils.benchmark`)
- **NumPy 2.3:** PR curves, calibration, discrete Frenet frame (curvature/torsion), null distribution statistics
- **SciPy 1.17:** Bootstrap CIs, Mann-Whitney U, logistic calibration via `scipy.optimize.minimize`
- **Matplotlib 3.10 + Seaborn 0.13:** All new visualizations (PR curves, reliability diagrams, spectral trajectory plots, benchmark charts)

**Key v1.1 APIs already available (no install):**
- `torch.svd_lowrank(A, q, niter)` -- randomized SVD approximation (Halko 2009), 2-10x speedup verified
- `torch.linalg.svdvals(A)` -- values-only SVD for the 6 metrics that do not need singular vectors
- `torch.utils.benchmark.Timer` -- proper CPU/GPU benchmarking with warmup and statistical reporting
- `np.gradient()` -- finite differences for curvature/torsion computation

**Two-tier SVD strategy recommended:**
- Tier 1 (cheap): `torch.linalg.svdvals()` for metrics needing only singular values (stable_rank, spectral_entropy, spectral_gap, condition_number, effective_rank, spectral_norm) -- 6 of 9 metrics
- Tier 2 (moderate): `torch.svd_lowrank(q=k+5)` for metrics needing singular vectors (grassmannian_distance, rank1_residual_norm, read_write_alignment) -- 3 of 9 metrics
- Expected overhead reduction: 40-60% depending on matrix size

See `.planning/research/STACK.md` for full API verification, benchmark results, and alternatives analysis.

### Expected Features

**Must have (table stakes -- reviewer-required for resubmission):**
- **[1] Null model baseline** -- all 3 reviewers: "How do you know the signal is not just normal attention dynamics?" Generate clean (no-jumper) sequences, compute Grassmannian drift null distribution, position-matched statistical comparison
- **[3] Multi-head ablation (1h/2h/4h)** -- reviewer concern: "Is the signal real or an artifact of single-head multiplexing?" Per-head SVD extraction, signal concentration analysis
- **[4] Precision-recall curves + calibration** -- standard for rare-event classifiers; AUROC alone is misleading for imbalanced classes
- **[5] Pre-registration framework** -- held-out evaluation protocol, locks analysis plan before seeing confirmatory results
- **[2] Softmax filtering bound** -- mathematician reviewer: "Derive the theoretical lag prediction"

**Should have (differentiators that strengthen the paper):**
- **[6] Sharp compliance curve** -- r/w sweep showing phase transition from near-perfect to failed compliance; the "money figure"
- **[7] Full spectrum trajectory with curvature/torsion** -- novel geometric analysis of spectral curve shape
- **[8] SVD computational overhead** -- benchmarks quantifying practical cost, cheaper approximation candidates

**Defer (anti-features -- explicitly do NOT build):**
- Multi-head beyond 4 heads (d_k too small at d_model=128)
- Full Bayesian calibration / Platt scaling (undermines pre-registration claims)
- Real-time SVD monitoring (v1.1 is offline research, not deployment)
- Symbolic math verification via SymPy (manual LaTeX is faster)
- Gradient-based attribution for SVD metrics (different research question)
- General-purpose spectral trajectory analysis library (premature generalization)

See `.planning/research/FEATURES.md` for the full feature dependency graph, complexity matrix, and MVP recommendation.

### Architecture Approach

The v1.1 features integrate into the existing v1.0 pipeline through three patterns: (A) core module modifications for multi-head and full spectrum storage, (B) additive analysis modules that extend existing pipelines without modifying them, and (C) standalone tools for benchmarking. The critical architectural constraint is backward compatibility: single-head results must emit both v1.0-format NPZ keys (`qkt.layer_0.grassmannian_distance`) and v1.1-format keys (`qkt.layer_0.head_0.grassmannian_distance`). Schema version bumps from "1.0" to "1.1" with v1.1 as a strict superset.

**Major components:**
1. **`src/analysis/null_baseline.py`** (NEW) -- Extract and compare null Grassmannian distributions from jumper-free experiment runs
2. **`src/model/attention.py`** (MODIFIED) -- Multi-head CausalSelfAttention with per-head QKT extraction; `AttentionInternals.qkt` becomes `[B, H, T, T]`
3. **`src/evaluation/pipeline.py`** (MODIFIED) -- Per-head SVD extraction loop, full spectrum storage, dual key emission for backward compatibility
4. **`src/analysis/auroc_horizon.py`** (EXTENDED) -- PR curves and calibration added as new functions alongside existing AUROC (not modifying it)
5. **`src/analysis/softmax_bound.py`** (NEW) -- Theoretical bound verification using existing NPZ data; no new data collection needed
6. **`src/analysis/spectrum.py`** (NEW) -- Curvature/torsion via discrete Frenet-Serret on full singular value trajectories
7. **`src/benchmarks/svd_overhead.py`** (NEW) -- Standalone SVD timing benchmarks; zero integration risk

**Key architectural patterns:**
- Feature flags via config tags (e.g., `tags=("null_model",)`) rather than boolean config fields
- Dual key emission for backward compatibility when NPZ key formats change
- Post-hoc aggregation over in-pipeline computation for cross-experiment analysis (compliance curve, null comparison)
- New functions alongside existing ones (PR curves next to AUROC), never modifying existing computation

See `.planning/research/ARCHITECTURE.md` for full integration map, data flow changes, build order analysis, and anti-patterns to avoid.

### Critical Pitfalls

1. **Null model confounded with signal** -- Using a different graph, unmatched temporal structure, or a model trained only on clean data produces a bad null that does not answer the reviewer's question. Prevention: same graph, same trained model, jumper-free evaluation walks only, position-matched Grassmannian comparison, report Cohen's d effect sizes and full percentile envelope (5th/25th/50th/75th/95th). Must be the first analysis feature built.

2. **Multi-head breaks SVD extraction semantics** -- Shape changes from `[B, T, T]` to `[B, H, T, T]` propagate silently through 10+ modules. Per-head QKT rank is bounded by d_k (32 for 4h), not d_model (128), fundamentally changing metric interpretation. Prevention: add head dimension to all tensor contracts, per-head metric keys, unit test that rank(QKT) <= d_k, report d_k context with all multi-head metrics. Build multi-head in 4 stages with passing tests at each stage.

3. **Softmax bound math errors** -- Using Lipschitz constant of 1 instead of 1/2 (cite arXiv:2510.23012), forgetting 1/sqrt(d_k) scaling (11x error for d_k=128), confusing row-wise and matrix-wise bounds, ignoring causal mask effects on per-row entropy. Prevention: empirical verification with 1000+ perturbations; if >5% exceed the bound, the derivation has an error. Test both random and adversarial perturbation directions.

4. **Pre-registration invalidated by data-dependent decisions** -- Writing pre-registration after seeing any v1.1 results makes it post-hoc and indefensible. Prevention: commit pre-registration document before any v1.1 analysis code runs. Frame honestly as "informed hypothesis testing" based on v1.0 exploratory results. Implement held-out walk split (50/50 exploratory/confirmatory). Maintain deviation log throughout.

5. **Breaking v1.0 functionality during integration** -- v1.1 touches nearly every pipeline layer. The v1.0 audit already identified P0 integration breaks. Prevention: fix v1.0 P0 issues first (stub run_experiment.py, set_seed never called, predictive_horizon never written). Create golden regression test suite. Feature flags over modifications. Separate NPZ namespaces for v1.1 data.

See `.planning/research/PITFALLS.md` for all 14 pitfalls with detection tests, prevention strategies, and phase-specific warnings.

## Implications for Roadmap

Based on research, the suggested phase structure follows dependency order with the most invasive change (multi-head) deferred to the end. Pre-registration must be committed before any analysis runs. All features except multi-head are additive and low-risk.

### Phase 1: Foundation -- Pre-Registration and v1.0 Stabilization
**Rationale:** Pre-registration must be committed before any v1.1 analysis to maintain methodological credibility. v1.0 integration breaks must be fixed before extending the codebase -- building on a broken foundation doubles debugging effort. The golden regression test suite provides the safety net for all subsequent phases.
**Delivers:** Locked pre-registration document with primary hypothesis, held-out protocol, and deviation log. Stable v1.0 baseline with golden regression tests. Fixed P0 integration issues.
**Addresses:** Feature [5] (Pre-Registration Framework)
**Avoids:** Pitfall 4 (post-hoc rationalization), Pitfall 5 (breaking existing functionality)

### Phase 2: Null Model Baseline
**Rationale:** All 3 reviewers demanded this. It validates the core signal claim that all other analyses depend on. If the null model shows Grassmannian drift is indistinguishable from background, the entire project premise is challenged -- better to know this first before investing in enrichment features.
**Delivers:** Null distribution of Grassmannian drift from jumper-free sequences. Statistical comparison (Mann-Whitney U, Cohen's d) of null vs. signal at each lookback distance. Marchenko-Pastur reference distribution. Null overlay on event-aligned plots. `null_baseline` section in result.json.
**Addresses:** Feature [1] (Null Model Baseline)
**Avoids:** Pitfall 1 (confounded controls -- same graph, same model, position-matched)

### Phase 3: Evaluation Enrichment -- PR Curves, Calibration, and SVD Benchmarks
**Rationale:** These three features are additive (no core module modifications), independent of each other, and address separate reviewer concerns. PR curves and calibration extend the existing AUROC pipeline using the same event extraction and metric values. SVD benchmarks are completely standalone. Grouping allows parallel development with no cross-dependencies.
**Delivers:** Precision-recall curves with AUPRC per metric per lookback. Reliability diagrams with ECE. SVD timing benchmarks (full vs. randomized vs. values-only). Two-tier SVD strategy recommendation. Cost summary table for report.
**Addresses:** Feature [4] (PR + Calibration), Feature [8] (SVD Overhead)
**Avoids:** Pitfall 6 (imbalanced PR baseline -- always report prevalence), Pitfall 8 (GPU timing artifacts -- CUDA events with warmup), Pitfall 12 (no probability outputs -- isotonic regression on exploratory split)

### Phase 4: Softmax Filtering Bound
**Rationale:** Mathematically the hardest feature (novel derivation). Independent of other features -- uses existing NPZ data, does not modify any pipeline. Can be developed in parallel with phases 2-3 if resources allow. The LaTeX derivation is the bottleneck; empirical verification is straightforward once the bound formula exists.
**Delivers:** LaTeX derivation of epsilon-bound from QKT perturbation to AVWo spectral change. Empirical verification with random and adversarial perturbations. Bound tightness visualization (theoretical envelope vs. empirical measurements). New MATH_SECTIONS entry in math PDF.
**Addresses:** Feature [2] (Softmax Filtering Bound)
**Avoids:** Pitfall 3 (wrong Lipschitz constant -- use 1/2, cite arXiv:2510.23012), Pitfall 10 (wrong perturbation direction -- test adversarial)

### Phase 5: Advanced Analysis -- Spectrum Trajectory and Compliance Curve
**Rationale:** Differentiator features that elevate the paper beyond "addresses reviewer concerns" to "thorough and impressive." Full spectrum tracking requires modifying the evaluation pipeline (store raw S vectors in NPZ), which should happen after the pipeline is stable from phases 2-3. The compliance curve requires 45 training runs (15 r values x 3 seeds), making it GPU-expensive and best scheduled after analysis infrastructure is complete.
**Delivers:** Full singular value vectors stored per step in NPZ. Curvature and torsion time series as new metrics fed into AUROC pipeline. Sharp compliance curve showing r/w phase transition with dual-axis visualization (compliance + predictive horizon). Spectral trajectory plots.
**Addresses:** Feature [7] (Full Spectrum Trajectory + Curvature/Torsion), Feature [6] (Sharp Compliance Curve)
**Avoids:** Pitfall 7 (numerical differentiation noise -- Savitzky-Golay smoothing, Grassmannian curvature preferred over pointwise), Pitfall 9 (undersampled transition -- dense r/w near 0.7-1.0, 3 seeds per value), Pitfall 11 (NPZ size -- separate spectrum file, float16, QKT-only by default)

### Phase 6: Multi-Head Ablation
**Rationale:** Built last because it is the most invasive change, touching model, types, config, evaluation, analysis, and visualization. All other features are validated on single-head first, then extended to handle multi-head output. The staged build approach avoids the monolithic refactor anti-pattern: (1) attention.py + types.py + tests, (2) transformer.py + block.py + tests, (3) pipeline.py + tests, (4) analysis + visualization + tests. Each stage has a passing test suite before the next begins.
**Delivers:** Multi-head CausalSelfAttention (1h/2h/4h variants). Per-head QKT SVD extraction with `[B, n_layers, n_heads, T, T]` tensor contracts. Per-head AUROC with signal concentration analysis. Ablation comparison on matched configs. Backward-compatible dual key emission for single-head.
**Addresses:** Feature [3] (Multi-Head Ablation)
**Avoids:** Pitfall 2 (shape/semantics breakage -- staged build, per-head keys, rank bound tests, d_k context), Pitfall 5 (v1.0 breakage -- dual key emission, golden regression tests)

### Phase Ordering Rationale

- **Pre-registration before all analysis:** Methodological requirement. Cannot be backdated. The pre-registration document must exist in git history before any v1.1 results are computed.
- **Null model before enrichment:** All other analyses depend on establishing the signal is real. If null model fails, the project pivots -- better to know in phase 2 than phase 6.
- **Additive features before invasive changes:** PR curves, calibration, benchmarks, and softmax bound are all additive (new functions, new modules). They pose minimal regression risk and can be developed and tested independently.
- **Multi-head last:** Every other feature can be built and validated on the single-head architecture. Multi-head changes propagate through the entire stack. Building it first would force every subsequent feature to develop against both single-head and multi-head codepaths, doubling testing complexity.
- **Compliance curve in phase 5:** Requires 45 GPU training runs. Must be scheduled after analysis infrastructure is complete to avoid wasting compute on pipeline bugs.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 4 (Softmax Bound):** Novel mathematical derivation. The specific bound chain through softmax + value projection + output projection is new work. Bound tightness unknown until derived. The math is the hard part, not the tooling. Needs careful attention to causal mask effects on per-row Lipschitz behavior.
- **Phase 5 (Spectrum Trajectory):** Discrete Frenet-Serret on noisy SVD output is numerically delicate. Scientific payoff uncertain -- curvature/torsion may or may not outperform existing scalar metrics. Treat as exploratory. Storage overhead needs empirical measurement.
- **Phase 6 (Multi-Head):** Per-head SVD extraction and signal concentration analysis for this architecture needs careful interface design. WvWo product semantics change (per-head OV circuits are rectangular `[d_head, d_model]`, not square). Metrics like read_write_alignment may not apply.

Phases with standard patterns (skip deep research):
- **Phase 1 (Pre-Registration + Stabilization):** Well-documented methodology. Process, not algorithm.
- **Phase 2 (Null Model):** Standard null hypothesis testing. Reuses existing pipeline with config change (`n_jumpers_per_block=0`).
- **Phase 3 (PR + Calibration + Benchmarks):** Well-established techniques. PR curves are simpler than existing AUROC implementation. `torch.utils.benchmark` is purpose-built.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero new dependencies. Every API verified as available in installed environment. Benchmark results for SVD approximations validated empirically. |
| Features | HIGH | 6 of 8 features use standard, well-documented techniques. Clear table stakes vs. differentiator distinction. Feature prioritization consistent across all 4 research files. |
| Architecture | HIGH | Dependency graph is clear. Build order well-justified by both dependency analysis and risk mitigation. Backward compatibility strategy (dual key emission, schema versioning) is sound. Multi-head integration complexity is the one MEDIUM area. |
| Pitfalls | HIGH | 14 pitfalls identified with concrete prevention strategies and detection tests. Critical pitfalls backed by published proofs (softmax Lipschitz), established methodology (pre-registration), and verified codebase analysis (shape propagation). |

**Overall confidence:** HIGH

### Gaps to Address

- **Softmax bound tightness:** The theoretical bound may be vacuous (empirical perturbations reach <1% of bound). Cannot be determined until derivation is complete. If vacuous, report honestly as a negative result rather than a contribution.

- **Curvature/torsion as predictive features:** Scientific payoff uncertain. Discrete Frenet-Serret on noisy data is numerically delicate. Treat as exploratory analysis only. Do not stake paper claims unless signal clearly outperforms existing scalar metrics.

- **Multi-head WvWo semantics:** Per-head OV circuits produce rectangular matrices `[d_head, d_model]`. SVD metrics assuming square input (read_write_alignment) need special handling or must be skipped for multi-head WvWo. Design decision deferred to phase 6 planning.

- **GPU benchmark timing:** `torch.cuda.Event` pattern is well-documented but untested locally (CPU-only environment). May reveal that SVD overhead is <1% of evaluation time, making the approximation analysis informational rather than critical.

- **NPZ storage for full spectrum:** Estimated 125-250 MB per experiment (QKT, float16). Compliance curve sweep (45 runs) reaches ~6 GB. May need selective storage (specific layers or r values only).

- **v1.0 P0 integration breaks:** Stub `run_experiment.py`, `set_seed` never called, `predictive_horizon` never written to result.json. Must be fixed in phase 1 before any v1.1 features are added. Scope of fixes needs assessment during phase planning.

## Sources

### Primary (HIGH confidence)
- Existing codebase analysis -- direct code reading of all modules referenced in research files
- [PyTorch torch.svd_lowrank](https://docs.pytorch.org/docs/stable/generated/torch.svd_lowrank.html) -- API verified, benchmarks run
- [PyTorch torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html) -- Timer API verified working
- [Softmax Lipschitz constant = 1/2](https://arxiv.org/abs/2510.23012) -- tight bound, published proof
- [Pre-registration deviation reporting](https://journals.sagepub.com/doi/10.1177/25152459231213802) -- Willroth & Atherton 2024
- [PR curves for rare events](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- Saito & Rehmsmeier
- [Null models in network neuroscience](https://www.nature.com/articles/s41583-022-00601-9) -- Nature Reviews, design principles
- [Halko, Martinsson, Tropp (2009)](https://arxiv.org/abs/0909.4061) -- Algorithm behind `torch.svd_lowrank`
- [Marchenko-Pastur distribution](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution) -- null model for random matrix singular values

### Secondary (MEDIUM confidence)
- [Local Lipschitz bound for self-attention](https://arxiv.org/abs/2507.07814) -- Li et al. 2025, tighter attention-dependent bounds
- [Small Singular Values Matter](https://arxiv.org/abs/2410.17770) -- Shlens et al., NeurIPS 2025, MP as transformer null model
- [Shape Analysis under Frenet-Serret](https://arxiv.org/abs/2511.17065) -- Chassat et al. 2023/2025, discrete curvature framework
- [Interpreting Transformers via Attention Head Intervention](https://arxiv.org/abs/2601.04398) -- Basile et al. 2025, per-head specialization
- [Pre-registration for predictive modeling](https://arxiv.org/html/2311.18807) -- ML-specific templates
- [The Lipschitz Constant of Self-Attention](http://proceedings.mlr.press/v139/kim21i/kim21i.pdf) -- Kim et al., ICML 2021

### Tertiary (needs validation during implementation)
- [PyTorch batched SVD performance issue](https://discuss.pytorch.org/t/batched-svd-lowrank-being-much-slower-than-loop-implementation-both-cpu-and-gpu/119336) -- performance may vary by version
- [Canonical angle bounds for randomized SVD](https://arxiv.org/html/2211.04676) -- accuracy bounds for Grassmannian from randomized SVD
- [Spectral rank collapse in attention layers](https://arxiv.org/html/2410.07799v2) -- recent preprint, needs verification

---
*Research completed: 2026-02-26*
*Ready for roadmap: yes*
