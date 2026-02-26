# Feature Landscape

**Domain:** Journal feedback capabilities for DCSBM transformer SVD hallucination prediction framework
**Researched:** 2026-02-26
**Mode:** Ecosystem (What does each v1.1 capability require and how does it integrate with v1.0?)
**Focus:** New features only -- null model baselines, softmax filtering bounds, multi-head ablation, PR curves + calibration, pre-registration framework, sharp compliance curves, full spectrum trajectory tracking, SVD computational overhead analysis

---

## Table Stakes

Features that reviewers explicitly requested or that are standard practice for the claims being made. Missing = paper is rejected or returned for major revision.

### 1. Null Model Baseline (Grassmannian Drift on Clean Sequences)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Generate clean (no-jumper) sequences from identical DCSBM graph | All 3 reviewers asked: "How do you know the Grassmannian drift is not just normal attention dynamics?" | Low | Reuse existing `walk/generator.py` with `n_jumpers_per_block=0`. Train a separate model on clean walks, or use same model on clean eval walks |
| Compute Grassmannian distance time series on clean sequences | Establishes the null distribution for subspace drift | Low | Reuse existing `grassmannian_distance()` from `svd_metrics.py` via `fused_evaluate()` pipeline |
| Marchenko-Pastur reference distribution for QK^T singular values | Random matrix theory baseline: are the SVD metrics distinguishable from random matrices of matching dimensions? | Medium | Compute theoretical MP distribution for T x T matrices with aspect ratio gamma = T/D. Compare empirical singular value histogram to MP density |
| Two-sample statistical test: drift(violation) vs drift(clean) | Quantify separation between null and signal. Must be more than AUROC -- need p-value that Grassmannian drift is elevated before violations vs clean baseline | Low | Mann-Whitney U or KS test between matched step positions. Reuse existing `auroc_from_groups()` pattern |
| Null distribution stored in result.json and visualized | Report must show the null alongside the signal | Low | Extend result.json schema with `null_model` block. Add overlay to event-aligned plots |

**Standard approach:** Run the full SVD pipeline on walks that never encounter jumper vertices. This gives the "background" Grassmannian drift rate. The signal claim is validated if drift before violations is statistically significantly elevated above this background. The Marchenko-Pastur distribution (Marchenko & Pastur, 1967) provides a theoretical baseline: for a T x T random matrix with i.i.d. entries, the singular value density follows a known distribution. Deviations from MP indicate learned structure rather than noise. Recent work by Shlens et al. (2024, "Small Singular Values Matter," NeurIPS 2025) demonstrates this approach for analyzing transformer weight matrices.

**Existing infrastructure dependencies:**
- `walk/generator.py` -- set `n_jumpers_per_block=0`
- `evaluation/pipeline.py` `fused_evaluate()` -- runs unchanged on clean walks
- `svd_metrics.py` `grassmannian_distance()` -- unchanged
- `analysis/auroc_horizon.py` -- add null comparison mode
- `visualization/event_aligned.py` -- add null overlay trace

**Confidence:** HIGH -- standard technique, well-understood mathematically, straightforward to implement with existing infrastructure.

---

### 2. Softmax Filtering Bound (QK^T -> AVWo Perturbation Propagation)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Theoretical derivation of epsilon-bound from QK^T perturbation to AVWo spectral change | Mathematician reviewer: "Derive the theoretical lag prediction -- why should QK^T instability propagate downstream?" | High | This is a math derivation, not primarily code. The Jacobian of softmax has known spectral norm bounds |
| LaTeX document with full derivation | Must be peer-reviewable standalone math | Medium | Extend existing `math_pdf.py` MATH_SECTIONS with new derivation section |
| Empirical verification: measure actual propagation vs theoretical bound | Verify the bound is not vacuous (i.e., it actually constrains the observed propagation) | Medium | Perturb QK^T by controlled epsilon at specific steps, measure resulting AVWo change, compare to bound |
| Bound tightness visualization | Show theoretical bound alongside empirical measurements | Low | Plot epsilon vs actual downstream change with theoretical envelope |

**Standard approach:** The perturbation path is QK^T -> softmax -> A -> AV -> AVWo. The softmax Jacobian J_softmax has spectral norm bounded by 1/2 in the L1 norm (Gao & Pavel, 2017), or more precisely, the spectral norm depends on the attention distribution's concentration. Recent work by Kim et al. (ICML 2021, "The Lipschitz Constant of Self-Attention") and Li et al. (2025, "Pay Attention to Attention Distribution") provides tighter local Lipschitz bounds that depend on the ordinal statistics of the attention distribution.

The derivation chain:
1. Start with perturbation: QK^T -> QK^T + epsilon * E
2. Through softmax: |delta_A| <= (spectral_norm of J_softmax) * |epsilon * E|
3. Through value multiplication: |delta(AV)| <= |delta_A| * ||V||
4. Through output projection: |delta(AVWo)| <= |delta(AV)| * ||Wo||

The key insight for the lag prediction is: when QK^T is near a rank transition (spectral gap closing), the softmax Jacobian amplifies perturbations because the attention distribution is transitioning between concentrated and diffuse states.

**Existing infrastructure dependencies:**
- `model/attention.py` -- extract J_softmax at specific steps (new hook)
- `evaluation/pipeline.py` -- add perturbation injection mode
- `reporting/math_pdf.py` -- add new MATH_SECTIONS entry
- `svd_metrics.py` -- unchanged (metrics computed on perturbed output)

**Confidence:** MEDIUM -- the theoretical framework is well-established (Lipschitz bounds on softmax are known), but the specific derivation chain for this architecture and connecting it to the lag prediction is novel. The bound may be loose.

---

### 3. Multi-Head Ablation (1h, 2h, 4h) with Per-Head SVD Signal Concentration

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Multi-head attention module (2h, 4h variants) | Reviewer concern: "Is the signal real or an artifact of single-head multiplexing?" | High | Major model change. Must split d_model into n_heads sub-dimensions, reshape Q/K/V, independent attention per head, concatenate, project through Wo |
| Per-head QK^T extraction | Each head produces its own T x T attention matrix; SVD analysis per head | Medium | Extend `AttentionInternals` to store per-head matrices. Shape changes from [B, T, T] to [B, n_heads, T, T] |
| Per-head SVD metric computation | Same metrics, applied independently to each head's QK^T | Medium | Loop over heads or batch the SVD computation. Key question: which head(s) carry the predictive signal? |
| Signal concentration analysis: entropy of AUROC across heads | Does one head specialize in rule-tracking while others do something else? | Low | Compute AUROC per head, measure entropy/Gini of the AUROC distribution across heads |
| Ablation comparison: 1h vs 2h vs 4h on matched configs | Same graph, same walks, same training -- only n_heads varies | Medium | Need to relax the `n_heads == 1` constraint in ExperimentConfig, retrain for each head count |
| Per-head results stored in result.json | Must be queryable per head for comparison reports | Medium | Extend schema: `svd_metrics` keys gain head dimension, e.g., `qkt.layer_0.head_0.grassmannian_distance` |

**Standard approach:** Standard multi-head attention splits the d_model dimension into n_heads sub-dimensions of size d_model / n_heads. Each head operates independently on its subspace. For SVD analysis, each head's QK^T is a T x T matrix in the head's d_k = d_model/n_heads dimensional subspace.

Recent mechanistic interpretability work (Elhage et al., 2021; Basile et al., 2025) shows that individual heads specialize for different semantic functions. The SVD of per-head QK^T matrices reveals whether the predictive signal concentrates in specific heads. The OV circuit per head (WvWo per head) decomposes into independent low-rank structures via SVD (recent work on "singular vector-based interpretability" by Li et al., 2025).

**Critical implementation detail:** With n_heads > 1, d_k = d_model / n_heads. For the anchor config (d_model=128), 2 heads gives d_k=64 and 4 heads gives d_k=32. The QK^T matrix is still T x T, but the effective rank is constrained by d_k. This means the SVD spectrum will have at most d_k non-zero singular values, which changes the interpretation of metrics like stable rank and spectral entropy.

**Existing infrastructure dependencies:**
- `model/attention.py` -- rewrite to support n_heads parameter (biggest change)
- `model/transformer.py` -- pass n_heads through
- `model/types.py` -- update `AttentionInternals` and `ForwardOutput` shapes
- `config/experiment.py` -- relax `n_heads == 1` constraint
- `evaluation/pipeline.py` -- iterate over heads in SVD loop
- `analysis/auroc_horizon.py` -- per-head AUROC curves
- All visualization modules -- per-head variants

**Confidence:** HIGH for the approach (standard multi-head + per-head analysis). MEDIUM for implementation complexity -- this touches nearly every module in the pipeline.

---

### 4. Precision-Recall Curves and Calibration (Reliability Diagrams)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Precision-recall curve at each lookback j | Reviewer concern: AUROC can be misleading for imbalanced classes (violations are rare events) | Medium | Use sklearn or manual implementation. Violations are the positive class |
| AUPRC (area under PR curve) per metric per lookback | Summary statistic complementing AUROC | Low | `sklearn.metrics.average_precision_score` or trapezoidal integration |
| Reliability diagram (calibration curve) | Shows whether predicted probabilities match observed frequencies | Medium | Bin predictions into 10 deciles, plot fraction positive vs mean predicted probability per bin |
| Expected Calibration Error (ECE) | Single-number calibration summary | Low | Weighted average of |accuracy - confidence| per bin |
| PR and calibration figures integrated into HTML reports | Must appear alongside existing AUROC plots | Low | New visualization functions, embed in existing report templates |

**Standard approach:** AUROC measures discrimination (can the metric separate violations from controls?) but not calibration (do high metric values reliably correspond to high violation probability?). For rare events like rule violations, precision-recall curves are more informative than ROC curves because they are sensitive to the positive class base rate (Saito & Rehmsmeier, PLOS ONE 2015; recent PMC article 2025 on AUPRC for rare critical events).

The calibration approach for this project requires framing the SVD metric as a predictor:
1. At each lookback j, threshold the SVD metric value to produce a binary prediction
2. Sweep thresholds to generate the PR curve
3. For calibration: bin the metric values, compute fraction of actual violations per bin

**Important nuance:** The existing AUROC analysis uses raw metric values and the rank-based method (equivalent to Mann-Whitney U). PR curves require the same data but compute precision = TP/(TP+FP) and recall = TP/(TP+FN) at each threshold. This is a different view of the same data, not new data collection.

**Existing infrastructure dependencies:**
- `analysis/auroc_horizon.py` -- extend to compute PR curves alongside AUROC
- `analysis/statistical_controls.py` -- add calibration metrics
- `visualization/auroc.py` -- add PR curve and reliability diagram plot functions
- `reporting/single.py` -- add PR and calibration sections to template
- `reporting/math_pdf.py` -- add ECE formula to math sections

**Confidence:** HIGH -- standard techniques, well-implemented in scikit-learn, direct extension of existing AUROC infrastructure.

---

### 5. Pre-Registration Framework

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Formal hypothesis specification document | Distinguishes confirmatory from exploratory analysis. Reviewers want to know Grassmannian distance was chosen a priori, not cherry-picked | Low | Markdown document specifying: primary hypothesis, primary metric (Grassmannian distance of QK^T), secondary metrics, alpha level, correction method |
| Held-out evaluation protocol | Data splitting: some walks reserved for confirmatory test, others for exploration | Medium | Split eval walks into exploratory (60%) and confirmatory (40%). Run exploratory first, lock analysis plan, then run confirmatory |
| Analysis plan lock mechanism | Prevent p-hacking by committing to analysis plan before seeing confirmatory results | Low | Git commit the analysis plan before running confirmatory analysis. Timestamp serves as proof |
| Deviation log | Any changes to pre-registered plan must be documented with rationale | Low | Markdown file tracking any deviations from the pre-registered protocol |
| Pre-registration metadata in result.json | Results must indicate whether they are exploratory or confirmatory | Low | Add `pre_registration` block to result.json with status field |

**Standard approach:** Pre-registration separates hypothesis-generating (exploratory) from hypothesis-testing (confirmatory) research (Nosek et al., 2018). The NeurIPS pre-registration workshop (2020, 2021) established templates for ML experiments. The key elements are:

1. **Primary hypothesis:** Grassmannian distance of QK^T is elevated before rule violations
2. **Primary test:** AUROC > 0.75 at lookback j >= 2, with Holm-Bonferroni correction across 5 primary metrics
3. **Held-out protocol:** Evaluate on 40% of eval walks not used during exploratory analysis
4. **Decision criterion:** Reject null if corrected p-value < 0.05 on held-out data

The existing `PRIMARY_METRICS` frozenset in `auroc_horizon.py` already captures the pre-registered metric set. The framework needs to formalize the held-out split and enforce the temporal ordering (explore, lock, confirm).

**Existing infrastructure dependencies:**
- `walk/corpus.py` -- add exploratory/confirmatory split
- `analysis/auroc_horizon.py` -- already has PRIMARY_METRICS; add held-out mode
- `analysis/statistical_controls.py` -- Holm-Bonferroni already implemented
- `results/schema.py` -- extend validation for pre_registration block
- New file: `.planning/pre_registration.md` or `docs/pre_registration.md`

**Confidence:** HIGH -- pre-registration is a process, not a complex algorithm. The existing Holm-Bonferroni and PRIMARY_METRICS infrastructure already supports the core mechanism.

---

## Differentiators

Features that elevate the paper from "addresses reviewer concerns" to "thorough and impressive." Not required for acceptance but strengthen the manuscript.

### 6. Sharp Compliance Curve (r/w Sweep Showing Phase Transition)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| r/w sweep with fine granularity | Shows the phase transition from near-perfect compliance (r << w) to failure (r >> w) | Medium | Sweep r across [0.3w, 0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w]. Already partially specified in jumper r-scale values |
| Compliance rate vs r/w ratio curve | The "step function" that demonstrates the context window boundary effect | Low | Plot rule_compliance rate as function of r/w. Should show sigmoid-like transition |
| Predictive horizon vs r/w overlay | Shows that predictive horizon is longest just before the compliance cliff | Low | Overlay predictive horizon (from existing AUROC analysis) on the compliance curve |
| Visualization as publication figure | This is a "money figure" for the paper | Low | Clean matplotlib with dual y-axis (compliance rate + predictive horizon) |

**Standard approach:** The compliance curve should show a sharp transition: when r << w (rule deadline well within context window), the transformer easily learns the rule. When r >> w (deadline beyond context window), the model cannot retain the jumper encounter. The transition region around r ~ w is where the predictive signal should be strongest -- the model is "trying" to comply but failing, creating the spectral instability that precedes violation.

This is analogous to phase transitions in statistical mechanics: the system exhibits critical behavior at the transition point. The sharpness of the transition (how steep the compliance drop is) depends on model capacity and training quality.

**Existing infrastructure dependencies:**
- `config/experiment.py` -- already supports varying r via training.r
- `graph/jumpers.py` -- already computes r from scale factors
- `evaluation/behavioral.py` -- compliance rate is already computed
- `analysis/auroc_horizon.py` -- predictive horizon already computed per r
- New visualization function for the composite curve

**Confidence:** HIGH -- this is primarily a sweep + visualization feature. The infrastructure exists; the main work is running multiple configs and producing the composite figure.

---

### 7. Full Spectrum Trajectory Tracking with Curvature and Torsion

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Store full singular value vector sigma_1...sigma_k at each step | Goes beyond scalar summaries to track the complete spectral shape over time | Medium | Currently only scalars (stable_rank, entropy, etc.) are stored. Need to store the full sigma vector per step |
| Spectral curve as trajectory in R^k | Treat the sequence of sigma vectors as a curve in R^k and analyze its geometry | Medium | Each step maps to a point in R^k; the sequence is a discrete curve |
| Curvature (discrete Frenet-Serret) | Measures how fast the spectral curve is "turning" -- sudden curvature = spectral regime change | High | Discrete approximation: curvature at step t = |T(t+1) - T(t)| / |ds| where T is the unit tangent vector |
| Torsion (3D+ generalized) | Measures "twisting" of the spectral trajectory out of a plane -- detects spectral transitions that are invisible to curvature alone | High | Requires computing the binormal vector from consecutive tangent and normal vectors. Numerically delicate for discrete data |
| Curvature/torsion spikes as predictive features | Do curvature/torsion spikes precede violations better than individual metric summaries? | Medium | Feed curvature/torsion into the AUROC analysis pipeline as additional metrics |

**Standard approach:** The Frenet-Serret apparatus generalizes from R^3 to R^k. For a curve gamma(t) in R^k, the generalized curvatures kappa_1, ..., kappa_{k-1} are defined via the Gram-Schmidt process on consecutive derivatives. Recent work by Chassat et al. (ICCV 2023, with 2025 extension) provides a shape analysis framework for Euclidean curves under the Frenet-Serret framework, showing that curvature and torsion capture geometric features invisible to simpler representations.

The connection to SVD: if the singular value vector sigma(t) at step t is treated as a point in R^k, then:
- **Curvature kappa_1** measures how fast the singular value distribution is changing direction (a spectral "turning point")
- **Torsion kappa_2** measures whether the change is planar (torsion=0) or genuinely three-dimensional (the spectrum is evolving in a complex, non-planar way)

**Implementation detail:** For discrete data, use finite differences:
1. Tangent: T(t) = (sigma(t+1) - sigma(t)) / ||sigma(t+1) - sigma(t)||
2. Curvature: kappa(t) = ||T(t+1) - T(t)|| / ||sigma(t+1) - sigma(t)||
3. Torsion requires the binormal vector from the Gram-Schmidt process on T, dT/ds, d^2T/ds^2

**Existing infrastructure dependencies:**
- `evaluation/pipeline.py` -- store full sigma vectors (currently only scalars extracted)
- `svd_metrics.py` -- new functions for discrete curvature and torsion
- `analysis/auroc_horizon.py` -- include curvature/torsion as additional metrics
- `visualization/` -- new trajectory and curvature plots
- NPZ storage -- sigma vectors are arrays, not scalars; increases storage

**Confidence:** MEDIUM -- the math is well-established, but discrete Frenet-Serret is numerically delicate. Torsion computation for noisy discrete curves requires careful smoothing. The scientific payoff is uncertain: curvature/torsion spikes may or may not be better predictors than existing scalar metrics.

---

### 8. SVD Computational Overhead Analysis

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Wall-clock timing of SVD computation per step | Quantifies the practical cost of the analysis pipeline | Low | `torch.cuda.Event` timing around `torch.linalg.svd` calls |
| Breakdown by SVD target (QK^T vs WvWo vs AVWo) | Which targets dominate the cost? WvWo is static (computed once), others are per-step | Low | Separate timers per target |
| Scaling analysis: timing vs matrix dimension (T x T, T x D) | How does cost scale with context window w? | Low | Run with w = 32, 64, 128, 256 and measure |
| Randomized SVD comparison: `torch.svd_lowrank` vs `torch.linalg.svd` | Can we get acceptable metrics from a cheaper approximation? | Medium | `torch.svd_lowrank(A, q=k+5, niter=2)` from Halko et al. (2009). Compare metric values and AUROC results against full SVD |
| Approximation quality analysis | Are the Grassmannian distances from randomized SVD within acceptable tolerance of full SVD? | Medium | Compute metrics from both full and randomized SVD on same data, measure relative error |
| Cost summary table in report | Reviewers want to know the practical overhead of SVD monitoring | Low | Table with: matrix size, time per step, total time per experiment, % of total evaluation time |

**Standard approach:** Full SVD of a T x T matrix costs O(T^3) via LAPACK/cuSOLVER. For the anchor config (T = w = 64), this is modest. For larger context windows, randomized SVD (`torch.svd_lowrank` based on Halko et al., 2009, Algorithm 5.1) provides O(T^2 * k) cost where k is the target rank.

**Critical finding from research:** PyTorch's `torch.svd_lowrank` has known performance issues with batched inputs (2x slower than loop on some GPU configurations per PyTorch issue #119336). For dense matrices of the sizes in this project (64x64 to 256x256), full SVD via `torch.linalg.svd` is likely faster than randomized SVD because the overhead of the random projection dominates at small matrix sizes. The crossover where randomized SVD wins is typically around T > 512 for target rank k ~ 10.

Recommendation: benchmark both approaches on the actual matrix sizes used in the project, report the crossover point, and use full SVD for the published results (accuracy) while noting the randomized alternative for larger-scale future work.

**Existing infrastructure dependencies:**
- `evaluation/pipeline.py` -- add timing instrumentation
- New module: `analysis/svd_overhead.py` or section in evaluation pipeline
- `reporting/single.py` -- add overhead table to report template
- No changes to core SVD computation logic (benchmarking is additive)

**Confidence:** HIGH -- benchmarking is straightforward. The PyTorch APIs exist and are well-documented.

---

## Anti-Features

Features to explicitly NOT build for v1.1. Building these would be scope creep that delays addressing reviewer feedback.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Multi-head beyond 4 heads | Diminishing returns; 1h/2h/4h demonstrates the principle. 8h or 16h requires larger d_model to maintain meaningful d_k per head | Document as future work. State that 4h with d_k=32 is the smallest meaningful head dimension for 128-dim model |
| Full Bayesian calibration (Platt scaling, isotonic regression) | The AUROC-based predictor is not a deployed classifier -- it is a research measurement. Post-hoc calibration methods (Platt, isotonic) require fitting additional parameters, introducing degrees of freedom that undermine the "pre-registered" claim | Use raw reliability diagrams only. Show calibration properties of the raw metric, do not fit a calibration model |
| Real-time SVD monitoring system | v1.1 is about offline analysis for a paper, not deploying a monitoring tool | Store timing benchmarks and note the feasibility in the discussion section |
| Automated pre-registration platform (OSF integration) | Over-engineering the process. A git commit + markdown document is sufficient for a single-lab study | Use git timestamps as pre-registration proof. Commit the analysis plan before running confirmatory analysis |
| Symbolic math verification (SymPy) of the softmax bound | The derivation is a manual mathematical proof, not a symbolic computation. Automating it with SymPy would take longer than writing the LaTeX directly and adds no scientific value | Write the LaTeX derivation by hand. Verify with numerical examples in Python |
| General-purpose spectral trajectory analysis library | The Frenet-Serret computation is specific to this project's spectral curves. Generalizing to a library is premature | Implement curvature/torsion as functions in svd_metrics.py, not a separate package |
| Gradient-based attribution for SVD metrics | Would answer "why does QK^T change before violations?" but this is a different research question than "does it change?" | Acknowledge as future work. The v1.1 scope is establishing that the signal exists, not explaining its mechanism |

---

## Feature Dependencies

```
v1.0 Infrastructure (existing, unchanged)
  |
  +-- [1] Null Model Baseline
  |     Requires: walk/generator.py (no-jumper mode)
  |     Requires: evaluation/pipeline.py (unchanged)
  |     Requires: svd_metrics.py (unchanged)
  |     New: null comparison in auroc_horizon.py
  |     New: overlay in event_aligned plots
  |     New: Marchenko-Pastur reference computation
  |
  +-- [2] Softmax Filtering Bound
  |     Requires: model/attention.py (add Jacobian extraction hook)
  |     Requires: reporting/math_pdf.py (add derivation section)
  |     New: perturbation injection mode in eval pipeline
  |     New: bound tightness visualization
  |
  +-- [3] Multi-Head Ablation  *** LARGEST CHANGE ***
  |     Modifies: model/attention.py (multi-head support)
  |     Modifies: model/transformer.py (pass n_heads)
  |     Modifies: model/types.py (shape changes)
  |     Modifies: config/experiment.py (relax n_heads constraint)
  |     Modifies: evaluation/pipeline.py (per-head SVD loop)
  |     Modifies: analysis/auroc_horizon.py (per-head AUROC)
  |     Modifies: all visualization modules (per-head variants)
  |     New: signal concentration analysis
  |
  +-- [4] Precision-Recall + Calibration
  |     Requires: analysis/auroc_horizon.py (extend, not replace)
  |     New: PR curve computation alongside AUROC
  |     New: calibration/reliability diagram functions
  |     New: visualization functions for PR and calibration
  |     Extends: reporting templates
  |
  +-- [5] Pre-Registration Framework
  |     Requires: walk/corpus.py (exploratory/confirmatory split)
  |     Requires: analysis/auroc_horizon.py (PRIMARY_METRICS already set)
  |     Requires: statistical_controls.py (Holm-Bonferroni exists)
  |     New: held-out evaluation protocol
  |     New: pre_registration.md document
  |     Extends: result.json schema
  |
  +-- [6] Sharp Compliance Curve
  |     Requires: multiple r-value configs (existing sweep infrastructure)
  |     Requires: behavioral evaluation (existing)
  |     New: composite compliance + horizon visualization
  |
  +-- [7] Full Spectrum Trajectory + Curvature/Torsion
  |     Modifies: evaluation/pipeline.py (store full sigma vectors)
  |     New: discrete Frenet-Serret functions in svd_metrics.py
  |     New: trajectory + curvature visualization
  |     Extends: NPZ storage format (arrays instead of scalars)
  |     Feeds into: auroc_horizon.py (curvature as additional metric)
  |
  +-- [8] SVD Computational Overhead
        Requires: evaluation/pipeline.py (add timing)
        New: timing instrumentation
        New: randomized SVD comparison benchmarks
        New: overhead table in report template
```

**Critical path:** Feature [3] (multi-head ablation) is the largest and most invasive change. It should be built first or with careful interface design so that other features (especially [1], [4], [5]) can work with both 1h and multi-head models.

**Independence:** Features [1], [4], [5], [6], [8] are largely independent and can be developed in parallel. Feature [7] depends on modified NPZ storage but not on other v1.1 features. Feature [2] is mathematically independent but touches the attention module.

---

## Complexity Assessment

| Feature | Code Complexity | Math Complexity | Infrastructure Impact | Overall |
|---------|-----------------|------------------|-----------------------|---------|
| [1] Null Model Baseline | Low | Medium (MP distribution) | Low (additive) | **Low-Medium** |
| [2] Softmax Filtering Bound | Medium | High (novel derivation) | Low (additive) | **High** |
| [3] Multi-Head Ablation | High | Low | High (modifies core model) | **High** |
| [4] PR + Calibration | Low | Low | Low (additive) | **Low** |
| [5] Pre-Registration | Low | Low | Low (mostly process) | **Low** |
| [6] Compliance Curve | Low | Low | Low (sweep + viz) | **Low** |
| [7] Spectrum Trajectory | Medium | High (Frenet-Serret) | Medium (storage format) | **Medium-High** |
| [8] SVD Overhead | Low | Low | Low (additive) | **Low** |

---

## MVP Recommendation

### Priority 1: Core Reviewer Concerns (must address for resubmission)

1. **[1] Null Model Baseline** -- All 3 reviewers asked for this. Fastest to implement, highest review impact.
2. **[4] Precision-Recall + Calibration** -- Standard practice for rare-event classifiers. Quick addition to existing infrastructure.
3. **[5] Pre-Registration Framework** -- Process + documentation, not complex code. Shows methodological rigor.
4. **[3] Multi-Head Ablation** -- Largest effort but directly addresses "is the signal real?" question. Build 2h first, then 4h.

### Priority 2: Mathematical Depth (strengthens the paper)

5. **[2] Softmax Filtering Bound** -- The mathematical derivation is the hardest part. Start the LaTeX early, implement empirical verification in parallel.
6. **[6] Sharp Compliance Curve** -- Requires running the r/w sweep. Can be done in parallel with implementation work.

### Priority 3: Advanced Analysis (elevates if time permits)

7. **[7] Full Spectrum Trajectory** -- Novel analysis. Scientific payoff uncertain.
8. **[8] SVD Computational Overhead** -- Useful for the paper's methods section but not a reviewer concern.

### Defer

- **Curvature/torsion as predictive features** (part of [7]): Compute and report, but do not stake claims on it unless the signal is clearly superior to existing metrics. This is exploratory, not confirmatory.

---

## Expected Output Formats and Integration

| Feature | Output Format | Integration Point |
|---------|---------------|-------------------|
| [1] Null Model | `null_model` block in result.json; null overlay on event-aligned plots; MP histogram figure | HTML report new section; math PDF MP derivation |
| [2] Softmax Bound | LaTeX derivation in math PDF; `softmax_bound` block in result.json; bound tightness figure | Math PDF new section; HTML report new section |
| [3] Multi-Head | Per-head metric arrays in NPZ; per-head AUROC in result.json; signal concentration heatmap | HTML report head-comparison section; comparison report across head counts |
| [4] PR + Calibration | PR curves + AUPRC in result.json; reliability diagram figures; ECE scalar | HTML report alongside AUROC section |
| [5] Pre-Registration | pre_registration.md document; `pre_registration` block in result.json; exploratory/confirmatory tags | Report section showing pre-reg compliance |
| [6] Compliance Curve | Composite figure (compliance + horizon vs r/w); sweep results | HTML comparison report; publication figure |
| [7] Spectrum Trajectory | Full sigma vectors in NPZ; curvature/torsion time series; trajectory figures | HTML report new section |
| [8] SVD Overhead | Timing data in result.json; overhead table; randomized SVD comparison | HTML report methods section |

---

## Sources

**Null model / random matrix baseline:**
- [Marchenko-Pastur distribution](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution) -- foundational null for random matrix singular value distributions
- [Small Singular Values Matter: A Random Matrix Analysis of Transformer Models](https://arxiv.org/abs/2410.17770) -- NeurIPS 2025, demonstrates MP as null model for transformer weight matrices
- [Uncovering functional signature in neural systems via random matrix theory](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006934) -- RMT baseline approach

**Softmax perturbation bounds:**
- [The Lipschitz Constant of Self-Attention](http://proceedings.mlr.press/v139/kim21i/kim21i.pdf) -- Kim et al., ICML 2021, foundational Lipschitz analysis
- [Pay Attention to Attention Distribution: A New Local Lipschitz Bound for Transformers](https://arxiv.org/abs/2507.07814) -- Li et al., 2025, tighter local bounds
- [Softmax is 1/2-Lipschitz](https://www.lakernewhouse.com/assets/writing/softmax-is-0-5-lipschitz.pdf) -- direct softmax Jacobian spectral norm result

**Multi-head SVD analysis:**
- [Interpreting Transformers Through Attention Head Intervention](https://arxiv.org/abs/2601.04398) -- Basile et al., 2025/2026, per-head specialization
- [Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](https://arxiv.org/abs/2511.20273) -- SVD decomposition within heads
- [Attention-Head OV Circuits in Transformers](https://www.emergentmind.com/topics/attention-head-ov-circuits) -- overview of per-head OV circuit analysis

**Precision-recall and calibration:**
- [Use of the Area Under the Precision-Recall Curve to Evaluate Prediction Models of Rare Critical Illness Events](https://pmc.ncbi.nlm.nih.gov/articles/PMC12133047/) -- 2025, AUPRC for rare events
- [The Precision-Recall Plot Is More Informative than the ROC Plot](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- Saito & Rehmsmeier, foundational PR vs ROC comparison
- [scikit-learn calibration documentation](https://scikit-learn.org/stable/modules/calibration.html) -- implementation reference

**Pre-registration:**
- [NeurIPS Pre-Registration Workshop](https://preregister.science) -- ML-specific pre-registration templates
- [Pre-registration for Predictive Modeling](https://arxiv.org/abs/2311.18807) -- framework for ML pre-registration
- [Introducing the Simulation Studies Preregistration Template](https://www.cos.io/blog/introducing-the-simulation-studies-preregistration-template) -- simulation study template

**Frenet-Serret / spectral trajectory:**
- [Shape Analysis of Euclidean Curves under Frenet-Serret Framework](https://arxiv.org/abs/2511.17065) -- Chassat et al., 2023/2025
- [Frenet-Serret and the Estimation of Curvature and Torsion](https://ieeexplore.ieee.org/document/6377229/) -- discrete estimation methods
- [Frenet-Serret formulas (Wikipedia)](https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas) -- mathematical reference

**SVD computational cost:**
- [torch.svd_lowrank documentation](https://docs.pytorch.org/docs/stable/generated/torch.svd_lowrank.html) -- API: `torch.svd_lowrank(A, q=6, niter=2)`
- [torch.linalg.svd documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html) -- full SVD API
- [PyTorch batched SVD performance issue](https://discuss.pytorch.org/t/batched-svd-lowrank-being-much-slower-than-loop-implementation-both-cpu-and-gpu/119336) -- known performance caveats
- [Grassmann Manifold Handbook](https://link.springer.com/article/10.1007/s10444-023-10090-8) -- computational Grassmannian methods

**Confidence:** HIGH for features [1], [3], [4], [5], [6], [8] (standard techniques, well-documented). MEDIUM for [2] (novel derivation, bound tightness unknown). MEDIUM for [7] (numerically delicate, scientific payoff uncertain).
