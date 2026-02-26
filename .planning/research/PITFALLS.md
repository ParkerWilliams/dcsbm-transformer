# Domain Pitfalls: v1.1 Journal Feedback Features

**Domain:** Adding null model baselines, softmax filtering bounds, multi-head ablation, PR curves + calibration, pre-registration framework, sharp compliance curves, full spectrum trajectory tracking, and SVD computational overhead analysis to an existing DCSBM transformer research framework.
**Researched:** 2026-02-26
**Confidence:** HIGH (verified against existing codebase, established numerical methods, recent literature on softmax Lipschitz bounds and attention spectral analysis)

**Scope:** Pitfalls specific to adding v1.1 features to the existing v1.0 framework. For foundational pitfalls (graph generation, training, basic SVD numerics), see the v1.0 PITFALLS.md.

---

## Critical Pitfalls

Mistakes that invalidate reviewer claims, require architectural rewrites, or break existing v1.0 functionality.

---

### Pitfall 1: Null Model Confounded With Signal -- Wrong Baseline Invalidates "Real Signal" Claim

**What goes wrong:** The mathematician reviewer's concern is that Grassmannian distance "conflates genuine instability with natural input variation." The null model must measure the *natural* Grassmannian drift that occurs on clean (no-jumper) sequences. Three common failures:

1. **Matched design failure:** Generating no-jumper sequences using a different graph (e.g., removing jumper vertices entirely from the DCSBM). This changes the graph topology, degree distribution, and walk statistics. The null model measures a different system, not the same system without the signal.

2. **Unmatched temporal structure:** Computing baseline Grassmannian drift as a single aggregate statistic (mean drift across all positions) rather than position-matched. The existing `grassmannian_distance` metric has inherent position dependence -- drift at step w+1 differs from drift at step 3w because the attention pattern evolves with context length. If the null model uses a grand mean while the signal model uses per-position values, the comparison is invalid.

3. **Training state confound:** Using the same trained model that learned jumper rules to evaluate null sequences. The model's QK^T patterns have been shaped by jumper-containing training data. A properly learned model may produce *different* Grassmannian dynamics on clean sequences not because of "natural variation" but because it is actively not-detecting the absence of jumpers. The null model would then be measuring the model's response to non-jumper sequences, not baseline drift.

**Why it happens in this codebase:** The existing `fused_evaluate` in `src/evaluation/pipeline.py` takes a fixed model and generates from it. There is no mechanism to generate evaluation walks that are guaranteed jumper-free. The `GraphConfig.n_jumpers_per_block=2` creates jumper vertices at the graph level -- you cannot simply filter them from walks because the model was trained on a graph that includes them. Additionally, `extract_events` in `src/analysis/event_extraction.py` identifies jumper encounters by checking tokens against `jumper_map`; it does not track "clean regions" between jumper encounters.

**Consequences:** If the null model is confounded, reviewers reject the revision: "You showed the signal differs from a bad null, not from natural variation."

**Warning signs:**
- Null model Grassmannian drift distribution has significantly different variance from the signal model (suggests different systems)
- Position-dependent drift plots for null and signal have different shapes (not just different means)
- Null model AUROC against shuffled labels is not centered at 0.5

**Prevention:**
1. **Same graph, same model, different walks.** Generate evaluation walks that start from non-jumper vertices and are guaranteed to never encounter a jumper vertex within the analysis window. Use the existing `jumper_map` to check all tokens in each walk candidate; reject walks with jumper encounters. This tests the *same system* without the *specific event*.
2. **Position-matched null.** For every jumper encounter at position p in a signal walk, extract the corresponding Grassmannian drift at position p from the closest clean walk. The existing `compute_auroc_curve` function at `src/analysis/auroc_horizon.py:57` already does position-indexed extraction -- extend this pattern to null-model computation.
3. **Compute and report the null distribution fully:** mean, std, percentiles (5th, 25th, 50th, 75th, 95th) of Grassmannian drift at each position, and overlay it on the signal distribution. The reviewer needs to see that the signal *exceeds* the null envelope, not just that signal != 0.
4. **Effect size over the null:** Report Cohen's d of signal vs. null at each lookback distance. The existing `cohens_d` function in `src/analysis/statistical_controls.py:146` can be reused directly.

**Detection:** Run null model analysis before any signal analysis. If null Grassmannian drift > 0 (which it always will be -- subspaces naturally drift), verify that the signal drift is distinguishable from null drift with p < 0.01 at the lookback distances where AUROC > 0.75.

**Phase:** This should be the FIRST phase of v1.1. All other analyses depend on establishing the signal is real.

---

### Pitfall 2: Multi-Head Ablation Breaks SVD Extraction Pipeline -- Shape and Semantics Change

**What goes wrong:** The existing codebase is hardcoded for single-head attention at multiple levels. Adding 2-head and 4-head variants requires changes that can silently break SVD metric semantics:

1. **Attention matrix shape change.** The current `CausalSelfAttention` in `src/model/attention.py` produces QK^T of shape `[B, T, T]` -- a single T x T matrix per layer. With h heads, each head produces a `[B, T/h, T]` or `[B, T, T]` matrix (depending on whether you split the sequence or the dimension). Actually, multi-head attention splits the *model dimension*: each head uses d_k = d_model/h, producing QK^T of shape `[B, h, T, T]`. The SVD of a T x T matrix is fundamentally different from the SVD of the concatenated or stacked per-head matrices. The existing `compute_all_metrics` in `src/evaluation/svd_metrics.py` expects a single matrix; it will either crash or silently compute wrong metrics on multi-head output.

2. **Head-size vs. model-size confusion.** With d_model=128 and h=4, each head has d_k=32. The QK^T matrix is still T x T (context window squared) regardless of d_k, but the *rank* of QK^T changes: rank(QK^T) <= min(T, d_k) = min(64, 32) = 32 for h=4 vs. 64 for h=1. This means spectral gap metrics, stable rank, and condition number have fundamentally different interpretations. A spectral gap of 0.5 in a rank-32 matrix means something very different from a spectral gap of 0.5 in a rank-64 matrix.

3. **Per-head vs. concatenated analysis ambiguity.** The reviewer wants to know if "the signal generalizes" beyond single-head. Two valid analyses exist: (a) per-head SVD, checking if any head carries the signal, and (b) concatenated-head SVD, checking if the signal survives head combination. The codebase currently has no infrastructure for either. If you compute per-head SVD, you need to track which head number to associate with which metric. If you compute concatenated SVD, you need to decide whether to concatenate along the batch dimension (wrong -- treats heads as independent samples) or compute SVD on the combined output.

4. **WvWo product changes.** The existing `get_wvwo` in `src/model/transformer.py:170` computes `W_v.weight.T @ W_o.weight` assuming both are `[d_model, d_model]`. With multi-head, W_v and W_o are either per-head `[d_k, d_model]` matrices or a single `[d_model, d_model]` matrix that is split. The WvWo product semantics change entirely.

**Why it happens in this codebase:** The `ModelConfig` in `src/config/experiment.py` has `n_heads: int = 1` with a validation guard `if self.model.n_heads != 1: raise ValueError("n_heads must be exactly 1")`. This guard must be relaxed, but doing so will propagate through the entire pipeline. The `AttentionInternals` dataclass in `src/model/types.py` has `qkt: torch.Tensor  # [B, T, T]` -- no head dimension. The `ForwardOutput` has `qkt: torch.Tensor | None = None  # [B, n_layers, T, T]` -- also no head dimension.

**Consequences:** (a) Metrics computed on multi-head models are not comparable to single-head metrics, invalidating the ablation. (b) Existing tests that check tensor shapes will silently pass if the head dimension is squeezed or averaged out, hiding the semantic error. (c) The reviewer specifically asked for "per-head SVD signal concentration" -- averaging or concatenating defeats the purpose.

**Warning signs:**
- Multi-head QK^T metrics have systematically different ranges than single-head metrics
- Per-head Grassmannian distances are identical across heads (suggests heads are seeing the same matrix, not per-head matrices)
- Stable rank of multi-head QK^T is bounded by d_k but reported as if bounded by d_model

**Prevention:**
1. **Add head dimension to tensor contracts.** Modify `AttentionInternals.qkt` to `[B, H, T, T]` and `ForwardOutput.qkt` to `[B, n_layers, H, T, T]`. This is a breaking change -- every consumer of these tensors must be updated. Mark the migration clearly.
2. **Per-head SVD extraction.** The `fused_evaluate` loop at `src/evaluation/pipeline.py:249` must iterate over heads within each layer. Metric keys become `qkt.layer_0.head_0.grassmannian_distance`. This is a naming convention change that propagates to `PRIMARY_METRICS` in `auroc_horizon.py`, all visualization code, and reporting templates.
3. **Report head-size context.** Every multi-head metric table must include d_k alongside d_model. When comparing 1h (d_k=128) vs. 4h (d_k=32), normalize stable rank and spectral gap by d_k.
4. **Signal concentration metric.** For the reviewer's question "does one head carry the signal?", compute variance of per-head AUROC across heads. If one head has AUROC=0.85 and others have AUROC~0.5, the signal is concentrated. If all heads have AUROC=0.65, it is distributed. This metric does not exist in the codebase and must be added.
5. **Do NOT modify `CausalSelfAttention`.** Instead, create a *new* `MultiHeadCausalSelfAttention` class that wraps multiple single-head instances. This preserves backward compatibility for single-head experiments and makes per-head extraction trivial. Avoid the standard PyTorch `nn.MultiheadAttention` because it fuses the head projection and does not expose per-head QK^T.

**Detection:** Unit test that verifies per-head QK^T has rank <= d_k. Integration test that runs 1h and 4h on the same input and verifies per-head metrics are not identical.

**Phase:** Multi-head ablation should be its own phase, depending on the null model phase. It touches the model, evaluation, analysis, and visualization layers simultaneously.

---

### Pitfall 3: Softmax Filtering Bound Derivation Errors -- Getting the Math Wrong

**What goes wrong:** The reviewer wants a theoretical bound: "given epsilon perturbation in QK^T, what is the resulting spectral change in AV W_o?" This is a chain of three mappings:

QK^T -> softmax -> A -> AV -> AVW_o

Common mathematical errors in deriving this bound:

1. **Using global Lipschitz constant of 1 for softmax.** Recent work (arXiv:2510.23012) proves the tight global Lipschitz constant of softmax is 1/2 (not 1). Using the old constant of 1 gives a bound that is 2x looser than necessary. Reviewers who know the current literature will flag this.

2. **Ignoring the 1/sqrt(d_k) scaling.** The QK^T in the codebase at `src/model/attention.py:71` is `(q @ k.transpose(-2, -1)) * scale` where `scale = 1.0 / math.sqrt(self.d_model)`. The perturbation epsilon in the scaled QK^T is epsilon/sqrt(d_k) in the unscaled product. Forgetting this scaling changes the bound by a factor of sqrt(d_k) = sqrt(128) = 11.3x.

3. **Confusing row-wise and matrix-wise Lipschitz.** Softmax operates row-wise on QK^T. The Lipschitz constant 1/2 applies to each row independently. The matrix-wise Lipschitz constant of the softmax-over-rows operator depends on how you measure "matrix perturbation." Using Frobenius norm vs. spectral norm gives different bounds. The codebase uses spectral properties (SVD), so the spectral norm bound is needed, but this is harder to derive than the Frobenius bound.

4. **Temperature scaling not accounted for.** If the model is trained with temperature scaling (not currently in the codebase, but a reviewer may ask "what about temperature?"), the effective Lipschitz constant of softmax(x/T) is 1/(2T). The bound must generalize to arbitrary temperature, even if T=1 in the current experiments.

5. **Causal mask discontinuity.** The attention matrix A is causal-masked: A[i,j] = 0 for j > i. The softmax filtering bound applies to the unmasked rows, but the mask introduces a structural asymmetry. Row i of A has only i+1 nonzero entries. The Lipschitz behavior of softmax on a truncated input differs from full-row softmax. Specifically, the entropy of the attention distribution varies by row, and the local Lipschitz constant of softmax depends on entropy (arXiv:2507.07814).

6. **Chain rule composition.** The total bound through softmax -> AV -> AVW_o is:
   ||delta(AVW_o)|| <= ||W_o|| * ||V|| * L_softmax * ||delta(QK^T)||
   But this is a worst-case bound. If V is approximately low-rank (which it is after training), the effective amplification through AV is smaller. Claiming a tight bound without acknowledging this overestimation will draw criticism.

**Why it happens in this codebase:** The existing `src/evaluation/svd_metrics.py` computes metrics on QK^T and AVW_o separately -- there is no function that relates perturbations in one to changes in the other. The bound derivation is entirely new mathematical work with no existing code to validate against.

**Consequences:** An incorrect bound is worse than no bound. The mathematician reviewer will check the derivation carefully. A factor-of-2 error in the Lipschitz constant or a missing sqrt(d_k) can be caught in review and undermine the entire paper's credibility.

**Warning signs:**
- Empirical perturbation magnitudes consistently exceed the theoretical bound (bound is too tight -- derivation error)
- Bound grows with sequence length T when it should be length-independent (mixed up global and local Lipschitz)
- Bound gives a useful prediction for 1-head but fails for 4-head (forgot to account for head count)

**Prevention:**
1. **Use the correct softmax Lipschitz constant: 1/2.** Cite arXiv:2510.23012 explicitly. This is the tight bound across all lp norms.
2. **Derive per-row, then aggregate.** Compute the softmax perturbation bound per row of the attention matrix, then aggregate to a matrix norm bound. For row i of length i+1 (causal), the local Lipschitz constant is at most 1/2 regardless of row length.
3. **Include the scaling factor.** The bound should reference the scaled QK^T: epsilon_scaled = epsilon / sqrt(d_k). Then: ||delta(A)|| <= (1/2) * epsilon / sqrt(d_k).
4. **Empirical verification.** For 1000 random QK^T perturbations of magnitude epsilon, compute the actual ||delta(AVW_o)||_F and verify it lies below the theoretical bound. This is the "empirical verification" the reviewer requests. Code: perturb QK^T by epsilon * random_direction, recompute A, AV, AVW_o, measure change. The existing pipeline already computes all these quantities.
5. **Present as an inequality, not an equality.** The bound is a ceiling. State: "The theoretical bound predicts that a perturbation of epsilon in QK^T produces at most [formula] change in AVW_o spectral structure. Empirically, we observe [actual values] which are [X]% of the bound."
6. **LaTeX review by co-author.** The math verification PDF generator exists (`src/reporting/math_pdf.py`) -- use it to generate the derivation and have a mathematician co-author verify before submission.

**Detection:** The empirical verification catches derivation errors. If >5% of random perturbations exceed the bound, the derivation has an error.

**Phase:** Softmax bound phase should be independent of null model and multi-head phases. It is purely mathematical and can be developed and verified in parallel.

---

### Pitfall 4: Pre-Registration Invalidated by Data-Dependent Decisions

**What goes wrong:** The reviewer requests a pre-registration framework with "Grassmannian distance as primary hypothesis, held-out evaluation protocol." Pre-registration in computational experiments is invalidated by:

1. **Seeing results before locking the protocol.** If the team runs any exploratory analysis on the anchor config before writing the pre-registration document, the pre-registration is post-hoc. The existing codebase already has `PRIMARY_METRICS` hard-coded in `src/analysis/auroc_horizon.py:26` -- these choices may have been data-informed, not pre-registered.

2. **Test set leakage through hyperparameter tuning.** The existing pipeline trains and evaluates on walks from the same graph. If the sufficiency gate thresholds (edge >95%, rule >80%) were tuned by looking at evaluation results, the held-out protocol is compromised.

3. **Undocumented deviations.** Pre-registration requires documenting every deviation from the plan (Willroth & Atherton, 2024). If the analysis code changes metric definitions, filtering criteria, or AUROC thresholds after seeing initial results, these deviations must be explicitly reported. The existing `contamination_audit` in `filter_contaminated_events` is good practice -- but was the 0.3 flagging threshold pre-registered or tuned?

4. **Multiple comparison correction applied selectively.** The existing `holm_bonferroni` correction in `src/analysis/statistical_controls.py:39` is applied only to primary metrics. If the primary metric set changes after seeing results (adding a metric that "happens" to be significant), the correction is invalid.

5. **Held-out split implemented incorrectly.** The pre-registration calls for a held-out evaluation protocol. This means a subset of evaluation walks reserved entirely for confirmatory analysis, never touched during exploratory analysis. If the same walks are used for both exploratory and confirmatory analysis (even with different metrics), the held-out guarantee is broken.

**Why it happens in this codebase:** The `run_auroc_analysis` function at `src/analysis/auroc_horizon.py:245` analyzes ALL evaluation walks. There is no train/validation/held-out split within the evaluation set. The `PRIMARY_METRICS` frozenset is defined in code, not in a separate timestamped pre-registration document.

**Consequences:** A savvy reviewer will ask: "When was this pre-registration written? Before or after you saw the results?" If the answer is "we wrote the pre-registration framework as part of the v1.1 revision," the reviewer knows it is post-hoc and will discount it.

**Warning signs:**
- Pre-registration document timestamp is after any result analysis
- Primary metrics exactly match the top-performing metrics from exploratory analysis
- No held-out split clearly documented in the evaluation pipeline

**Prevention:**
1. **Write the pre-registration document NOW, before running v1.1 experiments.** Lock it with a git commit hash. The document should specify:
   - Primary hypothesis: "Grassmannian distance of QK^T at lookback j predicts rule violations with AUROC > 0.75"
   - Primary metric: `qkt.grassmannian_distance` (already in PRIMARY_METRICS)
   - Secondary metrics: the other 4 in PRIMARY_METRICS
   - Exploratory metrics: all others
   - Significance threshold: alpha=0.05 after Holm-Bonferroni correction
   - Held-out protocol: 50% of evaluation walks for exploratory, 50% for confirmatory
   - Deviation reporting: any change to the above must be documented with rationale
2. **Implement held-out split in code.** Add a `split` parameter to `fused_evaluate` or a post-hoc split of `EvaluationResult`. The exploratory split feeds `run_auroc_analysis` for initial analysis. The confirmatory split is analyzed ONLY ONCE, AFTER all analysis decisions are made on the exploratory split.
3. **Version the pre-registration.** Store it as a file in the repository (e.g., `.planning/pre-registration.md`) and reference the git commit hash. This is the computational equivalent of a pre-registration timestamp.
4. **Acknowledge the v1.0 context.** The v1.0 results informed the choice of Grassmannian distance as the primary metric. This is NOT pre-registration in the strict sense; it is "informed hypothesis testing." Frame it honestly: "Based on v1.0 exploratory analysis, we pre-register the following confirmatory analysis protocol for v1.1."

**Detection:** The pre-registration document must exist in the git history BEFORE any v1.1 results are computed. If it does not, the pre-registration claim is indefensible.

**Phase:** Pre-registration must be the very first deliverable of v1.1. It must be committed before ANY new analysis code runs.

---

### Pitfall 5: Breaking Existing v1.0 Functionality During v1.1 Integration

**What goes wrong:** The v1.1 features touch nearly every layer of the existing codebase:

| v1.1 Feature | v1.0 Modules Affected |
|---|---|
| Null model | `evaluation/pipeline.py`, `analysis/event_extraction.py`, `analysis/auroc_horizon.py` |
| Multi-head | `model/attention.py`, `model/transformer.py`, `model/types.py`, `config/experiment.py`, `evaluation/pipeline.py`, ALL analysis, ALL visualization |
| Softmax bound | `evaluation/svd_metrics.py` (new functions), `reporting/` |
| PR curves | `analysis/auroc_horizon.py`, `visualization/auroc.py` |
| Compliance curve | `training/evaluate.py`, `visualization/`, `reporting/` |
| Spectrum trajectory | `evaluation/pipeline.py`, `evaluation/svd_metrics.py` |
| SVD benchmarks | New module, but integrates with `evaluation/pipeline.py` |

The v1.0 audit identified critical integration breaks (stub `run_experiment.py`, `predictive_horizon` never written to result.json, NPZ key format conflict). Fixing these WHILE adding v1.1 features creates a two-front war: stabilizing v1.0 integration AND extending functionality.

Specific breakage risks:
- Relaxing the `n_heads != 1` validation in `ExperimentConfig.__post_init__` may allow invalid configurations to slip through other validation gates
- Adding head dimensions to `AttentionInternals` breaks all existing tests that check `qkt.shape == (B, T, T)`
- Adding new metric keys like `qkt.layer_0.head_0.grassmannian_distance` will fail existing tests that enumerate expected metric keys
- Changing the `PRIMARY_METRICS` set breaks existing Holm-Bonferroni correction (different number of comparisons)
- Adding PR curves alongside AUROC in `auroc_horizon.py` may accidentally change the AUROC computation if shared data structures are modified

**Warning signs:**
- Tests pass for new features but existing tests fail
- Single-head experiment results differ between v1.0 and v1.1 code (regression)
- Visualization code crashes on v1.0 result.json files

**Prevention:**
1. **Regression test suite first.** Before any v1.1 code changes, create a "v1.0 golden test" that runs the anchor config single-head pipeline and captures exact metric values. This test must pass after every v1.1 change. The existing test files in `tests/` cover unit tests; add an integration regression test.
2. **Feature flags, not modifications.** Where possible, add new code paths alongside existing ones rather than modifying existing functions. For example, add `compute_pr_curve` as a new function rather than modifying `compute_auroc_curve`. Add `MultiHeadCausalSelfAttention` as a new class rather than modifying `CausalSelfAttention`.
3. **Backward-compatible tensor contracts.** For single-head models, `ForwardOutput.qkt` should remain `[B, n_layers, T, T]` (H=1 is squeezed). Only for multi-head models should it be `[B, n_layers, H, T, T]`. Check `n_heads == 1` and handle both shapes in consumers.
4. **Separate NPZ namespaces.** v1.1 metrics (null model, per-head SVD, spectrum trajectories) should use a distinct prefix (e.g., `v11_null.qkt.layer_0.grassmannian_distance`) to avoid colliding with v1.0 metric keys.
5. **Fix v1.0 integration breaks first.** Address the P0 integration issues (stub `run_experiment.py`, `set_seed` never called, `predictive_horizon` never written) BEFORE adding v1.1 features. Adding features on a broken foundation doubles the debugging effort.

**Detection:** CI that runs the full existing test suite after every v1.1 commit. Any test failure blocks the commit.

**Phase:** Integration testing should be a standing concern throughout all v1.1 phases, not a separate phase. But fixing v1.0 P0 integration breaks should be the zeroth phase.

---

## Moderate Pitfalls

---

### Pitfall 6: PR Curves With Imbalanced Classes -- Misleading Precision at Low Recall

**What goes wrong:** Rule violations are rare events. In the existing pipeline, the class balance depends on the r/w ratio and the number of jumper vertices. For the anchor config (r=57, w=64), violations occur at ~5-15% of jumper encounters. PR curves are more informative than ROC under imbalance (which is why the reviewer requests them), but they introduce new pitfalls:

1. **Interpolation method matters.** Standard linear interpolation between PR curve points overestimates AUPRC. The existing sklearn convention uses step-function interpolation (precision is held constant until recall changes). If the codebase implements linear interpolation, AUPRC will be inflated.

2. **Baseline not reported.** The random classifier baseline on a PR curve is not 0.5 (as with ROC) -- it is the positive class prevalence: P/(P+N). With 10% violations, the PR baseline is 0.1. Reporting AUPRC=0.35 without noting the 0.1 baseline makes it look weak; reporting it as "3.5x above baseline" is the correct framing.

3. **Threshold dependence.** The existing AUROC analysis in `auroc_from_groups` (line 35 of `auroc_horizon.py`) is threshold-free: it computes the rank statistic directly. PR curves require choosing thresholds on the SVD metric values to compute precision/recall at each point. If the metric has discrete values or many ties, the PR curve becomes jagged and AUPRC unstable.

4. **Calibration mismatch.** The reviewer requests calibration alongside PR curves. Calibration measures whether the predicted probability of violation matches the observed frequency. But the current pipeline does not produce probabilities -- it produces raw SVD metric values. Converting metric values to calibrated probabilities requires isotonic regression or Platt scaling, which introduces another degree of freedom.

**Prevention:**
1. Use `sklearn.metrics.precision_recall_curve` and `sklearn.metrics.average_precision_score` which implement correct step-function interpolation.
2. Always report the baseline prevalence alongside AUPRC. Add a horizontal line at y=prevalence on every PR curve plot.
3. For calibration, use isotonic regression (`sklearn.isotonic.IsotonicRegression`) to map SVD metric values to violation probabilities, then compute calibration curves. Report calibration error (ECE) alongside the PR curve.
4. Report PR curves per-r-value (not aggregated), matching the existing stratification in `stratify_by_r`. Aggregating across r values mixes different imbalance ratios and is meaningless.

**Detection:** Verify that AUPRC > prevalence (otherwise the metric is worse than random). Verify that calibration error < 0.1 (otherwise the probability mapping is unreliable).

**Phase:** PR curves and calibration can be added alongside existing AUROC analysis without modifying it. This is an additive feature.

---

### Pitfall 7: Spectral Curve Curvature -- Numerical Differentiation Amplifies Noise

**What goes wrong:** The reviewer asks for "full spectrum trajectory tracking with curvature and torsion analysis." Curvature of a curve in R^k requires the second derivative of the trajectory. The spectrum trajectory sigma_1(t), sigma_2(t), ..., sigma_k(t) is a discrete time series sampled at each token step. Computing curvature involves:

1. **First derivative:** d_sigma/dt approximated by finite differences. This amplifies noise by a factor of 1/dt.
2. **Second derivative:** d^2_sigma/dt^2 approximated by second-order finite differences. This amplifies noise by 1/dt^2.
3. **Torsion:** Requires the third derivative, amplifying noise by 1/dt^3.

The singular values from `torch.linalg.svd` have inherent numerical noise at the level of ~1e-6 (float32 machine epsilon times the matrix norm). For a matrix with ||QK^T|| ~ 10, the noise floor is ~1e-5. With dt=1 (one token step), the second derivative noise is ~1e-5, which is comparable to the actual curvature signal for smooth trajectories.

Specific failures:
- **Gibbs phenomenon:** If the spectrum has a sharp transition (e.g., at a jumper encounter), finite differences produce oscillatory artifacts near the transition -- exactly where the interesting signal is.
- **Ordering instability:** SVD returns singular values in descending order. If sigma_2(t) and sigma_3(t) cross (swap order), the "sigma_2 trajectory" has a discontinuity that produces infinite curvature, which is a sorting artifact, not a physical signal.
- **Insufficient data points:** Curvature estimation requires at least 3 consecutive valid data points. If SVD metrics are NaN at any step (due to guard activation), the curvature is undefined in a neighborhood of that step.

**Why it happens in this codebase:** The existing `compute_all_metrics` in `src/evaluation/svd_metrics.py` does not store the full singular value vector -- it computes scalar metrics from S and discards S. To compute spectrum trajectories, S must be stored at every step, which is a new storage requirement: for T steps and k singular values, this is T * k floats per sequence per layer.

**Prevention:**
1. **Savitzky-Golay filter before differentiation.** Apply a smoothing filter to the spectrum trajectory before computing derivatives. Window length of 5-7 steps with polynomial order 2-3 is appropriate for this temporal resolution. This removes high-frequency noise while preserving the curvature signal.
2. **Store full singular value vectors.** Modify `fused_evaluate` to store S (not just scalar metrics) at each step. With d_model=128 and w=64, S has min(64,64)=64 elements. Storing the top-k (k=8) is sufficient for curvature analysis. This adds 8 floats per step per layer per sequence.
3. **Handle ordering crossings explicitly.** When tracking sigma_i(t), detect when sigma_i and sigma_{i+1} swap by checking if sigma_i(t) < sigma_{i+1}(t). At crossings, the curvature is undefined; mark these steps and interpolate through them. Alternatively, track the spectrum as an unordered set and compute distances between consecutive sets (Hausdorff distance on spectra).
4. **Use curvature of the Grassmannian trajectory instead.** Rather than computing curvature of individual singular values, compute the curvature of the subspace trajectory on the Grassmannian. The existing `grassmannian_distance` already measures the "velocity" of subspace change; curvature is the rate of change of this velocity. This is more physically meaningful and avoids the ordering problem.
5. **Report curvature statistics, not raw curvature.** Mean curvature over a lookback window is more stable than pointwise curvature. The reviewer is asking "does the spectrum trajectory bend sharply before failures?" -- this can be answered with windowed statistics.

**Detection:** Compute curvature on a synthetic constant spectrum (curvature should be 0). If numerical curvature > 1e-3 on this control, the differentiation is too noisy.

**Phase:** Spectrum trajectory tracking requires modifying `fused_evaluate` to store S vectors. This touches the evaluation pipeline and should be coordinated with the null model phase (which also modifies the pipeline).

---

### Pitfall 8: SVD Benchmarking -- GPU Timing Artifacts Make Results Unreliable

**What goes wrong:** The reviewer asks for "SVD computational overhead: timing benchmarks, cost analysis, cheaper approximation candidates." GPU benchmarking has well-known pitfalls:

1. **Cold cache effect.** The first SVD call on GPU is 10-100x slower than subsequent calls because CUDA must: (a) load the cuSOLVER library, (b) allocate workspace memory, (c) JIT-compile any PTX kernels, and (d) warm the L2 cache. If benchmark timing includes the first call, results are meaningless.

2. **Asynchronous execution.** CUDA operations are asynchronous: `torch.linalg.svd()` returns immediately while the GPU is still computing. Measuring time with `time.time()` around the call measures kernel launch latency, not execution time. The measurement must use `torch.cuda.synchronize()` before start and stop, or use `torch.cuda.Event(enable_timing=True)`.

3. **Batch size dependence.** SVD of a single 64x64 matrix on GPU is slower than CPU (kernel launch overhead dominates). SVD of a batch of 32 matrices is faster than CPU. The benchmark must test the actual batch sizes used in the pipeline (`batch_size=32` from `ExperimentConfig`), not unit operations.

4. **Driver parameter choice.** `torch.linalg.svd` supports different CUDA drivers via the `driver` kwarg (gesvd, gesvdj, gesvda). The default varies by PyTorch version and matrix size. The Jacobi method (gesvdj) is faster for small matrices but less accurate. The benchmark must specify and report the driver.

5. **Comparing to wrong baseline.** The reviewer asks for "cheaper approximation candidates." Randomized SVD (`torch.svd_lowrank`) is the obvious candidate, but it only computes the top-k singular values/vectors. If the codebase needs all singular values (for condition number, which uses sigma_n), randomized SVD is not applicable. The benchmark must compare full SVD vs. randomized SVD for the specific metrics that are actually needed.

**Why it happens in this codebase:** The `fused_evaluate` function runs SVD inside a `with torch.no_grad()` block, interleaved with autoregressive generation. The SVD calls are not batched -- they process one layer at a time (`output.qkt[:, layer_idx]`). The actual overhead is dominated by the sequential nature of autoregressive generation, not the SVD itself. A benchmark that measures SVD in isolation may be irrelevant to the actual bottleneck.

**Prevention:**
1. **Benchmark in context.** Measure SVD overhead as a fraction of total evaluation time, not in isolation. Instrument `fused_evaluate` with timing around the SVD section vs. the forward pass section vs. the behavioral classification section.
2. **GPU timing protocol:**
   ```python
   # Warmup
   for _ in range(10):
       torch.linalg.svd(warmup_matrix, full_matrices=False)
   torch.cuda.synchronize()

   # Measurement
   start = torch.cuda.Event(enable_timing=True)
   end = torch.cuda.Event(enable_timing=True)
   start.record()
   for _ in range(100):
       torch.linalg.svd(test_matrix, full_matrices=False)
   end.record()
   torch.cuda.synchronize()
   elapsed_ms = start.elapsed_time(end) / 100
   ```
3. **Test the actual matrix sizes.** QK^T is `[B, T, T]` = `[32, 64, 64]`. AVW_o is `[B, T, D]` = `[32, 64, 128]`. Benchmark both shapes.
4. **Compare meaningful alternatives.** For metrics that only need top-k singular values (Grassmannian distance with k=2, spectral gap 1-2), compare `torch.svd_lowrank(k=4)` vs. full SVD. For metrics that need sigma_n (condition number), full SVD is required -- note this in the analysis.
5. **Report relative cost.** "SVD adds X% overhead to evaluation time" is more useful than "SVD takes Y ms per matrix." The former answers the reviewer's question; the latter does not.

**Detection:** If the benchmark shows SVD takes <1% of evaluation time, the computational overhead concern is moot and the analysis can state so directly. If it shows >20%, the approximation analysis becomes critical.

**Phase:** SVD benchmarking is independent of all other v1.1 features and can be done at any time. It only reads from the existing pipeline.

---

### Pitfall 9: Sharp Compliance Curve -- Undersampling the Critical Region

**What goes wrong:** The reviewer asks for a "sharp compliance curve: r/w sweep showing near-perfect compliance at r << w, degradation at r -> w, failure at r >> w." This requires training the model at multiple r values and measuring compliance at each. The pitfalls:

1. **Insufficient r/w sampling near the transition.** If the sweep tests r/w in {0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0}, the compliance curve may look smooth but miss a sharp transition between r/w=0.7 and r/w=0.9. The reviewer's point is that the transition is *sharp* -- testing must be dense enough to capture it.

2. **Training quality variation.** Each r value requires a separate trained model (or at least fine-tuned from the anchor). If some r values happen to get lucky training runs (good seed) and others do not, the compliance curve reflects training variance, not the r/w effect. The existing sufficiency gate (edge >95%, rule >80%) filters failed training, but does not control for training quality above the threshold.

3. **Confounding r with other difficulty factors.** As r increases, the number of eligible target-block paths changes, the contamination rate changes, and the event count changes. The compliance curve may reflect these structural changes, not just the r/w ratio.

**Prevention:**
1. **Dense r/w sampling near the expected transition.** Use r/w in {0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.1, 1.3, 1.5, 2.0}. The dense sampling between 0.7 and 1.0 captures the transition.
2. **3 seeds per r value.** Report mean +/- std compliance across seeds. This controls for training variance.
3. **Fixed graph across r values.** Use the same DCSBM graph for all r values in the sweep. This controls for graph-level confounds. (The existing graph caching already supports this -- same `GraphConfig` produces same graph.)
4. **Report both edge compliance and rule compliance separately.** The transition may be sharp in rule compliance but gradual in edge compliance. The reviewer's claim is about the *rule* compliance curve.

**Detection:** Plot compliance vs. r/w with error bars. If the transition spans more than 0.2 units of r/w, it may not be "sharp" and the claim needs to be softened.

**Phase:** The compliance curve requires training multiple models, which is the most GPU-expensive part of v1.1. Budget carefully: 15 r values * 3 seeds = 45 training runs.

---

### Pitfall 10: Softmax Bound Empirical Verification Uses Wrong Perturbation Direction

**What goes wrong:** Empirical verification of the softmax filtering bound requires perturbing QK^T and measuring the resulting change in AVW_o. Common errors in the verification:

1. **Random perturbation direction biases toward average case.** A random perturbation on a T x T matrix has expected spectral norm of sqrt(T). The worst-case perturbation (that maximizes the spectral change in A) is aligned with the attention pattern. Testing only random perturbations validates the average-case bound, not the worst case. The reviewer wants a *bound*, which means the worst case.

2. **Perturbation applied to the wrong matrix.** The QK^T in `src/model/attention.py:71` is already scaled by 1/sqrt(d_model). Perturbing the *unscaled* QK^T and then scaling gives a different epsilon than perturbing the *scaled* QK^T directly. The bound must specify which matrix is perturbed.

3. **Measuring change in the wrong norm.** The SVD metrics use different norms: Grassmannian distance uses principal angles (Frobenius-like), stable rank uses the ratio of Frobenius to spectral norm, condition number uses the ratio of extreme singular values. The bound's norm must match the metric's norm.

**Prevention:**
1. **Test both random and adversarial perturbations.** For adversarial, compute the gradient of ||delta(AVW_o)||_F with respect to the QK^T perturbation and use the gradient direction as the worst-case perturbation. This requires a backward pass through softmax and matrix multiply, which PyTorch autograd supports.
2. **Specify clearly:** "We perturb the scaled QK^T (after 1/sqrt(d_k) scaling, before softmax) by epsilon in spectral norm."
3. **Report bound tightness:** For each epsilon level, report the ratio of observed max ||delta(AVW_o)|| to the theoretical bound. A ratio of 0.3-0.8 means the bound is useful; a ratio of 0.01 means the bound is vacuous.

**Detection:** If the worst-case empirical perturbation exceeds the bound, the derivation is wrong. If no perturbation reaches more than 1% of the bound, the bound is too loose to be scientifically useful.

**Phase:** This is part of the softmax bound phase (Pitfall 3). The empirical verification should be implemented as a test, not as a one-off script.

---

## Minor Pitfalls

---

### Pitfall 11: NPZ Storage Grows Significantly With Full Spectrum Tracking

**What goes wrong:** Storing full singular value vectors (top-k=8 per step) adds 8 * T * n_layers * n_sequences floats to the NPZ file. With T=256, n_layers=4, n_sequences=1000: 8 * 256 * 4 * 1000 = 8.2M floats = ~33 MB. For multi-head with 4 heads, this becomes 132 MB per experiment. With 45 training runs for the compliance curve, storage reaches ~6 GB.

**Prevention:** Store spectrum trajectories in a separate NPZ file (e.g., `spectrum_trajectories.npz`) rather than adding to the existing `token_metrics.npz`. This preserves backward compatibility and allows selective loading.

**Phase:** Storage management throughout v1.1.

---

### Pitfall 12: Calibration Curves Require Probability Outputs the Pipeline Does Not Produce

**What goes wrong:** The reviewer asks for "reliability diagrams." These plot predicted probability vs. observed frequency. The current pipeline produces raw SVD metric values, not probabilities. Converting metric values to probabilities requires a calibration model (isotonic regression, Platt scaling), which introduces model selection and overfitting risk.

**Prevention:**
1. Use isotonic regression (non-parametric, no hyperparameters) rather than Platt scaling (parametric, two hyperparameters).
2. Fit the calibration model on the exploratory split, evaluate on the held-out split.
3. Report ECE (Expected Calibration Error) to quantify calibration quality.

**Phase:** Calibration depends on the pre-registration held-out split being implemented first.

---

### Pitfall 13: Signal Concentration Metric Has No Established Statistical Test

**What goes wrong:** The multi-head ablation should answer "is the SVD signal concentrated in one head or distributed across heads?" There is no standard statistical test for this. Ad hoc approaches (variance of per-head AUROC, max-head AUROC vs. mean-head AUROC) are reasonable but may not satisfy a rigorous reviewer.

**Prevention:** Frame signal concentration as a descriptive statistic, not a hypothesis test. Report per-head AUROC with confidence intervals. If one head's CI does not overlap with 0.5, the signal is present in that head. If all heads' CIs overlap with each other, the signal is distributed.

**Phase:** Multi-head ablation phase.

---

### Pitfall 14: Pre-Registration Document Becomes Stale As Implementation Progresses

**What goes wrong:** The pre-registration is written at the start of v1.1. During implementation, decisions are made that deviate from the plan (different smoothing window for curvature, different calibration method, additional metrics). If deviations are not documented, the pre-registration is undermined.

**Prevention:** Maintain a deviation log alongside the pre-registration document. Each deviation entry has: date, what changed, why, impact on confirmatory analysis. This is standard practice per Willroth & Atherton (2024).

**Phase:** Throughout v1.1.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Pre-registration setup | Post-hoc rationalization (Pitfall 4) | Write and commit before any v1.1 analysis code runs |
| v1.0 integration fixes | Breaking existing functionality (Pitfall 5) | Golden regression test suite before any changes |
| Null model baseline | Confounded controls (Pitfall 1) | Same graph, same model, jumper-free walks, position-matched |
| Multi-head ablation | Shape/semantics breakage (Pitfall 2) | New attention class, per-head metric keys, backward-compatible tensors |
| Softmax filtering bound | Mathematical errors (Pitfall 3) | Use L=1/2 Lipschitz, include 1/sqrt(d_k) scaling, empirical verification |
| Softmax empirical verification | Wrong perturbation direction (Pitfall 10) | Test both random and adversarial perturbations |
| PR curves and calibration | Imbalanced baseline (Pitfall 6), no probabilities (Pitfall 12) | Report prevalence baseline, use isotonic regression |
| Sharp compliance curve | Undersampled transition (Pitfall 9) | Dense r/w sampling near r/w=0.8-1.0, 3 seeds |
| Spectrum trajectory | Numerical differentiation noise (Pitfall 7) | Savitzky-Golay smoothing, Grassmannian curvature |
| SVD benchmarks | GPU timing artifacts (Pitfall 8) | CUDA events, warmup, in-context measurement |
| Storage and NPZ | Size growth (Pitfall 11) | Separate files for spectrum data |
| Pre-registration maintenance | Stale document (Pitfall 14) | Deviation log with date, reason, impact |

---

## Integration-Specific Warnings

These pitfalls are unique to adding v1.1 features to the existing v1.0 codebase.

| Integration Point | Risk | Mitigation |
|---|---|---|
| `ExperimentConfig.n_heads` validation | Relaxing from `== 1` to `in (1,2,4)` may skip other validations | Add explicit validation: `d_model % n_heads == 0` and `d_model // n_heads >= 16` |
| `AttentionInternals.qkt` shape | Adding head dimension `[B, H, T, T]` breaks all consumers | Use `H=1` squeeze for single-head; add `n_heads` field to `AttentionInternals` |
| `PRIMARY_METRICS` set | Adding per-head variants changes Holm-Bonferroni denominator | Keep primary metrics at the head-aggregated level; per-head analysis is exploratory |
| `fused_evaluate` modifications | Two independent modifications (null model, spectrum storage) can conflict | Design both as additive storage (new NPZ keys), not modifications to existing keys |
| `auroc_horizon.py` | Adding PR curves alongside AUROC must not change existing AUROC computation | Add `compute_pr_curve` as a new function called after `compute_auroc_curve`, not integrated into it |
| `statistical_controls.py` | Calibration and ECE are new statistical controls that should not modify existing Holm-Bonferroni pipeline | Add `compute_calibration` as a separate function; existing `apply_statistical_controls` calls it as a new step |
| Visualization | New plot types (PR curve, calibration, compliance curve, spectrum trajectory) must use existing style system | Import from `src/visualization/style.py`; add new modules rather than modifying existing ones |
| Report templates | New sections for null model, multi-head, softmax bound must not break existing report structure | Add new sections in the template after existing ones; handle missing sections gracefully |

---

## Sources

- Softmax Lipschitz constant (1/2, tight across all lp norms): [arXiv:2510.23012](https://arxiv.org/abs/2510.23012) -- HIGH confidence, published proof
- Local Lipschitz bound for self-attention depending on attention distribution: [arXiv:2507.07814](https://arxiv.org/abs/2507.07814) -- HIGH confidence, ICML-level work
- Spectral rank collapse in attention layers (random matrix theory analysis): [arXiv:2410.07799](https://arxiv.org/html/2410.07799v2) -- MEDIUM confidence, recent preprint
- Per-head vs. joint SVD decomposition in multi-head attention: [SDRM (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0950705125009918) and [Alignment Forum SVD analysis](https://www.alignmentforum.org/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight) -- MEDIUM confidence
- Mechanistic interpretability via singular directions in attention heads: [OpenReview](https://openreview.net/forum?id=7UbXEQNny7) -- MEDIUM confidence
- Pre-registration deviation reporting best practices: [Willroth & Atherton 2024](https://journals.sagepub.com/doi/10.1177/25152459231213802) -- HIGH confidence, published methodology
- Pre-registration for predictive modeling: [arXiv:2311.18807](https://arxiv.org/html/2311.18807) -- MEDIUM confidence
- Effect of class imbalance on PR curves: [MIT Press](https://direct.mit.edu/neco/article/33/4/853/97475/The-Effect-of-Class-Imbalance-on-Precision-Recall) -- HIGH confidence, published
- GPU kernel benchmarking pitfalls (warmup, async timing): [StandardKernel blog](https://standardkernel.com/blog/in-pursuit-of-high-fidelity-gpu-kernel-benchmarking/) -- HIGH confidence, well-documented practice
- PyTorch SVD GPU performance issues: [pytorch/pytorch#86234](https://github.com/pytorch/pytorch/issues/86234), [pytorch/pytorch#41306](https://github.com/pytorch/pytorch/issues/41306) -- HIGH confidence, verified issues
- torch.linalg.svd driver parameter documentation: [PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html) -- HIGH confidence, official docs
- Principal angles between random subspaces (distribution theory): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0024379505004878) -- HIGH confidence, established math
- Canonical angle bounds for randomized SVD: [arXiv:2211.04676](https://arxiv.org/html/2211.04676) -- MEDIUM confidence, recent preprint with 2024 revision
- Null models in network neuroscience (design principles): [Nature Reviews Neuroscience](https://www.nature.com/articles/s41583-022-00601-9) -- HIGH confidence, published review
- Numerical differentiation of noisy data: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7899139/) -- HIGH confidence, published methods
