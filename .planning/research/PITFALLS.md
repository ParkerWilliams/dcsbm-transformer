# Domain Pitfalls

**Domain:** DCSBM transformer SVD hallucination prediction research framework
**Researched:** 2026-02-24
**Confidence:** HIGH (draws on well-established numerical linear algebra, graph theory, and ML training failure modes; verified against project spec constraints)

---

## Critical Pitfalls

Mistakes that cause rewrites, wasted budget, or invalidated results.

---

### Pitfall 1: Degenerate DCSBM Graphs — Disconnected Blocks and Isolated Vertices

**What goes wrong:** With n=200 and 16 blocks, each block has ~12 vertices. At p_out=0.01, expected out-group edges per vertex are ~0.01 * 188 = 1.88. Some blocks will be disconnected from others entirely. Walks cannot traverse between blocks, so block jumper rules become impossible to satisfy. The graph also risks isolated vertices (degree 0) which poison walk generation.

**Why it happens:** The DCSBM parameter space includes extreme corners (small n, many blocks, low p_out) where expected degree drops below the connectivity threshold. Degree correction amplifies this: if the degree-correction distribution has heavy tails, some vertices get near-zero expected degree even in dense blocks.

**Consequences:** (a) Walk generation enters infinite loops trying to find valid transitions from dead-end vertices. (b) Block jumper rules are unsatisfiable, so training data contains no positive examples and the sufficiency gate can never be met. (c) Silent data corruption if the walk generator falls back to uniform random restarts, creating artificial distribution shifts. (d) Budget burned on configs that were dead on arrival.

**Warning signs:**
- Graph has connected components > 1 (check with BFS/DFS after generation)
- Any vertex has out-degree 0
- Block jumper path validation fails (no valid path of length r exists)
- Walk generation takes orders of magnitude longer than expected

**Prevention:**
1. After DCSBM generation, run a connectivity check: require the graph is strongly connected (directed graph). If not, reject the config and log why.
2. Validate that for every block jumper vertex, at least one valid path of length r to the target block exists. Do this with BFS on the product graph (vertex x step-count).
3. Set minimum expected degree threshold: reject configs where min(expected_degree) < 3 for any block.
4. For the degree correction distribution, use a bounded distribution (e.g., Uniform[0.5, 2.0]) not a power law. Power law degree correction with small blocks creates isolated vertices.
5. Pre-filter the sweep: before allocating GPU time, run graph generation and validation on CPU for all configs. Reject invalid ones before they enter the job queue.

**Detection:** Automated validation step immediately after graph generation, before any walk generation or training begins.

**Phase:** Graph Generation (Phase 1). This is a gate: no downstream work should proceed on an invalid graph.

---

### Pitfall 2: Trivial Block Jumper Rules — Rule Is Learnable From Local Topology

**What goes wrong:** If p_in is high and p_out is low, the block structure is so strong that vertices near block boundaries are obvious. A block jumper vertex in block A with target block B at distance r may have a unique path pattern that the transformer learns as a simple bigram or trigram pattern rather than learning the deep rule. The experiment then measures pattern matching, not the intended "hidden rule learning."

**Why it happens:** When the graph is very assortative (p_in >> p_out), the only paths between blocks go through a small set of "bridge" vertices. The block jumper rule collapses to "if you see vertex X, predict vertex Y" — a lookup table, not a learned rule. The spec acknowledges this: "Valid paths from v_i to the target block must exist at length r but must not be the only paths."

**Consequences:** (a) The transformer achieves high rule compliance trivially, passing the sufficiency gate but not actually learning the intended hidden rule. (b) SVD metrics show no meaningful signal because there is no uncertainty to detect — the model is confident and correct. (c) The research question becomes untestable.

**Warning signs:**
- Rule compliance reaches >95% very early in training (within first few percent of epochs)
- Confusion matrix shows near-zero rule violations even on held-out walks
- SVD metrics are flat (no variance) across block jumper positions
- Path analysis shows fewer than 3 distinct paths of length r from jumper to target block

**Prevention:**
1. After graph generation and block jumper assignment, enumerate paths of length r from each jumper vertex. Require: (a) at least 3 distinct paths to the target block exist, AND (b) at least 3 distinct paths of length r that do NOT reach the target block also exist. This ensures the rule is not topologically determined.
2. Compute a "rule difficulty" metric: the fraction of all length-r paths from the jumper vertex that reach the target block. If this fraction is >0.8 or <0.05, the rule is either trivial or near-impossible. Target the range 0.1-0.5.
3. The anchor config (n=500, p_in=0.25, p_out=0.03, 8 blocks) should be validated for non-triviality before anything else runs.

**Detection:** Automated path enumeration after block jumper assignment (can be done via matrix power: A^r where A is adjacency, check entries from jumper to target block vertices).

**Phase:** Graph Generation (Phase 1). Part of the same validation gate as Pitfall 1.

---

### Pitfall 3: SVD Memory Blowup From Every-Step Collection

**What goes wrong:** The spec requires SVD metrics "at every token step." For n=500 vertices (vocabulary), w=256 context window, d_model=256, walks of length 8w=2048 tokens: QK^T is w x w = 256 x 256 per step. Collecting ~20 metrics at each of 2048 steps per walk, across 200k walks in the training corpus, means storing 200k * 2048 * 20 = ~8.2 billion floats. At float32, that is ~33 GB per config. With 3 seeds and multiple configs, this is hundreds of GB.

**Why it happens:** The spec says "every token step" but this likely means every step during evaluation walks, not during training. However, even during evaluation, if you collect SVD on 10,000 held-out walks of length 512, that is 10k * 512 * 20 = 102M values = ~400 MB per config. Manageable, but only if you are disciplined about what gets stored and when.

**Consequences:** (a) OOM on the GPU during evaluation if intermediate SVD results are held in GPU memory. (b) Disk space exhaustion on RunPod, which typically has 20-50 GB of container storage. (c) result.json files become multi-GB, making reporting and plotting painfully slow. (d) Budget burned on storage costs if using network storage.

**Warning signs:**
- GPU OOM during evaluation (not training)
- result.json files exceed 100 MB
- Disk usage warnings on RunPod instance
- Evaluation takes 10x longer than training

**Prevention:**
1. Collect SVD metrics only during evaluation, not during training. Training only needs cross-entropy loss.
2. Use a streaming approach: compute SVD metrics per step, write to a memory-mapped file or HDF5, do not accumulate in Python lists or GPU tensors.
3. Store only the scalar metrics (~20 floats per step), not the full singular vectors. The full SVD decomposition (U, S, V^T) for a 256x256 matrix is ~400KB per step — never store this.
4. For the result.json sequences block, store only the evaluation walks that contain block jumper events (not all walks). Most walks never encounter a jumper and are uninteresting for the core analysis.
5. Budget evaluation walk count carefully: 1000 walks with jumper events is likely sufficient for statistical power. More is waste.
6. Use float16 for stored metrics where precision is not critical (all metrics except condition number, which can overflow in float16).

**Detection:** Profile memory usage during anchor config evaluation before launching the sweep. Set hard limits: abort if any single result.json exceeds 200 MB.

**Phase:** SVD Collection Pipeline (Phase 3). Must be designed correctly from the start — retrofitting streaming is a rewrite.

---

### Pitfall 4: SVD Numerical Instability With Repeated or Near-Zero Singular Values

**What goes wrong:** The QK^T matrix for a single-head transformer with d_model=64-256 and context window w=32-256 is a w x w matrix. Early in training, this matrix is near-random (all singular values similar). Later in training, it may become low-rank (a few dominant singular values, many near-zero). Several of the specified metrics fail in these regimes:
- **Condition number** (sigma_1/sigma_n): explodes to infinity when sigma_n approaches zero. At float32, sigma_n < 1e-7 gives condition number > 1e7, losing all precision.
- **Singular value entropy**: if any sigma_i is exactly 0, p_i = 0, and 0*log(0) is NaN. Even with the convention 0*log(0)=0, near-zero values create numerical noise.
- **Spectral gap** (sigma_1 - sigma_2): when sigma_1 and sigma_2 are nearly equal (common in early training), the gap is dominated by floating-point noise.
- **Angular velocity of principal vector**: when the top two singular values are nearly equal, the principal singular vector is not uniquely defined and can flip 180 degrees between steps due to sign convention or numerical perturbation. This produces spurious spikes.
- **Low-rank approximation error**: Frobenius norm of QK^T minus rank-k approximation can accumulate floating-point error for large matrices.

**Why it happens:** SVD is numerically stable for computing singular values, but singular vectors corresponding to repeated or clustered singular values span a subspace, and which specific vectors the algorithm returns within that subspace is implementation-dependent and non-deterministic. PyTorch's `torch.linalg.svd` uses cuSOLVER on GPU, which does not guarantee deterministic ordering within degenerate subspaces.

**Consequences:** (a) Spurious spikes in angular velocity, subspace drift, and condition number create false positives in the predictive signal. (b) NaN/Inf values propagate into AUROC calculation, producing nonsensical results. (c) If these artifacts correlate with any position pattern (e.g., early in context window where QK^T is more degenerate), they create a confound with the predictive horizon analysis.

**Warning signs:**
- NaN or Inf in any SVD metric time series
- Condition number > 1e6 at any step
- Angular velocity showing regular spike patterns unrelated to block jumper positions
- Principal vector direction flipping by ~180 degrees between adjacent steps

**Prevention:**
1. **Clamp condition number**: use `min(sigma_1/max(sigma_n, eps), 1e6)` where eps=1e-7. Log-transform before storing and plotting.
2. **Entropy safety**: compute as `entropy = -sum(p * log(p + eps))` where eps=1e-10. Better: compute `entropy = log(sum(sigma)) - sum(sigma * log(sigma + eps)) / sum(sigma)`.
3. **Angular velocity robustness**: instead of tracking the single principal vector, track the principal k-subspace (k=2 or 4) using Grassmannian distance (principal angles). This is invariant to rotations within the subspace. The spec already includes this as "principal angles between consecutive dominant subspaces" — use this instead of single-vector angle as the primary directional metric.
4. **Spectral gap**: report sigma_1/sigma_2 (ratio) instead of sigma_1 - sigma_2 (difference). The ratio is scale-invariant and more numerically stable.
5. **Deterministic SVD**: set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Be aware this may slow down SVD by 2-3x on GPU. Profile the tradeoff.
6. **Validation pass**: after collecting all metrics for a walk, scan for NaN/Inf. Replace with sentinel values (-999) and log a warning. Never silently drop or interpolate.

**Detection:** Automated NaN/Inf check after each evaluation walk. Histogram of condition numbers and angular velocities to spot degenerate regimes.

**Phase:** SVD Collection Pipeline (Phase 3). Numerical guards must be built into the metric computation functions, not added as post-processing.

---

### Pitfall 5: Burning $100 Budget on Misconfigured or Low-Priority Runs

**What goes wrong:** The full parameter sweep has approximately 4 * 4 * 3 * 3 * 3 * 4 * 8 * 3 * 3 * 3 * 3 = millions of configurations (conservatively). Even the "core" sweep across r values with the anchor config (varying only r across 8 values, 3 seeds each) is 24 runs. If each run takes 2 hours on an RTX 3090 at $0.44/hr, that is $21 just for the core experiment. Add architecture sweeps, corpus size sweeps, and graph parameter sweeps, and the budget is exhausted before the most interesting results are computed.

**Why it happens:** Researchers (and especially automated sweep systems) tend to launch the full grid and hope. With $100 and GPU rates of $0.44-$0.74/hr, you have 135-227 GPU-hours total. If a single anchor config takes 1-2 hours end-to-end (training + SVD collection + evaluation), you can run 67-227 configs. But the sweep has thousands of configs.

**Consequences:** (a) Budget exhausted on uninteresting parameter combinations while the core r-vs-w interaction is incomplete. (b) If the first runs reveal a bug, the budget spent before the bug is found is wasted. (c) Incomplete data across all dimensions is worse than complete data on the most important dimension.

**Warning signs:**
- No wall-time estimate before launching sweep
- Job queue not priority-ordered
- First run is not the anchor config (which calibrates timing)
- No kill switch or budget monitor

**Prevention:**
1. **Anchor config first**: run exactly one config (n=500, w=64, t=200k, d_model=128, n_layers=4, r=0.9w, 1 seed) end-to-end. Measure wall time for each phase: graph generation, walk generation, training, SVD collection, evaluation, reporting. This calibrates the entire budget plan.
2. **Priority-ordered job queue**: define three tiers:
   - **Tier 1 (must run)**: Core r-sweep at anchor config (8 r-values x 3 seeds = 24 runs). This is the headline result.
   - **Tier 2 (should run)**: w-sweep at anchor (4 w-values x 3 r-values x 3 seeds = 36 runs). Architecture sweep (3 d_model x 3 n_layers x 3 seeds = 27 runs). These validate robustness.
   - **Tier 3 (nice to have)**: Graph parameter sweeps (p_in, p_out, blocks). Corpus size sweep. Large n sweeps. These are secondary.
3. **Budget tracking**: after each run, log GPU-hours consumed and remaining budget. Halt the queue when budget reaches $10 reserve (for re-runs and debugging).
4. **Early termination**: if a config fails the sufficiency gate after 50% of the training budget, kill it and move on. Don't let one bad config consume 4 hours of training.
5. **Spot instances**: RTX 3090 spot pricing on RunPod is ~$0.22/hr vs $0.44/hr on-demand. Use spot for Tier 2 and Tier 3. Use on-demand only for Tier 1 to avoid interruption.

**Detection:** Budget dashboard: total spent, runs completed, runs remaining, estimated total cost. Check after every run.

**Phase:** Experiment Infrastructure (Phase 2) and Sweep Execution (Phase 4). The job queue priority system must be built before any sweep runs.

---

### Pitfall 6: Training Does Not Reach Sufficiency Gate

**What goes wrong:** The sufficiency gate requires edge compliance >95% and rule compliance >80%. For difficult configs (large r, low p_in, many blocks), the transformer may never reach this threshold within the training budget. The spec says these configs are "excluded from SVD analysis" — but if too many configs are excluded, the experiment has no data.

**Why it happens:**
- **Insufficient corpus**: t=50k walks with n=2000 vertices means each vertex appears in ~25 walks on average. Block jumper vertices (say 5 per block, 8 blocks = 40 jumpers out of 2000) each appear in ~1 walk. The model sees each jumper rule ~1 time — far too few to learn it.
- **Wrong learning rate**: too high causes oscillation and never converges; too low causes convergence so slow it times out.
- **Architecture too small**: d_model=64 with n_layers=2 may not have the capacity to learn block jumper rules for r > w. The model needs to encode a lookup from jumper vertex identity to target block, conditioned on step count — this requires memory circuits that small models may lack.
- **Vocabulary too large**: n=2000 vertices means a vocabulary of 2000 tokens. With d_model=64, the embedding matrix is 2000 x 64 = 128K parameters. The model may spend all capacity on the embedding layer and have nothing left for attention pattern learning.

**Consequences:** (a) Configs excluded from analysis, reducing statistical power. (b) If the anchor config fails the gate, the entire project is blocked. (c) Temptation to lower the gate thresholds, which invalidates the analysis (SVD on an untrained model is noise).

**Warning signs:**
- Training loss plateaus well above the edge compliance threshold
- Edge compliance stalls at 80-90% and does not improve
- Rule compliance stays below 50% after 80% of training steps
- Learning rate finder shows no clear minimum

**Prevention:**
1. **Anchor config sanity check**: before sweeping, verify that the anchor config (n=500, w=64, t=200k, d_model=128, 4 layers) can reach the gate. If it cannot, the problem is fundamental and the architecture or training setup needs fixing.
2. **Curriculum on corpus size**: use the formula t >= 100 * n * max_rule_r / w. For n=500, r=128 (2w), w=64: t >= 100 * 500 * 128/64 = 100k. The spec's t=200k should be sufficient for the anchor but may be insufficient for large r.
3. **Learning rate schedule**: use a cosine annealing schedule with warmup (5% of training). Start from a learning rate finder result. For NanoGPT-scale, 3e-4 is a reasonable starting point for d_model=128.
4. **Early stopping with patience**: if edge compliance has not improved in the last 10% of training steps, stop early and report the config as gate-failed.
5. **Architecture scaling rule**: d_model should be at least 2 * ceil(log2(n)). For n=2000, d_model >= 22, so even d_model=64 is sufficient. But n_layers matters more for rule learning — use at least 4 layers for r > w.
6. **Corpus composition**: ensure each block jumper vertex appears in at least 50 walks. If the random walk process undersamples jumpers, bias the walk starting distribution toward jumper vertices for a fraction of the corpus.

**Detection:** Training curves monitored in real time. Gate check at 50% and 80% of training budget. If not on track at 80%, abort.

**Phase:** Training Pipeline (Phase 2). The sufficiency gate logic must be implemented before any sweep runs.

---

### Pitfall 7: Confounded Predictive Horizon — Position Effects Masquerading as SVD Signal

**What goes wrong:** SVD metrics naturally vary with position in the context window. Early positions have less context, so the QK^T matrix is structurally different (many rows are padding or positional encoding dominated). If block jumper events tend to occur at specific positions in the walk (e.g., always near the middle because of how walks are generated), the "predictive signal" is actually a position artifact.

**Why it happens:** Block jumper vertices are placed at specific positions in the DCSBM. When walks traverse them, the jumper event occurs at a position determined by the walk structure. If all walks start from the same vertex or the walk generator has a bias, jumper events cluster at similar positions. Meanwhile, SVD metrics have inherent position trends (condition number tends to increase with context length as the attention pattern sharpens).

**Consequences:** (a) AUROC appears significant but is measuring position correlation, not causal prediction. (b) The predictive horizon result is an artifact of the walk generation process, not a property of the transformer's attention mechanism. (c) The paper's central claim is wrong.

**Warning signs:**
- Predictive horizon AUROC > 0.75 even for non-jumper events (the "correct" baseline should have AUROC ~0.5)
- SVD metric values at position j before a jumper event are similar to SVD metric values at the same absolute position in non-jumper walks
- Jumper events cluster at specific absolute positions in evaluation walks

**Prevention:**
1. **Position-matched baselines**: for every jumper event at position p in a walk, sample a control event at position p in a non-jumper walk. The AUROC should be computed on the jumper-vs-matched-control contrast, not jumper-vs-all-other-positions.
2. **Shuffle control**: permute the jumper event labels across walks (keeping positions intact). If AUROC remains high, the signal is positional, not jumper-related.
3. **Detrending**: subtract the position-wise mean SVD metric (computed from non-jumper walks) from the jumper walk metrics before computing AUROC. This removes the positional baseline.
4. **Walk generation diversity**: start walks from uniformly random vertices. Ensure jumper events occur at varied positions in the context window. If all jumper events occur at position w/2 +/- 5, the experiment is confounded.
5. **Report position distributions**: for every AUROC result, also report the distribution of jumper event positions in the evaluation set.

**Detection:** Run the shuffle control and position-matched baseline as part of the standard analysis pipeline. If shuffle control AUROC > 0.6, flag the result.

**Phase:** Experiment Design (Phase 2) and Analysis Pipeline (Phase 4). The position-matched baseline must be designed into the evaluation from the start.

---

## Moderate Pitfalls

---

### Pitfall 8: O(d^3) SVD Scaling Makes Large Configs Unaffordable

**What goes wrong:** SVD of a w x w matrix is O(w^3). For w=256, that is ~16.7M flops per step. At 2048 steps per walk and 1000 evaluation walks, that is ~34 billion flops just for SVD. On an RTX 3090 (35.6 TFLOPS fp32), this is ~1 second. Sounds fine, but: (a) `torch.linalg.svd` on GPU has significant kernel launch overhead for small matrices, so actual time is 10-100x the theoretical FLOP time; (b) you are also computing 20 derived metrics per step; (c) w=256 configs take 64x longer than w=64 configs.

**Prevention:**
1. Profile SVD time on the anchor config. Extrapolate to all configs using w^3 scaling.
2. For w=256, consider computing SVD on only a subset of evaluation steps (every 4th step) and interpolating. This trades temporal resolution for speed.
3. Use `full_matrices=False` in `torch.linalg.svd` — this returns only the first min(m,n) singular vectors, which is all you need.
4. Batch the SVD calls: accumulate QK^T matrices from multiple steps into a batch tensor and call SVD once. `torch.linalg.svd` supports batched inputs.
5. Consider whether all 20 metrics are needed per step. Some (like rank, variance) can be computed from singular values alone without the full decomposition. Split metrics into "values-only" (fast) and "vectors-needed" (slow) groups.

**Detection:** Wall-time logging per evaluation walk. If SVD collection exceeds training time, the pipeline is unbalanced.

**Phase:** SVD Collection Pipeline (Phase 3).

---

### Pitfall 9: Multiple Comparisons Inflation in Predictive Horizon Analysis

**What goes wrong:** The analysis tests ~20 SVD metrics at each lookback distance j from 1 to r. For r=128, that is 20 * 128 = 2560 statistical tests per config. With 24 configs in the core sweep and 3 seeds each, that is ~184K tests. At alpha=0.05, you expect ~9200 false positives by chance. Some metric at some lookback at some config will appear significant purely by chance.

**Prevention:**
1. **Pre-register primary metrics**: before running the experiment, designate 3-5 SVD metrics as primary (e.g., singular value entropy, condition number, subspace drift). All other metrics are exploratory. Only primary metrics contribute to the headline AUROC result.
2. **Bonferroni or Holm-Bonferroni correction**: within each config, correct across the number of primary metrics * number of lookback distances tested.
3. **FDR control**: use Benjamini-Hochberg procedure instead of Bonferroni if you want more power. Report both the raw and adjusted p-values.
4. **Cross-validation of AUROC**: compute AUROC on a held-out split of evaluation walks. If AUROC is high on the training split but low on the held-out split, the metric is overfitting to noise.
5. **Effect size, not just significance**: report Cohen's d or the AUROC confidence interval alongside the p-value. A statistically significant AUROC of 0.52 is meaningless.

**Detection:** If more than 30% of all metric-lookback combinations are significant at alpha=0.05, the result is likely inflated.

**Phase:** Analysis Pipeline (Phase 4). Multiple comparison correction must be built into the reporting, not added post-hoc.

---

### Pitfall 10: Misleading AUROC From Class Imbalance

**What goes wrong:** Rule violations are rare events. If only 5% of evaluation steps are rule violation events, and the predictor says "no violation" for all steps, accuracy is 95% but AUROC is 0.5. More insidiously, if the SVD metric has a slight upward trend over the course of a walk (e.g., entropy increases with context length), the metric will weakly discriminate violation-vs-non-violation by coincidence of position, giving AUROC of 0.55-0.65 that looks like a signal.

**Prevention:**
1. **Balance the evaluation set**: for AUROC computation, use equal numbers of violation and non-violation steps. Subsample the majority class.
2. **Report precision-recall curves alongside AUROC**: PR curves are more informative under class imbalance.
3. **Use AUROC only on the position-matched contrast** (see Pitfall 7): compare violation steps to control steps at the same position.
4. **Confidence intervals on AUROC**: use DeLong's method or bootstrap. AUROC on 50 violation events has wide confidence intervals (95% CI width ~0.15). You need at least 100 events per config for reasonable precision.
5. **Sample size calculation upfront**: for AUROC with 95% CI half-width of 0.05, you need ~400 violation events per config. Plan evaluation walk count accordingly.

**Detection:** If AUROC confidence interval includes 0.5 (chance level), the result is not significant regardless of the point estimate.

**Phase:** Analysis Pipeline (Phase 4).

---

### Pitfall 11: Non-Deterministic PyTorch Operations Breaking Reproducibility

**What goes wrong:** The spec requires reproducibility from "stored config + code hash + seed." But PyTorch has multiple sources of non-determinism: (a) cuDNN autotuning selects different algorithms across runs; (b) atomicAdd operations in some CUDA kernels are non-deterministic; (c) `torch.linalg.svd` on GPU uses cuSOLVER which is non-deterministic for matrices with repeated singular values; (d) data loading order with num_workers > 0; (e) Python hash randomization affects dict ordering.

**Prevention:**
1. Set all seeds at the start of each run:
   ```python
   import torch, numpy, random
   torch.manual_seed(seed)
   numpy.random.seed(seed)
   random.seed(seed)
   torch.cuda.manual_seed_all(seed)
   ```
2. Disable cuDNN autotuning: `torch.backends.cudnn.benchmark = False`
3. Enable deterministic mode: `torch.use_deterministic_algorithms(True)` and set `CUBLAS_WORKSPACE_CONFIG=:4096:8` as an environment variable.
4. Use `num_workers=0` in DataLoader, or use a worker seed generator.
5. Accept that SVD vectors for near-degenerate singular values will differ across runs. Design metrics to be robust to this (use subspace distances, not individual vector angles).
6. Define "reproducibility" as: same config + seed produces same training loss curve, same edge/rule compliance, and same scalar SVD metric values (within float32 tolerance of 1e-5). Do NOT require bitwise identical results.
7. Store the PyTorch version, CUDA version, and cuDNN version in result.json metadata. Different versions produce different results.

**Detection:** Run the anchor config with the same seed twice. Compare training loss at each step. If they diverge, determinism is broken — fix before proceeding.

**Phase:** Infrastructure (Phase 1). Seed management must be the very first thing implemented.

---

### Pitfall 12: Walk Corpus Does Not Cover the Graph Adequately

**What goes wrong:** Random walks on a DCSBM are biased toward high-degree vertices and dense blocks. Block jumper vertices may have atypical degree (either very high or very low depending on degree correction). With t=50k walks of length 2w=128, total tokens are 6.4M. For n=500, average tokens per vertex is 12,800 — seemingly enough. But the distribution is heavy-tailed: the highest-degree vertex might appear 50,000 times while the lowest-degree appears 100 times. Block jumper rules need the jumper vertex to appear enough times at the right context offset for the model to learn the rule.

**Prevention:**
1. After generating the corpus, compute a coverage report: min/max/median times each vertex appears, and specifically how many times each block jumper vertex appears.
2. If any block jumper appears fewer than 50 times in the corpus, increase t or bias the walk starts.
3. Use a mixed strategy: 80% of walks start from uniformly random vertices, 20% start from block jumper vertices. This ensures rule coverage without distorting the overall distribution too much.
4. Walk length matters: walks of length l=2w may not include enough room for a jumper event at position >w. Use l >= r + w to ensure every walk that starts at a jumper can fully express the rule.
5. The spec says l is swept at 2w, 4w, 8w. Use l=4w as the anchor value, not 2w, to avoid truncation of long-range rules.

**Detection:** Coverage report after corpus generation. Flag if any jumper has < 50 occurrences.

**Phase:** Walk Generation (Phase 1).

---

### Pitfall 13: Single Attention Head Creates Bottleneck Artifacts

**What goes wrong:** The spec mandates a single attention head for interpretability. This is scientifically justified but creates a real risk: with one head, the QK^T matrix must simultaneously encode all attention patterns. In multi-head models, different heads specialize (one for local patterns, one for positional, one for semantic). With one head, the SVD spectrum reflects a superposition of all these functions, making it harder to isolate the jumper-rule signal.

**Prevention:**
1. Accept this as a known limitation and design the analysis to account for it. The SVD metrics should be evaluated against the hypothesis that they detect the *composite* instability of a single-head model, not a clean separation of rule-learning from other functions.
2. Use the k-subspace metrics (k=2,4,8) to capture multi-component signals. A single principal component may not capture the rule signal, but the top-4 subspace might.
3. As a validation experiment (Tier 3): run one config with 4 heads for comparison. If the single-head AUROC is lower than the 4-head per-head AUROC, the bottleneck is real. This is not in scope for the main result but is good supplementary evidence.
4. d_model=128 with 1 head means d_k = d_v = 128. This is much larger than typical (d_k=64 in standard transformers). The large key dimension may help compensate for the single-head bottleneck.

**Detection:** If SVD singular value entropy is very high (near maximum entropy = log(w)), the QK^T matrix is "full rank" and not specializing. This suggests the attention is diffuse and the single head is struggling.

**Phase:** Architecture Design (Phase 2), Analysis Interpretation (Phase 4).

---

### Pitfall 14: Incorrect Baseline Invalidates AUROC Comparisons

**What goes wrong:** The predictive horizon AUROC is computed as "can this SVD metric discriminate pre-failure steps from baseline?" But what is the baseline? Options: (a) all non-failure steps in the same walk, (b) corresponding positions in non-jumper walks, (c) random steps from random walks. Each baseline gives different AUROC values, and using the wrong one can inflate or deflate the result.

**Prevention:**
1. Use a hierarchical baseline strategy:
   - **Primary baseline**: position-matched steps from non-jumper walks (controls for position effects)
   - **Secondary baseline**: non-failure steps from the same walk (controls for walk-specific effects but confounded with position)
   - **Null baseline**: shuffled labels (establishes chance level)
2. Report AUROC for all three baselines. If they disagree substantially, the signal is confounded and the section must discuss why.
3. Never use "all other steps" as a baseline — this mixes positional, walk-level, and model-state effects.

**Detection:** If AUROC vs. null baseline (shuffled) exceeds 0.6, something is wrong with the experimental setup.

**Phase:** Analysis Pipeline (Phase 4). Baseline strategy must be specified before any AUROC is computed.

---

## Minor Pitfalls

---

### Pitfall 15: Graph Generation Seed Conflated With Training Seed

**What goes wrong:** If the same seed controls both graph generation and training initialization, changing the seed changes both the graph topology and the model initialization simultaneously. This confounds "graph effects" with "training effects" and prevents attributing results to either.

**Prevention:** Use separate seed sequences: `graph_seed = seed`, `walk_seed = seed + 1000`, `model_seed = seed + 2000`. This allows varying one while holding others constant.

**Phase:** Infrastructure (Phase 1).

---

### Pitfall 16: RunPod Instance Preemption Loses Partial Results

**What goes wrong:** Spot instances on RunPod can be preempted mid-run. If results are only written at the end of a run, all progress is lost.

**Prevention:**
1. Checkpoint training state every N steps (N = enough to limit re-computation to < 10 minutes).
2. Write intermediate results (training curves, gate metrics) incrementally, not at the end.
3. Use on-demand instances for Tier 1 (core r-sweep) to avoid preemption risk.
4. Sync results to persistent storage (RunPod network volume or rsync to local machine) after each completed run.

**Phase:** Infrastructure (Phase 1).

---

### Pitfall 17: result.json Token Metrics Schema Mismatch

**What goes wrong:** The RESULTS_SCHEMA.md was written for a generic LLM hallucination detection project (it references GSM8K, Llama-3, temperature parameters). The DCSBM project has a different token metric structure: 20+ SVD metrics per step instead of just logprobs and entropy. If the schema is not adapted, the plotting and reporting code will break or produce wrong results.

**Prevention:**
1. Extend the result.json schema for this project before writing any collection code. Add a `token_metrics` dict in each sequence entry keyed by metric name, each containing a list of floats aligned with the tokens array.
2. Validate schema consistency: every sequence in result.json must have the same set of metric keys, and each metric array must be the same length as the tokens array.
3. The four-class hallucination outcome (edge_valid/invalid x rule_followed/violated/na) should replace the binary hallucinated/correct label from the generic schema.

**Detection:** Schema validation function that checks all these invariants. Run it before any plotting or analysis.

**Phase:** Infrastructure (Phase 1). Schema must be finalized before any code writes result.json.

---

### Pitfall 18: Degree Correction Distribution Choice Affects Everything

**What goes wrong:** The DCSBM degree correction parameters (theta_i for each vertex) control the expected degree of each vertex. The choice of distribution for these parameters (power law, log-normal, uniform, constant) has outsized effects on graph structure: power law creates a few hub vertices and many low-degree vertices, while uniform creates a more homogeneous graph. The spec does not specify which distribution to use.

**Prevention:**
1. Use a bounded distribution: Uniform[0.5, 2.0] or a truncated log-normal. This provides meaningful degree heterogeneity without creating pathological hubs or isolates.
2. Treat the degree correction distribution as a fixed choice (not a sweep parameter). Sweeping it adds a dimension with unclear scientific value and burns budget.
3. Document the choice and its rationale in the config. The degree correction is a nuisance parameter for this experiment, not a variable of interest.

**Detection:** After graph generation, check the degree distribution. If max_degree / median_degree > 20, the degree correction is too aggressive.

**Phase:** Graph Generation (Phase 1).

---

### Pitfall 19: Context Window Padding Poisoning SVD Metrics

**What goes wrong:** At the beginning of a walk (positions 0 to w-1), the context window is not full. The QK^T matrix for position 5 in a walk only has 5 meaningful tokens; the remaining w-5 positions are padding or zero. The SVD of a mostly-zero matrix is numerically degenerate and produces meaningless metrics. If block jumper events occur early in a walk, the "pre-failure" SVD metrics are dominated by padding effects, not attention patterns.

**Prevention:**
1. Only collect SVD metrics for positions >= w (after the context window is fully populated).
2. Alternatively, use causal masking to set QK^T entries for padding positions to -inf before softmax, so they contribute 0 attention. Then the effective QK^T is a submatrix and the SVD should be computed on this submatrix only. But this means the matrix size varies per step, complicating metric comparisons.
3. Simplest approach: require all evaluation walks to be at least 2w tokens long, and only collect SVD metrics starting at position w. This wastes the first w tokens as "warmup" but guarantees a full context window for every collected step.
4. Ensure no block jumper events occur in the first w tokens of any evaluation walk (by construction or by filtering).

**Detection:** Check if any SVD metrics in positions 0 to w-1 have anomalous distributions compared to later positions. If yes, the padding effect is contaminating the data.

**Phase:** SVD Collection Pipeline (Phase 3).

---

### Pitfall 20: Not Storing Enough Metadata for Retrospective Analysis

**What goes wrong:** After running 100+ configs, you discover that a key analysis requires knowing the graph's actual edge density (not just p_in/p_out, but the realized density after DCSBM sampling). Or you need the actual number of block jumper events in each evaluation walk. Or you need the wall-clock time per phase. But none of this was stored, and the runs are done, and re-running them costs money.

**Prevention:**
1. Store derived graph properties in config: actual edge count, actual density, connected component count, diameter, average path length, block sizes, jumper vertex degrees.
2. Store derived corpus properties: token count, vertex frequency distribution, jumper event count per walk.
3. Store per-phase wall times: graph generation, walk generation, training, evaluation, SVD collection, reporting.
4. Store training dynamics: loss at end, edge compliance at end, rule compliance at end, number of training steps, learning rate schedule used.
5. Over-store rather than under-store. An extra 1KB of metadata per config costs nothing compared to the cost of re-running.

**Detection:** Before launching the sweep, write a metadata checklist. After the anchor config completes, verify every item is present in result.json.

**Phase:** Infrastructure (Phase 1). Define the metadata schema before any runs.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Graph Generation | Disconnected graphs, trivial rules, degenerate degree correction | Validate connectivity + path existence + rule difficulty for every generated graph (Pitfalls 1, 2, 18) |
| Walk Generation | Insufficient jumper coverage, walks too short for long-range rules | Coverage report after generation, l >= r + w, biased starts for jumpers (Pitfall 12) |
| Training Pipeline | Failure to reach sufficiency gate, wrong learning rate, insufficient capacity | Anchor config first, LR finder, early abort at 80% budget, architecture scaling rules (Pitfall 6) |
| SVD Collection | Memory blowup, numerical instability, context padding effects, O(d^3) cost | Streaming storage, numerical guards, metrics only for position >= w, profile anchor first (Pitfalls 3, 4, 8, 19) |
| Analysis Pipeline | Multiple comparisons, class imbalance AUROC, position confounds, wrong baseline | Pre-register primary metrics, position-matched controls, Bonferroni correction, hierarchical baselines (Pitfalls 7, 9, 10, 14) |
| Budget Management | Misconfigured runs, no priority ordering, no budget tracking, preemption | Anchor-first calibration, 3-tier priority queue, budget dashboard, checkpointing (Pitfalls 5, 16) |
| Reproducibility | Non-deterministic GPU ops, seed conflation, missing metadata, schema mismatch | Deterministic mode, separate seed sequences, over-store metadata, validate schema (Pitfalls 11, 15, 17, 20) |

---

## Sources

- PyTorch documentation on deterministic algorithms: `torch.use_deterministic_algorithms` requires CUBLAS_WORKSPACE_CONFIG for cuBLAS determinism. Known limitation: `torch.linalg.svd` singular vector ordering for degenerate eigenvalues is implementation-dependent even in deterministic mode. (HIGH confidence — well-documented PyTorch behavior)
- Numerical linear algebra: SVD condition number instability for near-singular matrices is textbook material (Golub & Van Loan, "Matrix Computations"). The eps-clamping strategy is standard practice. (HIGH confidence)
- DCSBM connectivity thresholds: for a graph with n vertices and k blocks, expected average degree must be O(log(n)) for the graph to be connected with high probability (Erdos-Renyi threshold, extends to SBM). For n=200, k=16: need expected degree >= ~5.3. This constrains the p_in/p_out parameter space. (HIGH confidence — established graph theory)
- AUROC class imbalance sensitivity: DeLong's method for AUROC confidence intervals. Sample size requirements for AUROC precision follow standard statistical power analysis formulas. (HIGH confidence)
- Multiple comparisons: Benjamini-Hochberg FDR procedure is standard for large-scale testing. Pre-registration of primary metrics reduces the multiple comparison burden. (HIGH confidence)
- RunPod pricing: RTX 3090 at ~$0.44/hr on-demand, ~$0.22/hr spot; RTX 4090 at ~$0.74/hr on-demand. $100 budget = 135-454 GPU-hours depending on instance type and pricing tier. (MEDIUM confidence — prices may have changed since last verification)
- NanoGPT training dynamics: learning rate of 3e-4 with cosine annealing is the default in the minGPT/nanoGPT codebase for models of this scale. (MEDIUM confidence — based on established practice, not Context7-verified)
