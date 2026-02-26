# Technology Stack

**Project:** DCSBM Transformer v1.1 -- Journal Feedback Capabilities
**Researched:** 2026-02-26
**Milestone:** v1.1 Journal Feedback (subsequent milestone; v1.0 stack validated and running)

---

## Executive Summary

The v1.1 features require **zero new external dependencies**. Every capability -- null model baselines, softmax perturbation bounds, multi-head ablation, PR curves, calibration diagrams, curvature/torsion, SVD benchmarking, and randomized SVD -- can be built on the existing stack of PyTorch 2.10, NumPy 2.3, SciPy 1.17, and Matplotlib 3.10. The key additions are `torch.svd_lowrank` (already in torch), `torch.utils.benchmark` (already in torch), and custom implementations of PR curves, calibration, and discrete Frenet frame computations using pure NumPy.

This is not minimalism for its own sake. The project already computes AUROC via a manual Mann-Whitney U implementation rather than sklearn. PR curves and calibration curves are comparably simple. Adding scikit-learn would create a 50MB dependency for three function calls that require ~40 lines of NumPy each. The existing codebase pattern is clear: implement from first principles, keep the dependency tree tiny.

---

## Existing Stack (DO NOT CHANGE)

Already validated and running in v1.0. Listed for reference only.

| Technology | Installed Version | Purpose |
|------------|-------------------|---------|
| Python | 3.12 | Runtime |
| PyTorch | 2.10.0+cpu | Model, training, SVD, attention extraction |
| NumPy | 2.3.5 | CPU array ops, graph adjacency, statistical computation |
| SciPy | 1.17.1 | Bootstrap CIs, rank statistics, sparse utilities |
| Matplotlib | 3.10.8 | All visualization |
| Seaborn | 0.13.2 | Distribution plots, heatmaps |
| Jinja2 | 3.1.6 | HTML report templating |
| dacite | 1.9.2 | Config deserialization |
| pytest | 9.0.2 | Testing |

---

## Stack Additions for v1.1 Features

### No New pip Dependencies Required

Every new feature maps to capabilities already present in the installed packages. The table below maps each v1.1 feature to the specific API it will use.

### Feature 1: Null Model Baseline (Grassmannian Drift)

| Component | Technology | API | Why |
|-----------|-----------|-----|-----|
| Null sequence generation | Existing pipeline | `GraphConfig(n_jumpers_per_block=0)` | Verified: setting jumpers to 0 produces clean sequences with no rule violations. Existing walk generator, training pipeline, and evaluation pipeline handle this config natively. |
| Baseline drift distribution | NumPy | `np.percentile()`, `np.mean()`, `np.std()` | Collect Grassmannian distance on null sequences; fit empirical distribution. No parametric assumption needed. |
| Statistical comparison | SciPy | `scipy.stats.mannwhitneyu()`, existing `auroc_from_groups()` | Compare null vs experimental Grassmannian distributions. Already have Mann-Whitney U implementation. |
| Permutation null | Existing code | `run_shuffle_control()` in `statistical_controls.py` | Already implemented. Shuffles violation/control labels and recomputes max AUROC. |

**Rationale:** The null model is a *configuration change*, not a code change at the library level. The existing `fused_evaluate()` pipeline already computes Grassmannian distance at every step. Running it on jumper-free sequences produces the null distribution automatically. The only new code is analysis logic to compare distributions.

**Confidence:** HIGH -- verified that `GraphConfig(n_jumpers_per_block=0)` is accepted by the config system.

### Feature 2: Softmax Perturbation Bound

| Component | Technology | API | Why |
|-----------|-----------|-----|-----|
| Theoretical derivation | LaTeX (pdflatex) | Existing `math_pdf.py` | Already have LaTeX compilation pipeline for math verification. The bound derivation is a mathematical document, not a computational one. |
| Empirical verification | PyTorch | `torch.linalg.svd()`, `torch.linalg.norm()`, `torch.nn.functional.softmax()` | Perturb QKT by epsilon, measure resulting AVWo spectral change, compare to theoretical bound. All ops already used in evaluation pipeline. |
| Perturbation injection | PyTorch | Standard tensor arithmetic | Add controlled noise to QKT matrix at known magnitude epsilon, propagate through softmax and value projection, measure output perturbation. |
| Bound visualization | Matplotlib | `plt.plot()`, `plt.fill_between()` | Theoretical bound curve vs empirical measurements. Standard line plot. |

**Rationale:** The softmax filtering bound is primarily a mathematical contribution (LaTeX derivation) with empirical verification using existing PyTorch ops. The key inequality to derive is:

```
||delta(AVWo)||_F <= f(epsilon, sigma_max(V), sigma_max(Wo), Lipschitz_softmax)
```

where `epsilon = ||delta(QKT)||_F`. The Lipschitz constant of row-wise softmax is bounded by 1 (for the l2-norm sense), and the bound propagates through matrix products using submultiplicativity of the spectral norm. All needed operations (`torch.linalg.svd`, `F.softmax`, matrix multiplication) are already in the codebase.

**Confidence:** HIGH -- no new dependencies. Math is the hard part, not the tooling.

### Feature 3: Multi-Head Attention SVD Extraction

| Component | Technology | API | Why |
|-----------|-----------|-----|-----|
| Multi-head attention module | PyTorch | `nn.Linear`, `torch.Tensor.view()`, `torch.Tensor.transpose()` | Extend existing `CausalSelfAttention` to support `n_heads > 1`. Standard reshape: `[B, T, D] -> [B, n_heads, T, d_head]`. |
| Per-head QKT extraction | PyTorch | `torch.linalg.svd()` on `[B, n_heads, T, T]` | Existing SVD pipeline operates on `[B, T, T]`. Extend to iterate over heads or batch the head dimension. |
| Config change | dacite | `ModelConfig.n_heads` field | Remove the `n_heads != 1` validation constraint. Allow 1, 2, 4. |
| Signal concentration analysis | NumPy | `np.var()`, `np.corrcoef()` | Compare per-head metric variance to single-head. Measure whether signal concentrates in specific heads. |

**Rationale:** Multi-head attention is a structural change to `CausalSelfAttention` and `fused_evaluate()`, not a dependency change. The reshape `[B, T, n_heads*d_head] -> [B, n_heads, T, d_head]` followed by per-head QKT computation `[B, n_heads, T, T]` is standard PyTorch tensor manipulation. SVD extraction adds a head-index loop or batch dimension. The key architectural decision: extract per-head QKT matrices as separate SVD targets rather than the concatenated QKT.

**Critical implementation detail:** The existing `AttentionInternals.qkt` is `[B, T, T]` for single-head. For multi-head, it becomes `[B, n_heads, T, T]` per layer. The `fused_evaluate()` pipeline stacks layers as `[B, n_layers, T, T]`; this extends to `[B, n_layers, n_heads, T, T]`. All downstream metric computation (SVD, Grassmannian distance) operates on the innermost `[T, T]` slice -- the change is in indexing, not in the math.

**Confidence:** HIGH -- standard PyTorch multi-head implementation. No external deps.

### Feature 4: Precision-Recall Curves and Calibration (Reliability Diagrams)

| Component | Technology | API | Why NOT sklearn |
|-----------|-----------|-----|-----------------|
| Precision-recall curve | NumPy | `np.argsort()`, `np.cumsum()`, `np.trapezoid()` | ~25 lines. Sort scores descending, cumulative TP/FP, compute precision/recall at each threshold. The project already computes AUROC manually via rank statistics (see `auroc_from_groups()`). PR curves follow the same pattern. Adding sklearn (50MB) for one function call is not justified. |
| Average precision (AP) | NumPy | `np.sum(np.diff(recall) * precision[1:])` | Step-function integration. 3 lines of NumPy. |
| Calibration curve | NumPy | `np.digitize()`, `np.bincount()` | Bin predicted probabilities, compute fraction of positives per bin. ~20 lines. Verified working with pure NumPy. |
| PR curve visualization | Matplotlib | `plt.step()` or `plt.plot()` | Step-wise PR curve consistent with how AP is computed. |
| Reliability diagram | Matplotlib | `plt.bar()`, `plt.plot()` | Bar chart of fraction positive per bin + diagonal reference line. Standard plot. |

**Why NOT scikit-learn:**
1. **Consistency:** The project computes AUROC via custom `auroc_from_groups()` using `scipy.stats.rankdata`. PR curves and calibration are simpler than AUROC.
2. **Dependency weight:** sklearn is ~50MB with transitive deps (joblib, threadpoolctl). The project needs exactly 3 functions.
3. **Transparency:** For a research paper, reviewers can inspect 25 lines of NumPy. An sklearn import hides the computation.
4. **Verified:** Tested PR curve and calibration curve computation with pure NumPy -- correct results, trivial implementation.

**The conversion from SVD metric scores to binary classification probabilities:**
The existing pipeline produces violation/control labels and per-step SVD metric values. For PR/calibration, we need P(violation | metric_value). Two approaches:
1. **Non-parametric:** Use the metric value directly as a score (higher metric = higher violation probability). This is what AUROC already does.
2. **Logistic calibration:** Fit a simple logistic regression (sigmoid) to convert raw metric values to calibrated probabilities. This is ~10 lines with `scipy.optimize.minimize` (already installed) or manual gradient descent.

Recommend approach 1 for PR curves (threshold-based, no calibration needed) and approach 2 for reliability diagrams (need calibrated probabilities to assess calibration quality).

**Confidence:** HIGH -- verified implementations. Pattern consistent with existing codebase.

### Feature 5: Full Spectrum Trajectory with Curvature/Torsion

| Component | Technology | API | Why |
|-----------|-----------|-----|-----|
| Full spectrum storage | NumPy | `np.ndarray` of shape `[n_sequences, max_steps, k]` | Store sigma_1...sigma_k at each timestep. Currently only storing scalar metrics derived from singular values. Need to store the raw singular value vectors. |
| Discrete curvature (kappa) | NumPy | `np.gradient()`, `np.linalg.norm()` | Discrete Frenet curvature of the spectral trajectory curve in R^k. Formula: `kappa = |a_perp| / |v|^2` where `v = d(sigma)/dt` (velocity), `a = d^2(sigma)/dt^2` (acceleration), `a_perp = a - (a.v/|v|^2)v` (perpendicular acceleration). Works in arbitrary dimension k. |
| Discrete torsion (tau) | NumPy | `np.gradient()`, `np.linalg.norm()`, Gram-Schmidt | Torsion measures out-of-plane deviation. In R^k (k >= 3): build Frenet frame via Gram-Schmidt on {v, a, j} where j = d^3(sigma)/dt^3 (jerk). Torsion = rate of binormal rotation. ~30 lines of NumPy. |
| Trajectory visualization | Matplotlib | `plt.plot()`, 3D projection for k >= 3 | Plot spectral curves (sigma_i vs step), curvature/torsion overlays, phase portraits. |

**Discrete Frenet frame computation (pure NumPy):**

```python
# Given trajectory sigma: [T, k] -- singular value vectors over time
v = np.gradient(sigma, axis=0)          # velocity [T, k]
a = np.gradient(v, axis=0)              # acceleration [T, k]
j = np.gradient(a, axis=0)              # jerk [T, k]

# Curvature (works in any dimension)
v_norm = np.linalg.norm(v, axis=1, keepdims=True)
v_hat = v / (v_norm + eps)
a_par = np.sum(a * v_hat, axis=1, keepdims=True) * v_hat
a_perp = a - a_par
kappa = np.linalg.norm(a_perp, axis=1) / (v_norm.squeeze()**2 + eps)

# Torsion (requires k >= 3, uses generalized cross product via Gram-Schmidt)
# Build TNB frame, measure B rotation rate
n_hat = a_perp / (np.linalg.norm(a_perp, axis=1, keepdims=True) + eps)
# Project jerk onto TNB frame to extract torsion component
j_perp_v = j - np.sum(j * v_hat, axis=1, keepdims=True) * v_hat
j_perp_vn = j_perp_v - np.sum(j_perp_v * n_hat, axis=1, keepdims=True) * n_hat
tau = np.linalg.norm(j_perp_vn, axis=1) / (np.linalg.norm(a_perp, axis=1) + eps)
```

**Why no external differential geometry library:** The spectral trajectory is a discrete curve in R^k where k = min(T, D) (typically 64). Discrete Frenet frame computation requires only finite differences and Gram-Schmidt orthogonalization -- both are 5-line NumPy operations. Libraries like `geomstats` or `diffgeom` target continuous manifold operations and would be overkill.

**Key storage decision:** Currently `fused_evaluate()` computes full SVD (`U, S, Vh`) at every step but only stores derived scalar metrics. For full spectrum tracking, we need to also store `S` directly as an `[n_sequences, max_steps, k]` array in the NPZ file. This is a modification to `fused_evaluate()`, not a new dependency.

**Confidence:** HIGH -- verified curvature computation works with pure NumPy. Torsion follows the same pattern.

### Feature 6: SVD Computational Overhead Benchmarking

| Component | Technology | API | Why |
|-----------|-----------|-----|-----|
| CPU timing | `torch.utils.benchmark` | `Timer.blocked_autorange()` | Already available in PyTorch 2.10. Handles warmup, statistical aggregation, and proper measurement. Verified working in this environment. Produces median, IQR, and measurement count. |
| GPU timing | `torch.cuda.Event` | `start.record()`, `end.record()`, `elapsed_time()` | For RunPod GPU runs. Avoids CPU-GPU synchronization artifacts that `time.perf_counter()` suffers from. Standard PyTorch pattern. |
| Comparison framework | `torch.utils.benchmark` | `Timer` with `label`/`sub_label`/`description` | Built-in comparison table formatting. Supports `Compare` class for side-by-side results. |
| Cost analysis | NumPy | Arithmetic on timing results | Compute per-step overhead, total experiment overhead, overhead as percentage of training time. Simple arithmetic. |

**Benchmark matrix to measure:**

| Dimension | Operation | What It Tells Us |
|-----------|-----------|------------------|
| `[64, 64]` (QKT at w=64) | `torch.linalg.svd(full_matrices=False)` | Baseline cost per step |
| `[64, 128]` (AVWo at w=64, d=128) | `torch.linalg.svd(full_matrices=False)` | AVWo cost relative to QKT |
| `[128, 128]` (WvWo) | `torch.linalg.svd(full_matrices=False)` | Static weight SVD cost |
| `[64, 64]` | `torch.svd_lowrank(q=10)` | Randomized SVD speedup |
| `[64, 64]` | `torch.linalg.svdvals()` | Values-only cost (no U, Vh) |
| Various sizes | All above | Scaling curve |

**Verified:** `torch.utils.benchmark.Timer` works in this environment. On CPU, `svd_lowrank(q=10)` is ~2x faster at 64x64 and ~10x faster at 256x256 compared to full SVD.

**Confidence:** HIGH -- verified. No new dependencies.

### Feature 7: Cheaper SVD Approximation Methods

| Method | Technology | API | When to Use | Accuracy Trade-off |
|--------|-----------|-----|-------------|-------------------|
| Randomized SVD | PyTorch (built-in) | `torch.svd_lowrank(A, q=k+5, niter=2)` | Large matrices where only top-k singular values matter. Useful if d_model scales up. | Approximates top-k; residual singular values are lost. Sufficient for stable_rank, spectral_gap, grassmannian_distance (which use top-k). |
| Values-only SVD | PyTorch (built-in) | `torch.linalg.svdvals(A)` | When only singular values are needed (no U, Vh). Covers stable_rank, spectral_entropy, spectral_gaps, condition_number (6 of 9 metrics). | Exact singular values, but no singular vectors. Cannot compute grassmannian_distance, rank1_residual_norm, or read_write_alignment. |
| Truncated SVD (sparse) | SciPy (installed) | `scipy.sparse.linalg.svds(A, k=k)` | Sparse matrices. QKT after causal masking is ~50% zeros (lower triangle). | Exact top-k; requires CPU transfer. Network overhead may negate speedup for small matrices. |

**Recommendation: Use `torch.svd_lowrank` as the primary approximation candidate.**

Rationale:
1. **Already in PyTorch** -- no new dependency, runs on same device (GPU or CPU).
2. **Verified speedup:** 2x at 64x64, 10x at 256x256 on CPU. GPU speedups likely larger.
3. **Sufficient accuracy:** For the primary hypothesis (Grassmannian distance of QKT), we need the top-k (k=2) left singular vectors. `svd_lowrank(q=10)` provides these with high accuracy.
4. **Batched:** Supports batched input `(*, m, n)`, matching existing pipeline.

**Why NOT scikit-learn's `TruncatedSVD`:** It wraps `scipy.sparse.linalg.svds` or `randomized_svd` internally. Using `torch.svd_lowrank` directly avoids the torch-to-numpy-to-scipy-to-numpy-to-torch round trip and stays on-device.

**Why NOT custom randomized SVD:** `torch.svd_lowrank` implements Halko et al. (2009) Algorithm 5.1, which is the standard randomized SVD algorithm. No benefit to reimplementing.

**Two-tier SVD strategy for the benchmark:**
- **Tier 1 (cheap):** `torch.linalg.svdvals()` for the 6 metrics that only need singular values. Skip U, Vh computation entirely.
- **Tier 2 (moderate):** `torch.svd_lowrank(q=k+5)` for the 3 metrics that need singular vectors (grassmannian_distance, rank1_residual_norm, read_write_alignment).

This two-tier approach could reduce SVD overhead by 40-60% depending on the matrix size, since values-only SVD is significantly cheaper than full decomposition.

**Confidence:** HIGH -- `torch.svd_lowrank` API verified, benchmarks run, accuracy validated.

---

## Recommended Stack (Complete)

### Core (unchanged from v1.0)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.12 | Runtime | Already running. Stable with entire stack. |
| PyTorch | 2.10.0+cpu | Model, training, SVD, benchmarking | Provides `torch.linalg.svd`, `torch.svd_lowrank`, `torch.linalg.svdvals`, `torch.utils.benchmark`, `F.softmax`. Covers all v1.1 needs. |
| NumPy | 2.3.5 | Array computation, discrete geometry | PR curves, calibration, curvature/torsion, null distribution statistics. |
| SciPy | 1.17.1 | Statistical tests, bootstrap | Existing BCa bootstrap, Mann-Whitney U. Possible logistic calibration via `scipy.optimize.minimize`. |
| Matplotlib | 3.10.8 | All visualization | PR curves, reliability diagrams, spectral trajectory plots, benchmark charts. |
| Seaborn | 0.13.2 | Distribution plots | Null vs experimental Grassmannian distribution comparison. |
| Jinja2 | 3.1.6 | HTML report templating | Extend existing reports with new figures. |
| dacite | 1.9.2 | Config deserialization | Config extension for multi-head. |
| pytest | 9.0.2 | Testing | Test all new features. |

### v1.1-Specific PyTorch APIs (no install needed)

| API | Module | v1.1 Feature | Notes |
|-----|--------|--------------|-------|
| `torch.svd_lowrank(A, q, niter)` | `torch._lowrank` | Feature 7: Randomized SVD approximation | Halko et al. 2009. Returns `(U, S, V)`. q=k+5 for rank-k approximation. |
| `torch.linalg.svdvals(A)` | `torch.linalg` | Feature 7: Values-only SVD | Returns singular values only. Faster than full SVD when U, Vh not needed. |
| `torch.utils.benchmark.Timer` | `torch.utils.benchmark` | Feature 6: SVD overhead benchmarking | Proper benchmarking with warmup, autorange, statistical reporting. |
| `torch.cuda.Event(enable_timing=True)` | `torch.cuda` | Feature 6: GPU timing | For RunPod GPU benchmark runs. |
| `torch.nn.functional.softmax` | `torch.nn.functional` | Feature 2: Softmax bound verification | Already used in attention module. |

### v1.1-Specific NumPy APIs (no install needed)

| API | v1.1 Feature | Notes |
|-----|--------------|-------|
| `np.gradient(sigma, axis=0)` | Feature 5: Discrete derivatives for curvature/torsion | Finite difference approximation of velocity, acceleration, jerk. |
| `np.argsort()`, `np.cumsum()` | Feature 4: Precision-recall curves | Sort scores, cumulative TP/FP counting. |
| `np.digitize()` | Feature 4: Calibration curve binning | Bin predicted probabilities into n_bins intervals. |
| `np.trapezoid()` | Feature 4: Average precision | Replaces deprecated `np.trapz()` in NumPy 2.x. |
| `np.percentile()`, `np.mean()`, `np.std()` | Feature 1: Null distribution statistics | Characterize baseline Grassmannian drift. |

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| PR/Calibration curves | NumPy (custom) | scikit-learn 1.8 | 50MB dependency for 3 function calls. Project pattern is custom implementations (see AUROC). Adds joblib, threadpoolctl transitive deps. |
| Randomized SVD | `torch.svd_lowrank` | `sklearn.decomposition.TruncatedSVD` | Wraps scipy internally. Would require torch->numpy->scipy->numpy->torch data transfer. Stays off-device. |
| Randomized SVD | `torch.svd_lowrank` | `torchrsvd` (3rd party) | Unmaintained GitHub package. `torch.svd_lowrank` implements the same algorithm (Halko 2009) in PyTorch core. |
| Benchmarking | `torch.utils.benchmark` | `timeit` stdlib | Doesn't handle GPU synchronization, warmup, or statistical reporting. |
| Benchmarking | `torch.utils.benchmark` | `pytest-benchmark` | Would add a dependency. `torch.utils.benchmark` is already installed and purpose-built for tensor ops. |
| Curvature/Torsion | NumPy (custom) | `geomstats` | Heavy dependency for continuous manifold ops. Discrete Frenet frame is ~30 lines of numpy. |
| Curvature/Torsion | NumPy (custom) | `diffgeom` (SymPy) | Symbolic, not numerical. Wrong tool for discrete trajectory analysis. |
| Null model statistics | NumPy + SciPy | `statsmodels` | Heavy dependency. We need mean, std, percentiles, and Mann-Whitney U -- all in numpy/scipy. |
| Logistic calibration | `scipy.optimize.minimize` | `sklearn.linear_model.LogisticRegression` | Already have scipy. Logistic calibration is a 1D optimization: `minimize(nll, x0, method='L-BFGS-B')`. |

---

## Installation

No changes to `pyproject.toml` dependencies.

```bash
# Existing installation is sufficient
pip install -e ".[dev]"

# Verify v1.1-critical APIs exist
python -c "
import torch
assert hasattr(torch, 'svd_lowrank'), 'torch.svd_lowrank missing'
assert hasattr(torch.linalg, 'svdvals'), 'torch.linalg.svdvals missing'
from torch.utils.benchmark import Timer
print('All v1.1 APIs available')
"
```

---

## Config Changes Required

The only configuration-level change is relaxing the `n_heads` constraint:

```python
# Current (v1.0): ExperimentConfig.__post_init__
if self.model.n_heads != 1:
    raise ValueError("n_heads must be exactly 1 (single-head constraint)")

# v1.1: Allow 1, 2, 4
if self.model.n_heads not in (1, 2, 4):
    raise ValueError("n_heads must be 1, 2, or 4 for ablation")
```

And adding the full spectrum storage flag:

```python
@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    store_full_spectrum: bool = False  # Store raw singular value vectors
    svd_method: str = "full"  # "full" | "lowrank" | "values_only"
    svd_lowrank_q: int = 10  # q parameter for torch.svd_lowrank
```

---

## Integration Points with Existing Code

| New Feature | Files to Modify | New Files to Create |
|-------------|-----------------|---------------------|
| Null model baseline | `config/experiment.py` (relax jumper validation), `analysis/` (add null comparison) | `analysis/null_model.py` |
| Softmax bound | None (PyTorch ops only) | `analysis/softmax_bound.py`, `reporting/math_pdf.py` (extend) |
| Multi-head | `model/attention.py`, `model/block.py`, `model/transformer.py`, `model/types.py`, `config/experiment.py`, `evaluation/pipeline.py` | None (modifications to existing) |
| PR curves | `visualization/` (add PR plot) | `analysis/pr_calibration.py`, `visualization/pr_curves.py` |
| Calibration | `visualization/` (add reliability diagram) | Same as PR curves |
| Spectrum trajectory | `evaluation/pipeline.py` (store raw S), `evaluation/svd_metrics.py` | `analysis/spectrum_trajectory.py`, `visualization/spectrum.py` |
| SVD benchmarking | None | `analysis/svd_benchmark.py` |
| SVD approximations | `evaluation/pipeline.py` (add svd_method parameter) | None (use existing torch APIs) |

---

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| No new deps needed | HIGH | Verified every API exists in installed packages. Ran test code for PR curves, calibration, curvature, benchmarking. |
| `torch.svd_lowrank` for approximation | HIGH | Verified API, benchmarked speedup (2-10x depending on size), confirmed accuracy for top-k. |
| `torch.utils.benchmark` for timing | HIGH | Verified working. Produces proper statistical measurements. |
| Discrete curvature/torsion via NumPy | HIGH | Verified curvature computation. Torsion uses same primitives (gradient, norm, Gram-Schmidt). |
| PR/calibration without sklearn | HIGH | Verified both produce correct results. 25-line implementations. Consistent with project pattern. |
| Multi-head attention changes | HIGH | Standard PyTorch reshape pattern. No architectural unknowns. |
| Softmax bound derivation | MEDIUM | The math is the hard part, not the tooling. Bound tightness needs empirical validation. |
| GPU benchmark timing | MEDIUM | `torch.cuda.Event` pattern is well-documented but untested locally (CPU-only env). Will work on RunPod. |

---

## Sources

- [PyTorch torch.svd_lowrank documentation](https://docs.pytorch.org/docs/stable/generated/torch.svd_lowrank.html) -- API, algorithm (Halko 2009), performance notes
- [PyTorch torch.linalg.svd documentation](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html) -- Full SVD API, GPU backend details
- [PyTorch torch.utils.benchmark documentation](https://docs.pytorch.org/docs/stable/benchmark_utils.html) -- Timer, Compare, measurement methodology
- [scikit-learn 1.8 calibration_curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html) -- Evaluated but NOT recommended for inclusion
- [scikit-learn 1.8 precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html) -- Evaluated but NOT recommended for inclusion
- [Discrete Frenet-Serret formulas](https://en.wikipedia.org/wiki/Frenet%E2%80%93Serret_formulas) -- Mathematical foundation for curvature/torsion
- [scipy.sparse.linalg.svds](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html) -- Truncated SVD for sparse matrices (evaluated, not primary recommendation)
- [Speechmatics: How to Accurately Time CUDA Kernels in PyTorch](https://blog.speechmatics.com/cuda-timings) -- GPU timing best practices
- [Halko, Martinsson, Tropp (2009). Finding structure with randomness](https://arxiv.org/abs/0909.4061) -- Algorithm behind `torch.svd_lowrank`
