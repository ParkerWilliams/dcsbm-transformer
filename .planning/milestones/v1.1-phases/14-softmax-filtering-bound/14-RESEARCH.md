# Phase 14: Softmax Filtering Bound - Research

**Researched:** 2026-02-26
**Domain:** Theoretical spectral perturbation bounds, softmax Lipschitz analysis, controlled perturbation experiments
**Confidence:** HIGH

## Summary

Phase 14 formalizes the relationship between QK^T perturbation and downstream AVWo spectral change as a chain of operator norm bounds, then empirically verifies the bound via controlled perturbation injection. The derivation proceeds through three stages: (1) QK^T perturbation through softmax using the Lipschitz constant of softmax (1/2 per row for the Jacobian spectral norm) and the 1/sqrt(d_k) scaling factor, (2) softmax output perturbation through AV multiplication using submultiplicativity of the spectral norm, (3) AV perturbation through W_o projection. The three stage bounds compose into an end-to-end epsilon-bound.

The empirical verification injects perturbations of controlled magnitude (fractions of ||QK^T||_F) in both random and adversarial directions (aligned with top singular vector of QK^T), measures the actual AVWo spectral change via ||Delta sigma(AVWo)||_2, and compares against the theoretical bound. The success criterion is fewer than 5% of perturbations exceeding the theoretical bound.

**Primary recommendation:** Implement derivation as a standalone LaTeX document (docs/softmax_bound.tex) and perturbation experiments as a new analysis module (src/analysis/perturbation_bound.py) with visualization (src/visualization/perturbation_bound.py).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Full chain derivation with intermediate bounds at each stage (QK^T -> softmax, softmax -> AV, AV -> AVWo), then composed into end-to-end bound
- Standalone .tex file (docs/softmax_bound.tex) compiling to PDF -- academic-style document
- Standard academic notation (W_Q, W_K, etc.) self-contained without codebase-specific naming
- Include tightness analysis section discussing when the bound is tight vs loose
- Perturbation magnitudes sized as fractions of ||QK^T|| (1%, 5%, 10%, 25%) -- relative scaling
- Adversarial direction = alignment with top singular vector of QK^T
- Random directions as baseline comparison alongside adversarial
- Success criterion: fewer than 5% of perturbations exceed the theoretical bound

### Claude's Discretion
- Injection step strategy (event-aligned, uniform, or both)
- Number of random perturbation directions per (step, magnitude) pair
- Bound tightness visualization style (scatter + envelope, violin/box, or other)
- Chain decomposition strategy in experiments -- whether to track intermediate stages or only end-to-end
- Publication figure quality and layout choices

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SFTX-01 | LaTeX derivation of epsilon-bound from QK^T through softmax to AVWo spectral change, incorporating Lipschitz constant 1/2 and 1/sqrt(d_k) scaling | Chain rule on operator norms; softmax Jacobian spectral norm bounded by 1/2 (Gao & Pavel 2017); submultiplicativity of spectral norm for matrix products |
| SFTX-02 | Empirical verification via controlled perturbation injection (random + adversarial) with fewer than 5% exceeding the bound | Inject epsilon * direction into QK^T at selected steps; recompute softmax, AV, AVWo; measure ||Delta sigma(AVWo)||; compare vs theoretical bound |
| SFTX-03 | Bound tightness visualization showing theoretical envelope vs empirical measurements, with tightness ratio reported | Scatter plot of (epsilon, actual_change) with theoretical bound as upper envelope curve; report median(empirical) / bound as tightness ratio |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch.linalg.svd | (PyTorch) | SVD for measuring spectral change | Already used throughout project |
| torch.linalg.svdvals | (PyTorch) | Singular values without U/Vh for speed | Already used in benchmarks |
| numpy | (existing) | Array operations for experiment results | Already used throughout |
| matplotlib | (existing) | Bound tightness visualization | Already used throughout |
| subprocess/pdflatex | (system) | LaTeX compilation | Same pattern as math_pdf.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| seaborn | (existing) | Consistent styling | Used via style.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual LaTeX file | Extend math_pdf.py | The softmax bound is a standalone derivation; adding to math_pdf.py would bloat the verification document |
| Perturbation in eval pipeline | Standalone analysis module | Must not modify evaluation pipeline; perturbation experiments are separate |

**Installation:** No new packages needed -- all dependencies already in project.

## Architecture Patterns

### Pattern 1: LaTeX Document Generation

The project already has `src/reporting/math_pdf.py` which generates LaTeX via Jinja2 and compiles with pdflatex. However, the softmax bound derivation is a standalone academic document, not a code verification document. It should be:
- Placed at `docs/softmax_bound.tex` (raw LaTeX, not Jinja2-templated)
- Self-contained with standard academic formatting (article class, amsmath, theorem environments)
- Compilable independently: `pdflatex docs/softmax_bound.tex`

### Pattern 2: Perturbation Experiment as Standalone Analysis Module

Following the pattern of `src/analysis/null_model.py` and `src/analysis/svd_benchmark.py`:
- New file: `src/analysis/perturbation_bound.py`
- Takes a trained model + eval data as input
- Performs perturbation injection at selected steps
- Returns structured dict for result.json storage
- Does NOT modify the evaluation pipeline

```python
def run_perturbation_experiment(
    model: nn.Module,
    eval_walks: np.ndarray,
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    device: torch.device,
    magnitudes: list[float] = [0.01, 0.05, 0.10, 0.25],
    n_random_directions: int = 20,
    n_steps: int = 50,
    seed: int = 42,
) -> dict:
    """Run controlled perturbation experiments."""
    ...
```

### Pattern 3: Visualization Module

Following `src/visualization/calibration.py` and `src/visualization/svd_benchmark.py`:
- New file: `src/visualization/perturbation_bound.py`
- `plot_bound_tightness(results)` -> plt.Figure
- Uses PALETTE, apply_style(), save_figure() from style.py
- Integrated into render.py with try/except block
- Figure prefix: `perturbation_bound_*`

### Pattern 4: Result Storage

Following the schema pattern established in Phases 12 and 13:
- Store results in `result.json["metrics"]["perturbation_bound"]`
- Backward-compatible schema validation in `schema.py`
- Optional block -- only validated when present

```json
{
  "metrics": {
    "perturbation_bound": {
      "config": {
        "magnitudes": [0.01, 0.05, 0.10, 0.25],
        "n_random_directions": 20,
        "n_steps": 50,
        "seed": 42
      },
      "theoretical_bound_formula": "epsilon * (1/2) * ||V||_2 * ||W_o||_2 / sqrt(d_k)",
      "by_magnitude": {
        "0.01": {
          "adversarial": {
            "mean_ratio": 0.15,
            "max_ratio": 0.42,
            "n_exceeding_bound": 0,
            "n_total": 50
          },
          "random": {
            "mean_ratio": 0.08,
            "max_ratio": 0.25,
            "n_exceeding_bound": 0,
            "n_total": 1000
          }
        }
      },
      "tightness_ratio": 0.35,
      "violation_rate": 0.0,
      "bound_verified": true
    }
  }
}
```

### Anti-Patterns to Avoid
- **Modifying the evaluation pipeline:** Perturbation experiments are standalone, not integrated into fused_evaluate
- **Using torch.autograd for Jacobian:** The Lipschitz bound is analytical, not numerical. The derivation uses the known 1/2 bound for softmax Jacobian spectral norm
- **Computing tight bounds per-instance:** The theoretical bound is a worst-case (uniform) bound. Per-instance tightness is measured empirically but the bound itself is derived once
- **Storing all perturbation vectors:** Only store summary statistics (mean ratio, max ratio, count exceeding), not individual perturbation results

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SVD computation | Manual eigendecomposition | torch.linalg.svd | Numerically stable, GPU-accelerated |
| Softmax Lipschitz constant | Empirical estimation | Analytical result: 1/2 | Well-known theoretical result (Gao & Pavel 2017) |
| Random direction generation | Manual normalization | torch.randn + normalize | Standard approach for uniform sampling on unit sphere |
| LaTeX compilation | Custom build system | subprocess + pdflatex | Same pattern as math_pdf.py |

## Common Pitfalls

### Pitfall 1: Softmax Jacobian Bound Applies Per-Row
**What goes wrong:** Applying the 1/2 Lipschitz constant to the entire matrix rather than per-row.
**Why it happens:** Softmax operates independently on each row of the QK^T matrix. The Jacobian spectral norm bound of 1/2 applies to each row's softmax independently.
**How to avoid:** The bound for the full attention matrix A = softmax(QK^T/sqrt(d_k)) uses: ||Delta A||_F <= (1/2) * ||Delta (QK^T / sqrt(d_k))||_F = (epsilon / (2 * sqrt(d_k))) when epsilon is measured in QK^T Frobenius norm. Use Frobenius norm throughout for consistency.
**Warning signs:** Bound that is always vacuous (>1) even for small perturbations.

### Pitfall 2: Adversarial Direction Must Be Normalized
**What goes wrong:** Perturbation magnitude not properly controlled.
**Why it happens:** The adversarial direction (top singular vector outer product) must be normalized so that ||Delta QK^T||_F = epsilon * ||QK^T||_F.
**How to avoid:** Compute u_1 * v_1^T from QK^T SVD, normalize to unit Frobenius norm, then scale by epsilon * ||QK^T||_F.
**Warning signs:** Adversarial perturbations that are much larger or smaller than intended.

### Pitfall 3: Spectral Change Measurement
**What goes wrong:** Comparing singular values from different-sized matrices, or comparing unordered singular value vectors.
**Why it happens:** AVWo is [T, D] while QK^T is [T, T]; singular value vectors may have different lengths.
**How to avoid:** Compare ||sigma(AVWo_perturbed) - sigma(AVWo_original)||_2, where sigma denotes the sorted singular value vector. Both come from the same matrix shape so the comparison is well-defined.
**Warning signs:** Negative spectral changes or changes that don't monotonically increase with epsilon.

### Pitfall 4: Causal Mask Interaction
**What goes wrong:** Perturbing future positions in QK^T that are masked to -inf.
**Why it happens:** The QK^T matrix has a causal mask applied before softmax.
**How to avoid:** Only perturb the lower-triangular (visible) portion of QK^T. Zero out perturbation in the upper triangle before adding to QK^T.
**Warning signs:** NaN or Inf values after softmax when perturbation pushes masked values away from -inf.

### Pitfall 5: Scale Factor Placement
**What goes wrong:** Applying 1/sqrt(d_k) in the wrong place in the bound chain.
**Why it happens:** The scaling happens before softmax: softmax(QK^T / sqrt(d_k)). When perturbing QK^T by epsilon, the effective input perturbation to softmax is epsilon / sqrt(d_k).
**How to avoid:** Be explicit about whether epsilon measures perturbation to QK^T or to QK^T/sqrt(d_k). The CONTEXT.md says "fractions of ||QK^T||", meaning we perturb the unscaled QK^T and the 1/sqrt(d_k) appears in the bound.
**Warning signs:** Bound that doesn't match empirical measurements by a factor of sqrt(d_k).

## Derivation Outline

### Stage 1: QK^T -> Softmax

Let S = QK^T / sqrt(d_k) be the scaled score matrix. Softmax acts row-wise: A_i = softmax(S_i) for each row i.

The Jacobian of softmax at point p is J = diag(p) - p*p^T. Its spectral norm satisfies ||J||_2 <= 1/2 (tight when p is uniform).

For a perturbation Delta_S to row S_i:
||Delta A_i||_2 <= (1/2) * ||Delta S_i||_2

Aggregating over rows (Frobenius norm):
||Delta A||_F <= (1/2) * ||Delta S||_F = (1/2) * ||Delta QK^T||_F / sqrt(d_k)

If epsilon = ||Delta QK^T||_F / ||QK^T||_F (relative perturbation), then:
||Delta A||_F <= epsilon * ||QK^T||_F / (2 * sqrt(d_k))

### Stage 2: Softmax -> AV

AV = A @ V where V is [T, D].

Delta(AV) = (A + Delta A) @ V - A @ V = Delta A @ V

||Delta(AV)||_F <= ||Delta A||_F * ||V||_2

(submultiplicativity of Frobenius norm with spectral norm)

### Stage 3: AV -> AVWo

AVWo = AV @ W_o^T

Delta(AVWo) = (AV + Delta(AV)) @ W_o^T - AV @ W_o^T = Delta(AV) @ W_o^T

||Delta(AVWo)||_F <= ||Delta(AV)||_F * ||W_o||_2

### End-to-End Bound

Composing all three stages:

||Delta(AVWo)||_F <= epsilon * ||QK^T||_F * ||V||_2 * ||W_o||_2 / (2 * sqrt(d_k))

For the spectral change (change in top singular value):
|Delta sigma_1(AVWo)| <= ||Delta(AVWo)||_2 <= ||Delta(AVWo)||_F

So the spectral change bound is:
|Delta sigma_1(AVWo)| <= epsilon * ||QK^T||_F * ||V||_2 * ||W_o||_2 / (2 * sqrt(d_k))

### Tightness Analysis

The bound is tightest when:
1. Softmax output is close to uniform (low-entropy attention) -- Jacobian spectral norm approaches 1/2
2. V has a strong dominant singular direction aligned with Delta A
3. W_o amplifies the perturbation direction

The bound is loosest when:
1. Softmax output is peaked (high-entropy) -- effective Lipschitz constant much less than 1/2
2. V is well-conditioned (distributes perturbation across many directions)
3. W_o attenuates the perturbation direction

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Quick run command | `pytest tests/test_perturbation_bound.py -x -v` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SFTX-01 | LaTeX derivation compiles | integration | `pdflatex docs/softmax_bound.tex` | Wave 0 |
| SFTX-02 | Perturbation experiments with bound verification | unit | `pytest tests/test_perturbation_bound.py -x` | Wave 0 |
| SFTX-03 | Bound tightness visualization | unit | `pytest tests/test_perturbation_bound.py -x` | Wave 0 |

### Wave 0 Gaps
- [ ] `docs/softmax_bound.tex` -- standalone LaTeX derivation
- [ ] `src/analysis/perturbation_bound.py` -- perturbation experiment module
- [ ] `src/visualization/perturbation_bound.py` -- bound tightness plots
- [ ] `tests/test_perturbation_bound.py` -- tests for perturbation experiments

## Implementation Notes

### File Organization (New Files)
```
docs/softmax_bound.tex                    # Standalone LaTeX derivation (SFTX-01)
src/analysis/perturbation_bound.py        # Perturbation experiment module (SFTX-02)
src/visualization/perturbation_bound.py   # Bound tightness visualization (SFTX-03)
tests/test_perturbation_bound.py          # Tests for perturbation experiments
```

### Files to Modify
```
src/visualization/render.py              # Add render hooks for perturbation bound plots
src/reporting/single.py                  # Add figure collection + template vars
src/reporting/templates/single_report.html  # Add collapsible section
src/results/schema.py                    # Add validation for perturbation_bound block
```

## Open Questions

1. **Frobenius vs spectral norm in bound**
   - What we know: Both ||Delta||_F and ||Delta||_2 are valid choices; Frobenius gives a tighter per-element bound but spectral gives a tighter directional bound
   - What's unclear: Which is more natural for the user's purpose
   - Recommendation: Use Frobenius norm throughout the derivation (consistent with row-aggregated softmax bound), but report spectral change as ||Delta sigma||_2 (standard in perturbation theory)

2. **Step selection for perturbation injection**
   - What we know: Context says "event-aligned, uniform, or both" are all options
   - What's unclear: Which strategy yields the most informative results
   - Recommendation: Use uniform sampling across valid steps (positions >= w). Event-aligned would require the full eval pipeline setup which adds complexity. Uniform sampling tests the bound in general, not just at violation positions.

## Sources

### Primary (HIGH confidence)
- Gao & Pavel (2017): "On the Properties of the Softmax Function with Application in Game Theory and Reinforcement Learning" -- Lipschitz constant 1/2 for softmax
- Weyl's inequality: |sigma_i(A+E) - sigma_i(A)| <= ||E||_2 -- spectral perturbation bound
- Stewart & Sun (1990): "Matrix Perturbation Theory" -- submultiplicativity and operator norm bounds
- Existing codebase: src/model/attention.py (softmax computation), src/evaluation/pipeline.py (AVWo computation)

### Secondary (MEDIUM confidence)
- Vaswani et al. (2017): "Attention Is All You Need" -- original 1/sqrt(d_k) scaling motivation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project
- Architecture: HIGH -- follows established standalone analysis module pattern
- Derivation: HIGH -- well-known mathematical results (softmax Lipschitz, Weyl's inequality, submultiplicativity)
- Pitfalls: MEDIUM -- causal mask interaction and scale factor placement require careful implementation

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable mathematical domain)
