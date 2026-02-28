# Phase 12: Null Model Baseline - Research

**Researched:** 2026-02-26
**Domain:** Statistical null hypothesis testing, random matrix theory (Marchenko-Pastur), non-parametric comparison of Grassmannian drift distributions
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Same DCSBM graph, same trained model, walks that stay within their planted community (no block jumps)
- Generate 5x as many null walks as violation walks for tighter null distribution estimate
- Position-matched: null walks have the same total length and measurement positions as violation walks, just no jumper event at those positions
- Standalone generator function (not integrated into evaluation pipeline) -- takes a trained model + graph and produces null walks independently; can run null analysis on any existing experiment
- Holm-Bonferroni correction at alpha=0.05 (locked from Phase 7 and Phase 11 pre-registration -- already implemented in `src/analysis/statistical_controls.py`)
- Null model Mann-Whitney U tests form a **separate Holm-Bonferroni family** -- correct across lookback distances within the null comparison, independent of the primary metrics family
- Cohen's d threshold matches pre-registration Gate 2: d >= 0.5 is the bar for meaningful separation between null and violation drift
- Shaded 95% CI band of null distribution **plus** solid null median line on event-aligned plots
- Color scheme: light gray band with gray median line for null; existing color scheme for violation signal
- Full statistical summary in `null_model` block of result.json
- Per-lookback distance: null mean/std, violation mean/std, Mann-Whitney U statistic, raw p-value, Holm-Bonferroni adjusted p-value, Cohen's d, reject flag
- Aggregate: number of null walks, number of violation walks, global summary
- Standalone "Null Model Baseline" section in the report with its own plots and statistical summary table
- Marchenko-Pastur analysis appears as a subsection within the null model report section
- Overlay MP theoretical density curve on empirical histogram of QK^T singular values
- Compute KS (Kolmogorov-Smirnov) statistic and p-value as quantitative divergence metric between empirical SVs and MP distribution
- Compute at anchor points only: event position and a few reference positions (pre-event, post-event) -- not every evaluation step

### Claude's Discretion
- Exact null walk generation algorithm (how to ensure walks stay within community)
- MP density parameterization details (aspect ratio from d_k, matrix dimensions from anchor config)
- Internal structure of the standalone generator function
- Exact layout of the null model report section

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| NULL-01 | System generates evaluation walks with zero block jumpers (same graph, same trained model) to produce null Grassmannian drift distribution | Null walk generator: constrain start vertices and neighbor selection to exclude jumper vertices; use `generate_batch_unguided_walks` filtered to non-jumper vertices, or a custom generator that avoids jumper vertices entirely. Existing `GraphData.block_assignments` and `JumperInfo` provide all needed structure. |
| NULL-02 | System computes position-matched statistical comparison (Mann-Whitney U, Cohen's d) of Grassmannian drift between null and violation sequences at each lookback distance | `scipy.stats.mannwhitneyu` (v1.17.1 installed) provides U statistic + p-value. `cohens_d` already in `src/analysis/statistical_controls.py`. `holm_bonferroni` already implemented for correction. Position-matching uses same walk length and measurement positions. |
| NULL-03 | System computes Marchenko-Pastur reference distribution for QK^T singular values at the anchor config matrix dimensions | Hand-roll MP PDF/CDF (5-line formula). Use `scipy.stats.kstest` with callable CDF for KS test. QK^T is w x w, aspect ratio gamma = w/d_k where d_k = d_model (single head). Anchor: gamma = 64/128 = 0.5. No external library needed. |
| NULL-04 | System stores null model results in result.json `null_model` block and renders null overlay on event-aligned plots | Extend `result.json` schema with `null_model` block. Modify `plot_event_aligned` to accept optional null distribution data and render gray CI band + median. New report template section for null model. |
</phase_requirements>

## Summary

Phase 12 implements a null model baseline to demonstrate that the SVD Grassmannian drift signal at block jumper events is a genuine response rather than an artifact of normal attention dynamics. The phase has four distinct components: (1) generating jumper-free null walks, (2) position-matched Mann-Whitney U statistical comparison with Holm-Bonferroni correction, (3) Marchenko-Pastur random matrix reference distribution, and (4) result storage and visualization overlays.

The implementation is substantially simplified by the existing codebase. The Holm-Bonferroni correction and Cohen's d functions already exist in `src/analysis/statistical_controls.py`. The walk generation infrastructure in `src/walk/generator.py` provides the batch unguided walk generator that can be adapted. The evaluation pipeline in `src/evaluation/pipeline.py` provides the fused evaluate function that processes walks through the model. scipy 1.17.1 (already installed) provides `mannwhitneyu` and `kstest`. The Marchenko-Pastur distribution is simple enough to implement directly (5-line PDF formula) without adding a dependency.

The key architectural decision is that the null analysis is a **standalone** function, not integrated into the evaluation pipeline. It takes an existing experiment's trained model + graph + result data and produces null analysis results independently. This means it can be applied retroactively to any experiment.

**Primary recommendation:** Implement as a new `src/analysis/null_model.py` module containing the null walk generator, the statistical comparison pipeline, and the MP reference computation. Add a `src/visualization/null_overlay.py` for the event-aligned null overlay rendering. Extend result schema, report template, and render orchestrator.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scipy.stats.mannwhitneyu | 1.17.1 (installed) | Mann-Whitney U rank test for null vs violation comparison | Standard non-parametric two-sample test; returns U statistic and p-value |
| scipy.stats.kstest | 1.17.1 (installed) | Kolmogorov-Smirnov goodness-of-fit test for MP comparison | Accepts callable CDF; returns KS statistic and p-value |
| scipy.integrate.quad | 1.17.1 (installed) | Numerical integration of MP PDF to compute CDF | Standard for definite integrals of scalar functions |
| numpy | 2.x (installed) | Array operations, random walk generation | Already used throughout project |
| matplotlib | 3.x (installed) | Histogram + MP overlay, null CI band on event-aligned plots | Already used throughout project |

### Supporting (already in project)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| src/analysis/statistical_controls.holm_bonferroni | existing | Holm-Bonferroni correction for null model family | After computing per-lookback Mann-Whitney p-values |
| src/analysis/statistical_controls.cohens_d | existing | Cohen's d effect size | At each lookback distance for null vs violation |
| src/walk/generator.generate_batch_unguided_walks | existing | Vectorized random walk generation | Base for null walk generation (with vertex filtering) |
| src/evaluation/pipeline.fused_evaluate | existing | Autoregressive generation with SVD extraction | Running the model on null walks to collect Grassmannian drift |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled MP PDF | scikit-rmt MarchenkoPasturDistribution | Adds dependency for 5 lines of math; project has no scikit-rmt installed |
| scipy.stats.mannwhitneyu | Custom rank-based AUROC (already in auroc_horizon.py) | mannwhitneyu is the standard API; AUROC already exists but U test is more appropriate for explicit hypothesis testing |
| scipy.integrate.quad for MP CDF | Analytical MP CDF (exists in closed form for gamma != 1) | Closed-form involves inverse trig functions and is error-prone; quad is robust and negligible cost for single-point evaluation |

## Architecture Patterns

### Recommended Project Structure
```
src/
├── analysis/
│   ├── null_model.py         # NEW: null walk gen, MW-U comparison, MP reference
│   └── statistical_controls.py  # EXISTING: holm_bonferroni, cohens_d (reuse)
├── visualization/
│   ├── null_overlay.py       # NEW: null CI band overlay on event-aligned plots
│   ├── mp_histogram.py       # NEW: MP density overlay on SV histogram
│   └── event_aligned.py      # MODIFY: accept optional null_data parameter
├── reporting/
│   ├── single.py             # MODIFY: add null model section
│   └── templates/
│       └── single_report.html # MODIFY: add null model template block
└── results/
    └── schema.py             # MODIFY: validate null_model block
```

### Pattern 1: Standalone Null Analysis Pipeline
**What:** A single top-level function `run_null_analysis(model, graph_data, jumpers, config, eval_result, device)` that generates null walks, runs them through the model, computes Grassmannian drift, performs MW-U tests, computes Cohen's d, applies Holm-Bonferroni, and returns a dict ready for `result.json["null_model"]`.
**When to use:** After evaluation pipeline has produced the violation results. Can be called on any existing experiment.
**Example:**
```python
# Source: project pattern from apply_statistical_controls()
def run_null_analysis(
    model: nn.Module,
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    violation_drift: dict[str, np.ndarray],  # metric_key -> drift values at each lookback
    n_violation_walks: int,
    device: torch.device,
    null_seed: int = 9999,
) -> dict:
    """Generate null walks, compute drift, compare against violations.

    Returns dict with per-lookback MW-U results and aggregate summary.
    """
    # 1. Generate null walks (5x violation count)
    n_null = 5 * n_violation_walks
    null_walks = generate_null_walks(graph_data, jumpers, config, n_null, seed=null_seed)

    # 2. Run through model to collect SVD metrics (same as fused_evaluate)
    null_result = fused_evaluate(model, null_walks, graph_data, jumpers, config, device)

    # 3. Extract Grassmannian drift at position-matched lookback distances
    null_drift = extract_drift_at_lookbacks(null_result, ...)

    # 4. Per-lookback Mann-Whitney U + Cohen's d
    comparison = compare_null_vs_violation(null_drift, violation_drift)

    # 5. Holm-Bonferroni across lookback distances
    corrected = apply_holm_bonferroni_to_null(comparison)

    return corrected
```

### Pattern 2: Null Walk Generation (Community-Constrained)
**What:** Generate walks that never visit a jumper vertex. Two approaches:
1. **Vertex filtering:** Remove jumper vertices from the graph's adjacency structure before walking. Create a filtered CSR matrix that excludes jumper vertices as both sources and destinations.
2. **Walk filtering:** Generate many batch unguided walks and keep only those that never visit a jumper vertex. Simpler but potentially wasteful.

**Recommended:** Vertex filtering approach. The jumper set is small (typically 2 per block x 4 blocks = 8 vertices out of 500), so filtering is lightweight and guarantees zero jumper encounters without overgeneration.

**Example:**
```python
def generate_null_walks(
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    n_walks: int,
    seed: int,
) -> np.ndarray:
    """Generate walks that never encounter a jumper vertex."""
    jumper_set = {j.vertex_id for j in jumpers}

    # Option A: Filter adjacency (remove edges to/from jumper vertices)
    adj = graph_data.adjacency.copy()
    for v in jumper_set:
        adj[v, :] = 0  # remove outgoing edges from jumpers
        adj[:, v] = 0  # remove incoming edges to jumpers
    adj.eliminate_zeros()
    filtered_adj = adj.tocsr()

    # Generate start vertices from non-jumper set
    non_jumper_vertices = [v for v in range(graph_data.n) if v not in jumper_set]
    rng = np.random.default_rng(seed)
    starts = rng.choice(non_jumper_vertices, size=n_walks, replace=True)

    # Use existing batch generator with filtered adjacency
    return generate_batch_unguided_walks(
        starts, config.training.walk_length, rng,
        filtered_adj.indptr, filtered_adj.indices
    )
```

### Pattern 3: Position-Matched Drift Extraction
**What:** Extract Grassmannian drift values at the same absolute sequence positions for null and violation walks. For violation walks, the "event position" is the resolution_step. For null walks (no events), pick the same set of absolute positions.
**When to use:** When constructing the two distributions for MW-U comparison.
**Key insight:** Without position-matching, drift differences could be attributed to sequence position effects (drift naturally varies across the sequence). Position-matching isolates the effect of jumper events.

```python
def extract_position_matched_drift(
    metric_array: np.ndarray,  # [n_sequences, max_steps-1]
    positions: list[int],       # absolute positions to sample
) -> dict[int, np.ndarray]:
    """Extract metric values at specific positions across all sequences.

    Returns dict mapping position -> array of values (one per sequence, NaN-filtered).
    """
    result = {}
    for pos in positions:
        if 0 <= pos < metric_array.shape[1]:
            vals = metric_array[:, pos]
            result[pos] = vals[np.isfinite(vals)]
    return result
```

### Pattern 4: Marchenko-Pastur Reference
**What:** Compute the MP density and CDF for a random w x w matrix and compare against empirical QK^T singular values using KS test.
**Parameterization:**
- QK^T is w x w (after causal masking + zero fill)
- The QK^T matrix arises from Q @ K^T where Q, K are [T, d_k] with T = w, d_k = d_model
- Aspect ratio gamma = T/d_k = w/d_model (for anchor config: 64/128 = 0.5)
- sigma^2 calibrated from the data (mean of squared singular values, or set to 1 and compare shape only)

```python
def marchenko_pastur_pdf(x: float, gamma: float, sigma2: float = 1.0) -> float:
    """Marchenko-Pastur probability density function.

    f(x) = sqrt((lam+ - x)(x - lam-)) / (2 * pi * sigma^2 * gamma * x)
    for x in [lam-, lam+], where lam+/- = sigma^2 * (1 +/- sqrt(gamma))^2
    """
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    if x <= lam_minus or x >= lam_plus:
        return 0.0
    return np.sqrt((lam_plus - x) * (x - lam_minus)) / (2 * np.pi * sigma2 * gamma * x)
```

### Anti-Patterns to Avoid
- **Integrating null generation into the evaluation pipeline:** The user explicitly decided on a standalone function. Do NOT modify `fused_evaluate` or `evaluation/pipeline.py` to add null walk mode.
- **Using the same Holm-Bonferroni family for null and primary metrics:** Null model MW-U tests form a SEPARATE family. Do not merge p-values from null comparison with the existing primary AUROC p-values.
- **Computing MP at every evaluation step:** Only compute at anchor points (event position, pre-event, post-event). The KS test is a diagnostic, not a per-step metric.
- **Removing jumper vertices from the graph entirely:** Only skip them in walks. The graph topology stays intact. Removing vertices would change the graph structure and invalidate the trained model.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Mann-Whitney U test | Custom rank-based implementation | `scipy.stats.mannwhitneyu` | Handles ties, continuity correction, exact/asymptotic methods; already imported elsewhere in project |
| Kolmogorov-Smirnov test | Custom CDF comparison | `scipy.stats.kstest` with callable CDF | Handles critical values, p-value computation, numerical precision |
| Holm-Bonferroni correction | New implementation | `src/analysis/statistical_controls.holm_bonferroni` | Already implemented, tested, and used in Phase 7-8 |
| Cohen's d | New implementation | `src/analysis/statistical_controls.cohens_d` | Already implemented with proper pooled standard deviation |
| Batch walk generation | Custom loop-based generator | `src/walk/generator.generate_batch_unguided_walks` | Vectorized, handles CSR arrays, already tested |

**Key insight:** Phase 12 can reuse approximately 60% of its statistical machinery from existing code. The novel work is: (1) null walk generation with jumper filtering, (2) wiring MW-U into the per-lookback comparison structure, (3) MP PDF/CDF implementation, and (4) visualization overlays.

## Common Pitfalls

### Pitfall 1: Position Confound in Null Comparison
**What goes wrong:** Comparing null drift at arbitrary positions against violation drift at event positions introduces a positional confound -- Grassmannian drift naturally varies across sequence position.
**Why it happens:** Null walks have no events, so there's no natural "anchor position" to align to.
**How to avoid:** Use the SAME absolute positions as the violation events. For each violation event at resolution_step R, sample null walk drift at position R. This means the null distribution is position-matched per lookback distance.
**Warning signs:** If null drift shows systematic position-dependent trends that match violation drift trends, position-matching wasn't done correctly.

### Pitfall 2: Graph Disconnection After Jumper Removal
**What goes wrong:** Removing edges to/from jumper vertices could disconnect the graph, causing dead-end walks.
**Why it happens:** In small graphs or sparse configurations, jumper vertices might be cut vertices (bridges).
**How to avoid:** Only remove jumper vertices from the start vertex pool and filter walk steps to avoid landing on jumpers. Alternatively, remove only outgoing edges from jumpers (so walks can pass through jumper vertices but never start there and the jumper rule never triggers because the walk never "visits" the jumper as a current position that would trigger a constraint). Actually, the simplest correct approach: generate walks using the FULL graph but exclude jumper vertices from start positions AND check that no generated position is a jumper. If a walk visits a jumper, discard and regenerate. With 8 jumpers out of 500 vertices, the discard rate will be low.
**Warning signs:** Walks terminating early or generation hanging.

### Pitfall 3: MP Parameterization Mismatch
**What goes wrong:** Using the wrong aspect ratio or sigma for the MP distribution, making the KS test meaningless.
**Why it happens:** QK^T = Q @ K^T where Q, K are [T, d_k]. The "random matrix" underlying QK^T is not QK^T itself but the factors Q and K. The squared singular values of QK^T relate to the Wishart distribution, not directly to MP.
**How to avoid:** Two approaches: (a) Compare singular values of QK^T against MP for a w x w matrix with gamma = w/d_k and sigma^2 estimated from data. (b) Compare squared singular values of Q (or K) directly against MP with gamma = w/d_k. Approach (a) is what the user specified (compare QK^T singular values) but sigma^2 should be calibrated from the null distribution's empirical variance. The KS test measures deviation from random baseline, not exact fit.
**Warning signs:** KS p-value is always 0 (always rejects) -- suggests sigma^2 or gamma is wrong.

### Pitfall 4: Holm-Bonferroni Family Mixing
**What goes wrong:** Including null model p-values in the same family as primary AUROC p-values, over-correcting both.
**Why it happens:** Natural tendency to merge all p-values for "more rigorous" correction.
**How to avoid:** The CONTEXT.md explicitly states separate families. The null model family corrects across lookback distances ONLY. Keep primary metrics family and null model family independent.
**Warning signs:** Null model results showing different reject flags depending on which other tests are included.

### Pitfall 5: Insufficient Null Walks Generating Empty Distributions
**What goes wrong:** After filtering for valid finite values at position-matched lookback positions, some lookback distances end up with too few null samples for a meaningful MW-U test.
**Why it happens:** NaN values in SVD metrics (e.g., before context warmup at position w) combined with position-matching can thin out the sample.
**How to avoid:** Generate 5x violation count (user decision). Also add a minimum sample size check (e.g., n >= 5 per group) before computing MW-U. Report "insufficient samples" for lookback distances that fail the check.
**Warning signs:** MW-U p-values of exactly 1.0 or NaN at certain lookback distances.

## Code Examples

### Mann-Whitney U Test at Each Lookback Distance
```python
# Source: scipy.stats.mannwhitneyu API (scipy 1.17.1)
from scipy.stats import mannwhitneyu

def null_vs_violation_mw_u(
    null_drift: np.ndarray,     # [n_null_walks] values at lookback j
    violation_drift: np.ndarray, # [n_violation_walks] values at lookback j
) -> dict:
    """Position-matched Mann-Whitney U test."""
    if len(null_drift) < 5 or len(violation_drift) < 5:
        return {"U": np.nan, "p_value": np.nan, "insufficient_samples": True}

    result = mannwhitneyu(
        violation_drift, null_drift,
        alternative='two-sided',
        method='auto',
    )
    return {
        "U": float(result.statistic),
        "p_value": float(result.pvalue),
        "insufficient_samples": False,
    }
```

### Marchenko-Pastur PDF and KS Test
```python
# Source: MP distribution formula from Random Matrix Theory
# KS test: scipy.stats.kstest with callable CDF
from scipy.stats import kstest
from scipy.integrate import quad

def marchenko_pastur_pdf(x, gamma, sigma2=1.0):
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    if x <= lam_minus or x >= lam_plus:
        return 0.0
    return np.sqrt((lam_plus - x) * (x - lam_minus)) / (2 * np.pi * sigma2 * gamma * x)

def marchenko_pastur_cdf(x, gamma, sigma2=1.0):
    lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
    lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2
    if x <= lam_minus:
        return 0.0
    if x >= lam_plus:
        return 1.0
    result, _ = quad(marchenko_pastur_pdf, lam_minus, x, args=(gamma, sigma2))
    return result

def run_mp_ks_test(
    singular_values: np.ndarray,  # empirical SVs from QK^T at one position
    gamma: float,                 # w / d_k
) -> dict:
    """Compare empirical QK^T SVs against MP distribution."""
    # Calibrate sigma^2 from mean of squared SVs
    sv_squared = singular_values ** 2
    sigma2 = float(np.mean(sv_squared)) / (1.0 + gamma)  # MP mean = sigma^2 * (1 + gamma)

    ks_stat, p_value = kstest(
        sv_squared,
        lambda x: marchenko_pastur_cdf(x, gamma, sigma2),
    )
    return {
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(p_value),
        "gamma": gamma,
        "sigma2": float(sigma2),
        "lambda_minus": float(sigma2 * (1 - np.sqrt(gamma)) ** 2),
        "lambda_plus": float(sigma2 * (1 + np.sqrt(gamma)) ** 2),
    }
```

### Null Overlay on Event-Aligned Plot
```python
# Source: project pattern from src/visualization/event_aligned.py
# Colors: gray for null (per CONTEXT.md locked decision)
NULL_BAND_COLOR = (0.7, 0.7, 0.7, 0.3)  # light gray with alpha
NULL_MEDIAN_COLOR = (0.5, 0.5, 0.5, 1.0)  # solid gray

def plot_event_aligned_with_null(
    metric_values: np.ndarray,
    events: list[AnalysisEvent],
    null_distribution: dict,  # {position: {"median": float, "ci_low": float, "ci_high": float}}
    window: int = 10,
    metric_name: str = "SVD metric",
    ax=None,
):
    """Event-aligned plot with null distribution overlay."""
    # Render base violation + control traces (existing code)
    fig = plot_event_aligned(metric_values, events, window, metric_name, ax)

    # Overlay null distribution
    positions = np.arange(-window, window + 1)
    null_medians = [null_distribution.get(p, {}).get("median", np.nan) for p in positions]
    null_ci_low = [null_distribution.get(p, {}).get("ci_low", np.nan) for p in positions]
    null_ci_high = [null_distribution.get(p, {}).get("ci_high", np.nan) for p in positions]

    ax = fig.axes[0]
    ax.fill_between(positions, null_ci_low, null_ci_high, color=NULL_BAND_COLOR, label="Null 95% CI")
    ax.plot(positions, null_medians, color=NULL_MEDIAN_COLOR, linewidth=1.5, linestyle='-', label="Null median")
    ax.legend(fontsize=8)

    return fig
```

### result.json null_model Block Structure
```python
# Source: project pattern from result.json schema
null_model_schema = {
    "null_model": {
        "config": {
            "n_null_walks": 500,          # 5x violation count
            "n_violation_walks": 100,
            "null_seed": 9999,
            "alpha": 0.05,
            "cohens_d_threshold": 0.5,
        },
        "by_lookback": {
            "1": {
                "null_mean": 0.15,
                "null_std": 0.03,
                "violation_mean": 0.42,
                "violation_std": 0.08,
                "mann_whitney_U": 3245.0,
                "p_value_raw": 0.001,
                "p_value_adjusted": 0.005,  # Holm-Bonferroni within null family
                "cohens_d": 1.23,
                "reject": True,
                "n_null_valid": 480,
                "n_violation_valid": 95,
            },
            # ... more lookback distances
        },
        "aggregate": {
            "n_lookbacks_tested": 8,
            "n_lookbacks_rejected": 6,
            "max_cohens_d": 1.45,
            "max_cohens_d_lookback": 3,
            "signal_exceeds_noise": True,  # at least one reject with d >= 0.5
        },
        "marchenko_pastur": {
            "gamma": 0.5,
            "sigma2": 0.0123,
            "lambda_minus": 0.0037,
            "lambda_plus": 0.0365,
            "anchor_positions": {
                "event": {"ks_statistic": 0.234, "ks_p_value": 0.001},
                "pre_event_5": {"ks_statistic": 0.187, "ks_p_value": 0.012},
                "post_event_5": {"ks_statistic": 0.156, "ks_p_value": 0.034},
            },
        },
    }
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No null model | Position-matched null with MW-U + Cohen's d | Phase 12 (this phase) | Transforms signal from "interesting pattern" to "statistically validated departure from null" |
| Visual inspection of SVD metric divergence | Quantitative Holm-Bonferroni corrected comparison | Phase 7-8 + Phase 12 | Publishable statistical rigor |
| No random matrix reference | Marchenko-Pastur baseline for QK^T | Phase 12 (this phase) | Grounds the interpretation: if MP fits, the attention matrix is random; if it doesn't, the model has learned structure |

## Open Questions

1. **MP Distribution for Squared vs Raw Singular Values**
   - What we know: MP describes the limiting distribution of squared singular values (eigenvalues) of XX^T/n for random X. QK^T singular values are NOT eigenvalues of a Wishart matrix.
   - What's unclear: Whether to apply MP to raw SVs, squared SVs, or the eigenvalues of QK^T (QK^T)^T.
   - Recommendation: Apply to squared singular values of QK^T (these are eigenvalues of QK^T (QK^T)^T which has Wishart-like structure). Calibrate sigma^2 from data. The KS test is a diagnostic tool, not a proof -- exact MP fit is not expected. What matters is the DEGREE of deviation at event vs non-event positions. Use LOW confidence on exact MP parameterization but HIGH confidence on the comparative analysis (event vs non-event deviation from MP).

2. **Null Walk Generation: Filter vs Avoid**
   - What we know: Must produce jumper-free walks. Jumpers are ~8 vertices out of 500.
   - What's unclear: Whether to modify the adjacency matrix to remove jumper vertex edges or to generate walks on the full graph and discard those that visit jumpers.
   - Recommendation: **Filter-and-discard approach.** Generate batch unguided walks on the FULL graph (preserving transition probabilities), then discard any walk that visits a jumper vertex. With 8/500 = 1.6% jumper fraction, the probability a walk of length 256 visits no jumper is roughly (1-0.016)^256 ~ 0.016 -- meaning most walks WILL visit a jumper. This means the discard rate is very high. **Better approach:** Modify the adjacency matrix to zero out columns corresponding to jumper vertices (so walks can't REACH jumper vertices), then regenerate walks. This preserves the local transition structure for non-jumper vertices and guarantees 100% jumper-free walks. Dead-end risk: a non-jumper vertex whose only outgoing edges lead to jumpers. Handle by checking for zero-degree vertices after filtering and avoiding them as walk positions (fall back to random restart if stuck).

3. **Null Walk "Measurement Positions" for Position-Matching**
   - What we know: Violation events have a natural anchor: the resolution_step. Null walks have no events.
   - What's unclear: Exactly how to define the measurement positions for null walks.
   - Recommendation: For each violation event at resolution_step R with lookback distances j=1..r, sample the null walk metric at the SAME absolute positions R, R-1, R-2, ..., R-r. Pool across all violation events' positions. This means each null walk contributes values at multiple positions, one per violation event's anchor point.

## Sources

### Primary (HIGH confidence)
- `src/analysis/statistical_controls.py` -- Existing holm_bonferroni and cohens_d implementations
- `src/walk/generator.py` -- Existing generate_batch_unguided_walks function
- `src/evaluation/pipeline.py` -- Existing fused_evaluate for SVD metric collection
- `src/visualization/event_aligned.py` -- Existing plot_event_aligned for overlay extension
- `src/visualization/style.py` -- Existing color palette and save_figure utility
- `src/results/schema.py` -- Existing result validation and write functions
- scipy 1.17.1 API: `mannwhitneyu`, `kstest`, `quad` -- verified installed and working
- `src/config/experiment.py` -- ExperimentConfig with d_model=128, w=64 defaults

### Secondary (MEDIUM confidence)
- [Marchenko-Pastur distribution Wikipedia](https://en.wikipedia.org/wiki/Marchenko%E2%80%93Pastur_distribution) -- PDF formula and parameterization
- [scipy.stats.mannwhitneyu documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html) -- API reference for MW-U test
- [scipy.stats.kstest documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html) -- API reference for KS test with callable CDF

### Tertiary (LOW confidence)
- MP distribution parameterization for QK^T specifically (aspect ratio choice of w/d_k) -- this is the researcher's interpretation of "aspect ratio from d_k" in CONTEXT.md. The exact correct parameterization depends on whether we treat QK^T as a product of two random matrices (which has a different spectral distribution than a single random matrix). However, the KS test as a comparative diagnostic (event vs non-event) is valid regardless of exact parameterization.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and used in project; APIs verified working
- Architecture: HIGH -- standalone analysis pattern matches existing statistical_controls.py pattern; clear file structure
- Pitfalls: HIGH -- well-known issues with position-matching and null model construction in neuroimaging/signal processing literature
- MP parameterization: MEDIUM -- exact gamma for QK^T is debatable; calibrating sigma^2 from data mitigates this
- Null walk generation: HIGH -- simple adjacency filtering with fallback for dead-ends

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable domain, no library version sensitivity)
