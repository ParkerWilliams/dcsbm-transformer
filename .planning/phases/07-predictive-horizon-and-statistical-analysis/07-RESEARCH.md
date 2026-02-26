# Phase 7: Predictive Horizon and Statistical Analysis - Research

**Researched:** 2026-02-25
**Domain:** Event-aligned AUROC analysis, bootstrap inference, multiple testing correction, correlation/redundancy analysis
**Confidence:** HIGH

## Summary

Phase 7 consumes the fused evaluation output from Phase 6 (token_metrics.npz with per-step SVD metrics and behavioral labels) and answers the core research question: how far in advance can each SVD metric predict transformer rule violations? The analysis proceeds in three stages: (1) event extraction -- identifying violation and non-violation (successful completion) events from the behavioral classification arrays, with contamination filtering and per-r stratification; (2) AUROC computation -- for each SVD metric at each lookback distance j from 1 to r, computing AUROC between violation-event and control-event metric values at step (resolution - j); (3) statistical controls -- BCa bootstrap confidence intervals on AUROC, shuffle permutation tests, Holm-Bonferroni correction across primary metrics, Cohen's d effect sizes, correlation matrices, and metric importance rankings.

The existing codebase provides all necessary data in `EvaluationResult` and its NPZ serialization. However, a critical data gap exists: the current NPZ does not store per-encounter jumper metadata (encounter step, jumper vertex, r value) needed for event extraction and r-stratification. Phase 7 must either extend the NPZ output to include encounter metadata, or re-derive encounters from the generated token sequences plus jumper list. The recommended approach is to add encounter metadata arrays to the NPZ during evaluation (a small Phase 6 extension) and also support a re-derivation path from generated sequences for standalone analysis.

All statistical computations can be implemented with the existing stack (scipy 1.17, numpy 2.3) without adding new dependencies. AUROC is computed via the rank-based Mann-Whitney U statistic; BCa bootstrap uses `scipy.stats.bootstrap`; Holm-Bonferroni is a trivial manual implementation (no statsmodels needed); Cohen's d and Pearson correlation are straightforward numpy operations.

**Primary recommendation:** Create a new module `src/analysis/` with three files: `event_extraction.py` (extract and classify events from NPZ data, contamination filtering, r-stratification), `auroc_horizon.py` (AUROC at each lookback distance, predictive horizon computation, shuffle controls), and `statistical_controls.py` (BCa bootstrap CIs, Holm-Bonferroni correction, Cohen's d, correlation matrices, metric importance ranking). Each module is independently testable with synthetic data.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **Violation events**: First rule violation per walk only. Avoids double-counting from cascading failures after initial break
- **Non-violation comparison class**: Rule-followed steps at jumper+r (successful jumper completions). Same context type, opposite outcome -- not arbitrary non-violation steps
- **Multiple jumper encounters**: Each encounter is an independent event, with contamination filter: exclude any encounter whose countdown window [-r, 0] overlaps with a preceding violation's countdown window in the same walk. Specifically, if encounter B starts at step j_B and a prior violation occurred at step v_A, exclude B if j_B < v_A + r
- **Contamination audit**: Record exclusion count per configuration in result.json. Flag configurations losing >30% of encounters to contamination filtering
- **Alignment**: Rule resolution step (jumper_step + r) is step 0. All curves count backward from 0 to -r
- **Stratification**: Separate AUROC curves and event-aligned analysis per r value. Never mix different r values in the same curve. This is the primary analysis axis
- **Pre-registered primary metrics (Holm-Bonferroni corrected)**:
  1. QK^T Grassmannian distance
  2. QK^T spectral gap (sigma1 - sigma2)
  3. QK^T spectral entropy
  4. AVWo stable rank
  5. AVWo Grassmannian distance
- **Secondary metrics**: All 21 metrics (7 metrics x 3 targets) computed through full AUROC pipeline; 5 primary metrics Holm-Bonferroni corrected; 16 remaining reported as exploratory (uncorrected); clear labeling distinguishes primary from exploratory
- **Headline prediction (descriptive)**: QK^T predictive horizon > AVWo predictive horizon across all r values, with gap widening as r > w. Reported with bootstrap CIs, no formal test
- **Bootstrap**: 10,000 iterations, 95% BCa confidence intervals
- **Shuffle controls**: 10,000 permutations. Flag any metric where shuffled AUROC > 0.6 (configurable)
- **Predictive horizon threshold**: AUROC > 0.75 (configurable). Furthest j exceeding threshold
- **Two correlation matrices**: (1) raw metric values pooled -- measurement redundancy, (2) AUROC values across lookback -- predictive redundancy
- **Redundancy threshold**: |r| > 0.9 flags pair as redundant
- **Per-layer ranking**: Multi-layer models produce per-layer metric importance rankings (not aggregated across layers)

### Claude's Discretion
- WvWo handling: per-checkpoint reference metric, not per-step predictor -- compute but don't expect per-step predictive signal
- Exact AUROC implementation details (sklearn vs manual)
- Computational optimization (batching, parallelization of bootstrap/shuffle)
- result.json schema extensions for new metrics blocks
- Edge cases: what to do when event count is too small for reliable AUROC

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PRED-01 | AUROC at each lookback distance j (1 to r) for each SVD metric, comparing violation vs non-violation events | Rank-based AUROC via Mann-Whitney U statistic (scipy.stats.rankdata). For each metric at lookback j: extract metric values at step (resolution_step - j) for violation events and control events, compute AUROC = (rank_sum_violations - n_v*(n_v+1)/2) / (n_v * n_c). Per-r stratification means separate AUROC curves per r value. |
| PRED-02 | Predictive horizon as furthest j where AUROC > 0.75 per metric | Simple argmax scan backward from j=r: find largest j where AUROC exceeds configurable threshold (default 0.75). Store as integer per metric per r-value per layer. -1 or 0 when no j exceeds threshold. |
| PRED-03 | Position-matched baselines (control events at same absolute position in non-jumper walks) | Control events = successful jumper completions (rule_outcome == FOLLOWED at same resolution step type). Matched by absolute step position within walks to control for positional confounds. Alternative: control events from non-jumper walks at matched positions (if needed for larger control pool). |
| PRED-04 | Shuffle controls with permuted labels; flag if shuffled AUROC > 0.6 | scipy.stats.permutation_test or manual label shuffle: randomly permute violation/control labels 10,000 times, recompute AUROC each time. If any shuffled AUROC > 0.6 threshold, flag metric as potentially positional artifact. Store shuffle distribution statistics in result.json. |
| PRED-05 | Per-metric AUROC curves stored in result.json metrics block | JSON structure: `metrics.predictive_horizon.{r_value}.{target}.layer_{L}.{metric_name}` containing `auroc_by_lookback` (array of floats), `horizon` (int), `shuffle_flag` (bool), `bootstrap_ci` (pair of floats). |
| STAT-01 | Holm-Bonferroni correction across pre-registered primary metrics | Manual implementation (no statsmodels needed): sort 5 primary metric p-values, apply step-down correction p_adj[i] = p[i] * (5 - rank_i + 1), enforce monotonicity. Applied to the AUROC significance test (whether AUROC significantly > 0.5) for each primary metric at its max-AUROC lookback distance. |
| STAT-02 | Bootstrap confidence intervals on AUROC and predictive horizon | scipy.stats.bootstrap with method='BCa', n_resamples=10000, confidence_level=0.95. Statistic function computes AUROC from resampled violation/control groups. BCa handles skew in AUROC distribution near 0 and 1. |
| STAT-03 | Effect sizes (Cohen's d) for pre-failure vs post-failure metric distributions | Cohen's d = (mean_violation - mean_control) / pooled_std. Computed per metric at each lookback distance. Convention: positive d means violations have higher metric values. Straightforward numpy computation. |
| STAT-04 | SVD metric correlation matrix to identify redundant metrics | Two matrices: (1) Pearson correlation of raw metric values pooled across events (numpy corrcoef), (2) Pearson correlation of AUROC-by-lookback vectors across metrics. Flag pairs with |r| > 0.9 as redundant. Per-layer computation. |
| STAT-05 | Metric importance ranking by max AUROC across j values | Per layer: rank all metrics by their maximum AUROC across lookback distances. Annotate which top-ranked metrics are flagged as redundant (|correlation| > 0.9). Separate rankings for primary and secondary metrics. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 2.3.5 | Array operations, correlation matrices, Cohen's d | Already installed, sufficient for all numerical operations |
| scipy.stats | 1.17.1 | bootstrap (BCa), permutation_test, rankdata, mannwhitneyu | Already installed, provides BCa bootstrap CIs and permutation tests natively |
| torch | 2.10.0+cpu | Not directly used in analysis; data already in numpy arrays from NPZ | Available but analysis operates on numpy arrays post-extraction |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| scipy.stats.bootstrap | 1.17.1 | BCa confidence intervals on AUROC | Every AUROC computation gets bootstrap CIs |
| scipy.stats.rankdata | 1.17.1 | AUROC via rank method (vectorizable) | Core AUROC computation |
| scipy.stats.mannwhitneyu | 1.17.1 | AUROC p-value (alternative to permutation test) | Quick significance check before expensive permutation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual AUROC via rankdata | sklearn.metrics.roc_auc_score | sklearn not installed; adding 100MB+ dependency for one function is wasteful. Rank-based AUROC is mathematically equivalent and 10 lines of code. |
| Manual Holm-Bonferroni | statsmodels.stats.multitest.multipletests | statsmodels not installed; Holm-Bonferroni is 15 lines of code. No benefit to adding heavyweight dependency. |
| scipy.stats.bootstrap BCa | Manual bootstrap loop | scipy's BCa implementation handles bias-correction and acceleration automatically, avoids subtle bugs in jackknife acceleration factor. Use the standard library. |

**Installation:** No new packages needed. All operations use numpy 2.3 and scipy 1.17 already in the environment.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── analysis/                      # NEW: Phase 7 analysis module
│   ├── __init__.py
│   ├── event_extraction.py        # Event extraction, contamination filter, r-stratification
│   ├── auroc_horizon.py           # AUROC computation, predictive horizon, shuffle controls
│   └── statistical_controls.py    # Bootstrap CIs, Holm-Bonferroni, Cohen's d, correlation, ranking
├── evaluation/                    # Phase 6 (existing)
│   ├── pipeline.py                # Fused evaluation (may need small extension for encounter metadata)
│   ├── behavioral.py              # Behavioral classification
│   └── svd_metrics.py             # SVD metric functions
└── ...
```

### Pattern 1: Event Extraction from Evaluation Output
**What:** Extract violation and control events from EvaluationResult or loaded NPZ, producing structured event records with encounter step, resolution step, r value, and outcome.
**When to use:** Before any AUROC computation.
**Key data flow:**

```python
@dataclass(frozen=True, slots=True)
class AnalysisEvent:
    """A single jumper encounter event for predictive horizon analysis."""
    walk_idx: int           # Index into the sequences array
    encounter_step: int     # Step where jumper vertex was generated
    resolution_step: int    # encounter_step + r (where rule resolves)
    r_value: int            # Jump length for this encounter
    outcome: int            # RuleOutcome.FOLLOWED or RuleOutcome.VIOLATED
    is_first_violation: bool  # True if this is the first violation in its walk


def extract_events(
    generated: np.ndarray,       # [n_sequences, max_steps]
    rule_outcome: np.ndarray,    # [n_sequences, max_steps-1]
    failure_index: np.ndarray,   # [n_sequences]
    jumper_map: dict[int, JumperInfo],
) -> list[AnalysisEvent]:
    """Extract all jumper encounter events from evaluation output.

    For each sequence, scan tokens to find jumper encounters.
    Cross-reference with rule_outcome to determine FOLLOWED/VIOLATED.
    Apply contamination filter: exclude encounters whose countdown window
    overlaps with a preceding violation's window.
    """
    ...


def stratify_by_r(events: list[AnalysisEvent]) -> dict[int, list[AnalysisEvent]]:
    """Group events by r value. Each group gets independent AUROC analysis."""
    ...
```

### Pattern 2: AUROC at Each Lookback Distance
**What:** For a given metric and set of events at a single r value, compute AUROC at each lookback j from 1 to r.
**When to use:** Core analysis loop, called once per metric per r-value per layer.

```python
def compute_auroc_curve(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,     # [n_sequences, max_steps-1]
    r_value: int,
) -> np.ndarray:
    """Compute AUROC at each lookback distance j=1..r.

    For lookback j:
      - violation_values = metric_array[event.walk_idx, event.resolution_step - j]
        for each violation event
      - control_values = metric_array[event.walk_idx, event.resolution_step - j]
        for each control event
      - AUROC = rank-based computation

    Returns array of shape [r_value] with AUROC at each j.
    """
    aurocs = np.full(r_value, np.nan)
    for j in range(1, r_value + 1):
        viol_vals = []
        ctrl_vals = []
        for ev in violation_events:
            idx = ev.resolution_step - j
            if 0 <= idx < metric_array.shape[1]:
                val = metric_array[ev.walk_idx, idx]
                if np.isfinite(val):
                    viol_vals.append(val)
        for ev in control_events:
            idx = ev.resolution_step - j
            if 0 <= idx < metric_array.shape[1]:
                val = metric_array[ev.walk_idx, idx]
                if np.isfinite(val):
                    ctrl_vals.append(val)

        if len(viol_vals) >= 2 and len(ctrl_vals) >= 2:
            aurocs[j - 1] = _auroc_from_groups(
                np.array(viol_vals), np.array(ctrl_vals)
            )
    return aurocs


def _auroc_from_groups(violations: np.ndarray, controls: np.ndarray) -> float:
    """AUROC via rank method: P(violation > control)."""
    n_v, n_c = len(violations), len(controls)
    combined = np.concatenate([violations, controls])
    ranks = scipy.stats.rankdata(combined)
    rank_sum = ranks[:n_v].sum()
    return (rank_sum - n_v * (n_v + 1) / 2) / (n_v * n_c)
```

### Pattern 3: Contamination Filter
**What:** Exclude encounters whose countdown window overlaps with a preceding violation's window in the same walk.
**When to use:** During event extraction, before AUROC computation.

```python
def filter_contaminated_events(
    events_per_walk: dict[int, list[AnalysisEvent]],
) -> tuple[list[AnalysisEvent], int]:
    """Apply contamination filter within each walk.

    For each walk, process encounters in step order.
    Track violation windows. Exclude encounter B if
    B.encounter_step < prior_violation.resolution_step.

    Returns (filtered_events, exclusion_count).
    """
    filtered = []
    excluded = 0
    for walk_idx, walk_events in events_per_walk.items():
        sorted_events = sorted(walk_events, key=lambda e: e.encounter_step)
        last_violation_end = -1  # resolution step of most recent violation
        for event in sorted_events:
            if event.encounter_step < last_violation_end:
                excluded += 1
                continue
            filtered.append(event)
            if event.outcome == RuleOutcome.VIOLATED and event.is_first_violation:
                last_violation_end = event.resolution_step
    return filtered, excluded
```

### Pattern 4: BCa Bootstrap on AUROC
**What:** Compute BCa confidence intervals on AUROC using scipy.stats.bootstrap.
**When to use:** For every AUROC value that will be reported.

```python
from scipy.stats import bootstrap

def auroc_with_bootstrap_ci(
    violation_vals: np.ndarray,
    control_vals: np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng: int | np.random.Generator = 42,
) -> tuple[float, float, float]:
    """Compute AUROC with BCa bootstrap confidence interval.

    Returns (auroc, ci_low, ci_high).
    """
    def auroc_statistic(viol, ctrl, axis):
        # Vectorized for scipy.stats.bootstrap
        n_v = viol.shape[axis]
        n_c = ctrl.shape[axis]
        combined = np.concatenate([viol, ctrl], axis=axis)
        ranks = np.apply_along_axis(scipy.stats.rankdata, axis, combined)
        # Extract violation ranks
        if axis == 0:
            viol_ranks = ranks[:n_v]
        else:
            viol_ranks = ranks[..., :n_v]
        rank_sum = viol_ranks.sum(axis=axis)
        return (rank_sum - n_v * (n_v + 1) / 2) / (n_v * n_c)

    point_estimate = _auroc_from_groups(violation_vals, control_vals)

    res = bootstrap(
        (violation_vals, control_vals),
        auroc_statistic,
        n_resamples=n_resamples,
        method='BCa',
        confidence_level=confidence_level,
        rng=rng,
        vectorized=True,
    )
    return point_estimate, res.confidence_interval.low, res.confidence_interval.high
```

### Anti-Patterns to Avoid
- **Mixing r values in AUROC computation:** Never pool events with different r values into the same AUROC curve. Always stratify by r first. Different r values have different countdown window lengths and different baseline difficulty.
- **Using arbitrary non-violation steps as controls:** Controls must be successful jumper completions (FOLLOWED events) at the same r value, not random steps where no jumper rule was active.
- **Computing AUROC with fewer than ~10 events per class:** AUROC is unreliable with very few samples. Minimum recommended: 10 violations and 10 controls per r-value stratum. Report NaN and skip bootstrap when below threshold.
- **Applying Holm-Bonferroni to all 21 metrics:** Only the 5 pre-registered primary metrics get Holm-Bonferroni correction. The 16 secondary metrics are reported as exploratory with uncorrected p-values.
- **Aggregating per-layer rankings:** Multi-layer models get per-layer rankings, not a single aggregated ranking.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| BCa bootstrap confidence intervals | Manual jackknife + acceleration factor | scipy.stats.bootstrap(method='BCa') | BCa requires computing jackknife influence values and acceleration factor; subtle numerical issues near AUROC boundaries of 0 and 1 |
| AUROC computation | Custom comparison counting | scipy.stats.rankdata + rank formula | Rank-based AUROC handles ties correctly via midrank; custom counting is error-prone with ties |
| Permutation test | Manual shuffle loop | scipy.stats.permutation_test | scipy handles the exact p-value computation and batch vectorization; matches the bootstrap API pattern |

**Key insight:** The statistical primitives (AUROC, bootstrap, permutation test) are individually simple but composed into a multi-metric, multi-r, multi-layer pipeline. The complexity is in the event extraction and stratification logic, not the statistical functions themselves. Use scipy's proven implementations for the statistical building blocks and invest implementation effort in the event extraction and contamination filtering.

## Common Pitfalls

### Pitfall 1: Off-by-One in Event Alignment
**What goes wrong:** Lookback j=1 should access the metric at resolution_step - 1 (one step before rule resolves), but indexing errors can shift the entire AUROC curve by one step.
**Why it happens:** The rule_outcome array is indexed differently than the metric arrays (rule_outcome[t] reflects the transition from step t to t+1, while svd_metrics[t] is the metric at step t). The resolution_step in behavioral.py is the step where the deadline is checked (t such that t+1 == deadline).
**How to avoid:** Define a clear convention: `resolution_step` = the step index in the metric array where the rule resolves. For lookback j, access `metric_array[walk_idx, resolution_step - j]`. Verify with a known synthetic case where the metric jumps at a known step.
**Warning signs:** AUROC curve is shifted by 1 relative to expected; predictive horizon is systematically +1 or -1 from manual inspection.

### Pitfall 2: NaN Propagation in AUROC
**What goes wrong:** Some metric values at lookback positions are NaN (positions < w due to SVD warmup, or beyond sequence length). NaN values in violation or control groups corrupt the AUROC computation.
**Why it happens:** SVD metrics are NaN for positions < w-1 (warmup). If a lookback j reaches into the warmup zone (resolution_step - j < w-1), the metric value is NaN.
**How to avoid:** Filter NaN values before AUROC computation. Track how many events have valid values at each lookback j and report this count alongside AUROC. Minimum sample sizes should be enforced.
**Warning signs:** AUROC is NaN for early lookback distances; `_auroc_from_groups` receives empty arrays.

### Pitfall 3: Contamination Filter Excluding Too Many Events
**What goes wrong:** If violations are common and r is large, the contamination filter excludes most subsequent encounters in a walk, leaving very few events for analysis.
**Why it happens:** A violation at step v contaminates all encounters with encounter_step < v + r. For large r, this excludes a wide window of subsequent encounters.
**How to avoid:** Record exclusion counts per configuration and flag when >30% are excluded. Report per-r-value exclusion rates. Consider whether the first-only violation restriction plus contamination filter is too aggressive for some configurations.
**Warning signs:** Few events survive filtering; AUROC estimates have very wide bootstrap CIs; contamination audit flags fire.

### Pitfall 4: scipy.stats.bootstrap BCa Failure
**What goes wrong:** BCa bootstrap returns NaN confidence intervals when the bootstrap distribution is degenerate or has too few unique values.
**Why it happens:** AUROC can be exactly 0.5 for all bootstrap resamples (no signal), or exactly 1.0 (perfect separation). BCa's acceleration factor involves a jackknife that can fail with degenerate data.
**How to avoid:** Check for NaN in bootstrap results. Fall back to percentile method if BCa fails. Report the method used alongside the CI.
**Warning signs:** Bootstrap CI contains NaN; warnings from scipy about degenerate bootstrap distribution.

### Pitfall 5: Holm-Bonferroni Applied to Wrong Set of Tests
**What goes wrong:** Correction applied to all 21 metrics instead of just the 5 pre-registered primary metrics, reducing power unnecessarily.
**Why it happens:** Misunderstanding of which metrics are "primary" vs "secondary/exploratory".
**How to avoid:** Hardcode the 5 primary metric identifiers (from CONTEXT.md locked decisions). Label all output clearly as "primary (corrected)" vs "exploratory (uncorrected)".
**Warning signs:** Holm-Bonferroni adjustment factor is 21 instead of 5; all metrics lose significance.

### Pitfall 6: Correlation Matrix Computed Across r Values
**What goes wrong:** Pooling events from different r values into the same correlation matrix mixes apples and oranges.
**Why it happens:** Convenience -- more data points gives a nicer correlation matrix.
**How to avoid:** Compute correlation matrices per r value, or clearly document that pooling is intentional for the measurement-redundancy matrix (which may be acceptable since it characterizes metric relationships in general, not at a specific r).
**Warning signs:** Correlation structure changes dramatically when stratified by r vs pooled.

## Code Examples

### AUROC via Rank Method (Verified with scipy.stats.rankdata)
```python
import numpy as np
from scipy.stats import rankdata

def auroc_from_groups(violations: np.ndarray, controls: np.ndarray) -> float:
    """AUROC via rank-based method: equivalent to P(X_viol > X_ctrl).

    Mathematically equivalent to sklearn.metrics.roc_auc_score and
    to U / (n1 * n0) from Mann-Whitney U test.
    Handles ties via midrank (default in scipy.stats.rankdata).
    """
    n_v, n_c = len(violations), len(controls)
    if n_v == 0 or n_c == 0:
        return np.nan
    combined = np.concatenate([violations, controls])
    ranks = rankdata(combined)
    rank_sum = ranks[:n_v].sum()
    return (rank_sum - n_v * (n_v + 1) / 2) / (n_v * n_c)

# Verification:
# Perfect separation -> AUROC = 1.0
assert auroc_from_groups(np.array([5, 6, 7]), np.array([1, 2, 3])) == 1.0
# No separation -> AUROC = 0.5
# Random overlap -> AUROC between 0 and 1
```
**Confidence: HIGH** -- Verified against scipy.stats.mannwhitneyu in Python 3.12 with scipy 1.17.1 on this environment.

### Holm-Bonferroni Correction
```python
import numpy as np

def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    Args:
        p_values: Array of p-values to correct.
        alpha: Family-wise error rate.

    Returns:
        Tuple of (adjusted_p_values, reject_flags).
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    adjusted = np.zeros(m)
    for i in range(m):
        adjusted[i] = sorted_p[i] * (m - i)

    # Enforce monotonicity (step-down)
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result = np.zeros(m)
    result[sorted_idx] = adjusted
    return result, result <= alpha
```
**Confidence: HIGH** -- Standard algorithm, verified against known examples.

### Cohen's d Effect Size
```python
import numpy as np

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d: standardized mean difference with pooled std.

    Positive d means group1 has higher values than group2.
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return np.nan
    return (np.mean(group1) - np.mean(group2)) / pooled_std
```
**Confidence: HIGH** -- Standard formula.

### BCa Bootstrap CI on AUROC via scipy.stats.bootstrap
```python
from scipy.stats import bootstrap, rankdata
import numpy as np

def auroc_with_bca_ci(
    violation_vals: np.ndarray,
    control_vals: np.ndarray,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    rng: int | np.random.Generator = 42,
) -> tuple[float, float, float]:
    """AUROC with BCa bootstrap confidence interval.

    Returns (point_estimate, ci_low, ci_high).
    Falls back to percentile method if BCa produces NaN.
    """
    def auroc_stat(viol, ctrl, axis):
        n_v = viol.shape[axis]
        n_c = ctrl.shape[axis]
        combined = np.concatenate([viol, ctrl], axis=axis)
        ranks = np.apply_along_axis(rankdata, axis, combined)
        viol_ranks = ranks[:n_v] if axis == 0 else ranks[..., :n_v]
        rank_sum = viol_ranks.sum(axis=axis)
        return (rank_sum - n_v * (n_v + 1) / 2) / (n_v * n_c)

    point = auroc_from_groups(violation_vals, control_vals)

    try:
        res = bootstrap(
            (violation_vals, control_vals),
            auroc_stat,
            n_resamples=n_resamples,
            method='BCa',
            confidence_level=confidence_level,
            rng=rng,
            vectorized=True,
        )
        ci_low = res.confidence_interval.low
        ci_high = res.confidence_interval.high
    except Exception:
        # Fallback to percentile method
        res = bootstrap(
            (violation_vals, control_vals),
            auroc_stat,
            n_resamples=n_resamples,
            method='percentile',
            confidence_level=confidence_level,
            rng=rng,
            vectorized=True,
        )
        ci_low = res.confidence_interval.low
        ci_high = res.confidence_interval.high

    # Handle NaN from degenerate distributions
    if np.isnan(ci_low) or np.isnan(ci_high):
        res = bootstrap(
            (violation_vals, control_vals),
            auroc_stat,
            n_resamples=n_resamples,
            method='percentile',
            confidence_level=confidence_level,
            rng=rng,
            vectorized=True,
        )
        ci_low = res.confidence_interval.low
        ci_high = res.confidence_interval.high

    return point, float(ci_low), float(ci_high)
```
**Confidence: HIGH** -- Verified working with scipy 1.17.1 on this environment. BCa is the default method in scipy.stats.bootstrap and handles the jackknife acceleration internally.

## Critical Data Gap: Encounter Metadata

### The Problem
The current Phase 6 NPZ output contains:
- SVD metric arrays: `{target}.layer_{L}.{metric_name}` shape [n_sequences, max_steps-1]
- Behavioral arrays: `edge_valid`, `rule_outcome`, `failure_index`, `sequence_lengths`

Phase 7 needs to:
1. Know which step each jumper was encountered (to define countdown windows)
2. Know the r value for each encounter (to stratify by r)
3. Know which encounters resolved as FOLLOWED vs VIOLATED
4. Apply contamination filtering across encounters within a walk

The `rule_outcome` array marks resolution steps (where FOLLOWED/VIOLATED appear), but does NOT store encounter steps or r values. You can find resolution steps from `rule_outcome != NOT_APPLICABLE`, but to get the encounter step you need `resolution_step - r`, and to get r you need to know which jumper was involved.

### Recommended Solution

**Approach A (preferred): Re-derive encounters from generated sequences.** The `EvaluationResult` contains `generated` (the full token sequence). By scanning for jumper vertices in `generated` and cross-referencing with the jumper list, Phase 7 can reconstruct all encounters:
- For each sequence, for each step, check if `generated[walk, step]` is a jumper vertex
- If so, the encounter_step = step, r = jumper_map[vertex].r, resolution_step = step + r
- Cross-reference with `rule_outcome[walk, resolution_step]` for outcome

This requires access to: (1) the `generated` array (currently in `EvaluationResult` but NOT in the NPZ), and (2) the jumper list.

**Action required:** Either:
1. **Save `generated` to NPZ** (add `npz_data["generated"] = result.generated` in `save_evaluation_results`). This is a one-line change to pipeline.py. ~4MB for 1000 sequences x 1000 steps x int64.
2. **Save encounter metadata arrays to NPZ** as flat parallel arrays (encounter_walk_idx, encounter_step, encounter_r, encounter_outcome). More structured but requires more pipeline changes.

**Recommendation:** Option 1 (save generated to NPZ) is simpler and more general. Phase 7 re-derives encounters, which also serves as a cross-check on the behavioral classification.

**Alternatively**, Phase 7 can operate on the in-memory `EvaluationResult` directly (without going through NPZ), which already has `generated`. This is the simplest path for the initial implementation, with NPZ extension as a follow-up for standalone analysis of saved results.

## result.json Schema Extensions

Phase 7 adds a `predictive_horizon` block to `metrics`:

```json
{
  "metrics": {
    "scalars": { "...existing..." },
    "predictive_horizon": {
      "config": {
        "horizon_threshold": 0.75,
        "shuffle_flag_threshold": 0.6,
        "n_bootstrap": 10000,
        "n_shuffle": 10000,
        "bootstrap_confidence_level": 0.95,
        "primary_metrics": [
          "qkt.grassmannian_distance",
          "qkt.spectral_gap_1_2",
          "qkt.spectral_entropy",
          "avwo.stable_rank",
          "avwo.grassmannian_distance"
        ]
      },
      "contamination_audit": {
        "total_encounters": 150,
        "excluded_encounters": 12,
        "exclusion_rate": 0.08,
        "flagged": false,
        "per_r": {
          "32": { "total": 20, "excluded": 2, "exclusion_rate": 0.10 },
          "45": { "total": 25, "excluded": 3, "exclusion_rate": 0.12 }
        }
      },
      "by_r_value": {
        "32": {
          "n_violations": 15,
          "n_controls": 40,
          "by_layer": {
            "layer_0": {
              "primary": {
                "qkt.grassmannian_distance": {
                  "auroc_by_lookback": [0.55, 0.60, 0.65, "...array of r floats..."],
                  "horizon": 12,
                  "max_auroc": 0.82,
                  "max_auroc_lookback": 5,
                  "bootstrap_ci": [0.73, 0.89],
                  "shuffle_auroc_mean": 0.50,
                  "shuffle_auroc_p95": 0.56,
                  "shuffle_flag": false,
                  "p_value": 0.001,
                  "p_value_adjusted": 0.005,
                  "cohens_d_by_lookback": [0.2, 0.3, 0.4, "..."],
                  "n_valid_by_lookback": [15, 15, 14, "..."]
                }
              },
              "exploratory": {
                "qkt.stable_rank": { "...same structure, no p_value_adjusted..." }
              }
            }
          }
        }
      },
      "correlation_matrices": {
        "measurement_redundancy": {
          "metric_names": ["qkt.layer_0.grassmannian_distance", "..."],
          "matrix": [[1.0, 0.3, "..."], "..."],
          "redundant_pairs": [["metric_a", "metric_b", 0.95]]
        },
        "predictive_redundancy": {
          "metric_names": ["..."],
          "matrix": [["..."]],
          "redundant_pairs": []
        }
      },
      "metric_ranking": {
        "layer_0": {
          "primary": [
            { "metric": "qkt.grassmannian_distance", "max_auroc": 0.82, "best_r": 32, "redundant_with": [] },
            { "metric": "avwo.stable_rank", "max_auroc": 0.78, "best_r": 45, "redundant_with": ["avwo.condition_number"] }
          ],
          "all": [
            { "metric": "qkt.grassmannian_distance", "max_auroc": 0.82, "best_r": 32, "is_primary": true, "redundant_with": [] }
          ]
        }
      },
      "headline_comparison": {
        "description": "QK^T vs AVWo predictive horizon comparison (descriptive)",
        "by_r_value": {
          "32": {
            "qkt_max_horizon": 12,
            "avwo_max_horizon": 5,
            "qkt_leads": true,
            "gap": 7,
            "qkt_horizon_ci": [9, 15],
            "avwo_horizon_ci": [3, 8]
          }
        }
      }
    }
  }
}
```

## Edge Case Handling

### Too Few Events for Reliable AUROC
**Threshold:** Minimum 5 events per class (violation and control) to compute AUROC. Minimum 10 per class for bootstrap CIs.
**Action when below threshold:**
- 0-1 events per class: Report NaN for all statistics. Skip entirely.
- 2-4 events per class: Compute point estimate AUROC only (no bootstrap, no effect size). Flag as "low_n".
- 5-9 events per class: Compute AUROC and effect size. Bootstrap with percentile method (BCa unreliable with <10). Flag as "moderate_n".
- 10+ events per class: Full analysis (BCa bootstrap, effect size, shuffle controls).

Store the `n_valid_by_lookback` array in results so downstream consumers know how reliable each AUROC estimate is.

### WvWo Handling
WvWo is a static weight matrix (does not change during evaluation). Its SVD metrics are constant across all steps within a single checkpoint. Therefore:
- WvWo metrics have no per-step predictive signal (AUROC will be ~0.5 at all lookback distances)
- Compute WvWo AUROC curves for completeness but expect flat lines near 0.5
- Do NOT include WvWo metrics in the primary metric set or Holm-Bonferroni correction
- Report WvWo as a per-checkpoint reference metric in the result.json metadata, not in the predictive horizon curves

### Degenerate Configurations
If a configuration has 0 violations (model perfectly compliant) or 0 controls (every encounter is a violation):
- Record the configuration as "no_violations" or "no_controls" in result.json
- Skip AUROC computation entirely
- This is not an error -- it means the model is too good or too bad for predictive horizon analysis at this r value

## Open Questions

1. **NPZ vs in-memory event extraction**
   - What we know: EvaluationResult has `generated` in memory, but NPZ does not save it
   - What's unclear: Whether Phase 7 always runs immediately after Phase 6 (in-memory) or also supports standalone re-analysis from saved NPZ files
   - Recommendation: Support both paths. Save `generated` to NPZ (one-line change). Primary path is in-memory from EvaluationResult; NPZ path is for re-analysis. The NPZ extension is a small addition to Phase 6's save function.

2. **Per-layer vs aggregated headline comparison**
   - What we know: Rankings are per-layer (locked decision). The headline QK^T vs AVWo comparison needs a layer for comparison.
   - What's unclear: Which layer(s) to use for the headline comparison in multi-layer models
   - Recommendation: Report headline comparison per layer. Also report the "best layer" (layer with highest max AUROC for QK^T primary metrics) as the summary comparison.

3. **Bootstrap seed determinism**
   - What we know: Bootstrap with rng=42 is deterministic within a single run
   - What's unclear: Whether bootstrap results need to be reproducible across runs for the same data
   - Recommendation: Use a deterministic rng derived from the experiment seed. Store the bootstrap rng seed in the config block.

## Sources

### Primary (HIGH confidence)
- scipy.stats.bootstrap documentation -- verified BCa method works with AUROC statistic on scipy 1.17.1, tested in this environment
- scipy.stats.rankdata documentation -- verified rank-based AUROC computation matches Mann-Whitney U
- scipy.stats.permutation_test documentation -- verified signature and API compatibility

### Secondary (MEDIUM confidence)
- Holm-Bonferroni procedure: standard textbook algorithm, manual implementation verified against known examples
- Cohen's d: standard effect size formula, verified with synthetic data in this environment
- BCa bootstrap theory: Efron & Tibshirani (1993), well-established statistical methodology

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All computations use already-installed scipy 1.17 and numpy 2.3, verified working in this environment
- Architecture: HIGH - Clear data flow from EvaluationResult through event extraction to AUROC analysis, all data structures known from Phase 6 code inspection
- Pitfalls: HIGH - Off-by-one alignment, NaN handling, contamination filter edge cases are well-understood from Phase 6 experience and CONTEXT.md locked decisions
- Data gap: HIGH - The generated array omission from NPZ is clearly identified and has a one-line fix

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (stable dependencies, no fast-moving APIs)
