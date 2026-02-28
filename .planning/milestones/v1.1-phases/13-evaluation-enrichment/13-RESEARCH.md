# Phase 13: Evaluation Enrichment - Research

**Researched:** 2026-02-26
**Domain:** Precision-recall analysis, calibration diagnostics, SVD benchmarking
**Confidence:** HIGH

## Summary

Phase 13 enriches the evaluation pipeline with three independent capabilities: (1) precision-recall curves and AUPRC computation that mirrors the existing AUROC pipeline, (2) reliability diagrams with Expected Calibration Error for violation prediction calibration, and (3) SVD computational overhead benchmarking comparing full, randomized, and values-only SVD methods.

All three capabilities build on the existing event extraction infrastructure (`src/analysis/event_extraction.py`) and follow established patterns in the codebase: analysis functions in `src/analysis/`, visualization functions in `src/visualization/`, render integration in `src/visualization/render.py`, report integration in `src/reporting/single.py` with HTML template additions, and result schema validation in `src/results/schema.py`.

**Primary recommendation:** Implement the three capabilities as independent modules sharing the existing event extraction pipeline, with each adding its own analysis function, visualization function, render hook, template section, and schema validation block.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- PR Curve Presentation: Mirror existing AUROC layout (same grid structure with one subplot per metric x lookback distance), AUPRC values as summary table and annotated on plots, no-skill baseline as dashed horizontal line at positive class prevalence rate, store AUPRC and PR curve data in result.json alongside existing AUROC fields
- Calibration Diagnostics: 10 equal-width bins for reliability diagrams, one reliability diagram per metric with lookback distances as separate colored lines, ECE annotated on each reliability diagram plus summary table, include histogram of predicted probabilities below each reliability diagram
- SVD Benchmark Reporting: Cost summary table plus grouped bar chart (targets on x-axis, SVD methods as groups), report both relative Frobenius error and singular value correlation as accuracy metrics, separate profiling mode (not during every evaluation), 5 warmup iterations + 20 timed iterations for CUDA event benchmarking
- Report Section Ordering: AUROC -> PR curves -> Calibration -> SVD cost, each new section is a collapsible block collapsed by default

### Claude's Discretion
- Exact color palettes and styling for new plots
- Collapsible section implementation details (HTML/CSS approach)
- PR curve interpolation method
- How to handle metrics with insufficient positive samples for meaningful PR curves

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PRCL-01 | PR curves and AUPRC per metric per lookback distance, same event extraction as AUROC, stored in result.json | Use sklearn.metrics.precision_recall_curve + average_precision_score; reuse extract_events/filter_contaminated_events/stratify_by_r from event_extraction.py |
| PRCL-02 | Reliability diagrams with ECE for violation prediction | Use sklearn.calibration.calibration_curve for binning; compute ECE as weighted mean absolute bin error |
| PRCL-03 | PR curves and reliability diagrams in HTML reports alongside AUROC | Add collapsible sections to single_report.html template; new figure categories in _collect_figures and render_all |
| OVHD-01 | Wall-clock SVD cost per step by target and matrix dimension using CUDA events with warmup | Use torch.cuda.Event for GPU timing; benchmark each target (QK^T, WvWo, AVWo) at representative matrix dimensions |
| OVHD-02 | Compare full SVD vs randomized vs values-only with accuracy-cost tradeoff | torch.linalg.svd (full), torch.svd_lowrank (randomized), torch.linalg.svdvals (values-only); measure relative Frobenius error and SV correlation |
| OVHD-03 | Cost summary table in HTML reports | Add SVD cost section to template with table and grouped bar chart |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sklearn.metrics | (bundled with scikit-learn) | precision_recall_curve, average_precision_score | Standard PR curve computation; handles interpolation and edge cases |
| sklearn.calibration | (bundled with scikit-learn) | calibration_curve | Standard binning for reliability diagrams |
| torch.cuda.Event | (bundled with PyTorch) | GPU timing with CUDA events | Correct GPU timing (not wall-clock) for SVD benchmarks |
| torch.svd_lowrank | (bundled with PyTorch) | Randomized SVD | Built-in randomized SVD implementation |
| torch.linalg.svdvals | (bundled with PyTorch) | Values-only SVD | No U/Vh overhead when only singular values needed |
| matplotlib | 3.x | All new plots (PR curves, reliability diagrams, bar charts) | Already used throughout project |
| numpy | (existing) | Array operations | Already used throughout |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| seaborn | (existing) | Consistent styling | Already used via style.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| sklearn PR curve | Manual computation | sklearn handles interpolation, edge cases, and is well-tested |
| Manual ECE | netcal or calibration-belt | Extra dependency not needed for simple equal-width binning |

**Installation:** No new packages needed -- sklearn and torch already in project dependencies.

## Architecture Patterns

### Pattern 1: Mirror AUROC Pipeline for PR Curves

The existing AUROC pipeline in `src/analysis/auroc_horizon.py` follows a clear pattern:
1. Reuse events from `event_extraction.py` (extract_events, filter_contaminated_events, stratify_by_r)
2. For each r_value, for each metric_key, compute the metric (AUROC curve -> PR curve)
3. Return nested dict matching result.json schema
4. Orchestrator (`run_auroc_analysis`) ties it together

**PR curve implementation should mirror this exactly:**
- New file: `src/analysis/pr_curves.py`
- Same function signature pattern as `compute_auroc_curve` -> `compute_pr_curve`
- Same `run_pr_analysis` orchestrator pattern as `run_auroc_analysis`
- Output stored in `metrics.pr_curves` block in result.json (alongside `metrics.predictive_horizon`)

```python
# Pattern: compute_pr_curve mirrors compute_auroc_curve
def compute_pr_curve(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,
    r_value: int,
    lookback: int,
    min_per_class: int = 2,
) -> dict:
    """Compute PR curve at a specific lookback distance.

    Returns dict with precision, recall, auprc, prevalence.
    """
    # Gather metric values at lookback distance (same as AUROC)
    viol_vals, ctrl_vals = _gather_values_at_lookback(
        violation_events, control_events, metric_array, lookback
    )

    # Build binary labels and scores
    labels = np.concatenate([np.ones(len(viol_vals)), np.zeros(len(ctrl_vals))])
    scores = np.concatenate([viol_vals, ctrl_vals])

    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = average_precision_score(labels, scores)
    prevalence = len(viol_vals) / (len(viol_vals) + len(ctrl_vals))

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "auprc": float(auprc),
        "prevalence": float(prevalence),
    }
```

### Pattern 2: Visualization Module Pattern

Each new visualization follows the project convention:
- Function takes data + parameters, returns `plt.Figure`
- Uses `PALETTE`, `apply_style()`, `save_figure()` from `src/visualization/style.py`
- File naming: `pr_curves.py`, `calibration.py`, `svd_benchmark.py`

### Pattern 3: Render Integration Pattern

`src/visualization/render.py::render_all()` uses try/except blocks for each plot type:
```python
# Each section follows this pattern:
try:
    from src.visualization.new_module import plot_function
    fig = plot_function(data_args)
    paths = save_figure(fig, figures_dir, "prefix_name")
    generated_files.extend(paths)
    log.info("Generated: prefix_name")
except Exception as e:
    log.warning("Failed to generate prefix_name: %s", e)
```

### Pattern 4: Report Template Integration

`src/reporting/single.py::_collect_figures()` categorizes figures by filename prefix.
`src/reporting/single.py::generate_single_report()` passes data to the Jinja2 template.
New sections need:
1. New figure category keys in `_collect_figures()` result dict
2. New template variables passed to `template.render()`
3. New HTML sections in `single_report.html`

### Pattern 5: Collapsible Sections (Claude's Discretion)

Use `<details>/<summary>` HTML elements for collapsible sections -- native HTML5, no JavaScript needed, works everywhere:
```html
<details>
  <summary><h2 style="display:inline">PR Curves</h2></summary>
  <!-- content -->
</details>
```

### Anti-Patterns to Avoid
- **Modifying event_extraction.py:** Reuse existing functions, don't change them
- **Coupling PR/calibration/benchmark code:** Keep as three independent modules
- **Running SVD benchmark during normal evaluation:** Separate profiling mode only
- **Storing raw PR curve points in result.json for all metrics:** Store only AUPRC scalar + a few key points, not full interpolated curves (bloats JSON)

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PR curve computation | Manual precision/recall at thresholds | sklearn.metrics.precision_recall_curve | Handles interpolation, ties, edge cases |
| Average precision | Manual trapezoid integration | sklearn.metrics.average_precision_score | Correct interpolation method |
| Calibration binning | Manual histogram binning | sklearn.calibration.calibration_curve | Handles edge cases, empty bins |
| GPU timing | time.time() or time.perf_counter() | torch.cuda.Event | Correct GPU synchronization |

## Common Pitfalls

### Pitfall 1: PR Curve Score Direction
**What goes wrong:** AUROC is direction-invariant (higher/lower both work), but PR curves need consistent score direction -- higher score must mean more likely to be positive class.
**Why it happens:** SVD metrics may be higher or lower before violations depending on the specific metric.
**How to avoid:** For each metric, determine score direction from the AUROC analysis (if AUROC > 0.5, higher score predicts violation; if AUROC < 0.5, negate scores). Or compute AUPRC both ways and take the max.
**Warning signs:** AUPRC equal to or very close to prevalence rate.

### Pitfall 2: Insufficient Positive Samples
**What goes wrong:** PR curves are heavily affected by class imbalance. With very few violations, curves become noisy/meaningless.
**Why it happens:** Some lookback distances or metric combinations may have very few violation events.
**How to avoid:** Set minimum threshold (same min_per_class=2 as AUROC). When below threshold, report NaN/skip rather than misleading curves. Log a warning.
**Warning signs:** Jagged PR curves, AUPRC variance > 0.2 across small perturbations.

### Pitfall 3: ECE Bin Emptiness
**What goes wrong:** With 10 equal-width bins, some bins may be empty (especially extreme probability bins).
**Why it happens:** SVD metrics don't naturally produce calibrated probabilities -- the "probability" is derived from metric percentile or rank.
**How to avoid:** Handle empty bins gracefully (exclude from ECE computation, mark in visualization). Consider: what does "predicted probability" mean for an SVD metric? The natural approach is to convert metric values to probabilities via logistic regression or rank-based mapping.
**Warning signs:** Multiple empty bins, ECE dominated by a single bin.

### Pitfall 4: CUDA Event Timing Without Warmup
**What goes wrong:** First few CUDA operations are slower due to kernel compilation, memory allocation.
**Why it happens:** CUDA lazy initialization and JIT compilation.
**How to avoid:** User decision already specifies 5 warmup + 20 timed iterations. Ensure torch.cuda.synchronize() before start event and after end event.

### Pitfall 5: SVD Benchmark Matrix Dimensions
**What goes wrong:** Benchmark with wrong matrix dimensions doesn't match actual evaluation.
**Why it happens:** Matrix sizes depend on config (w for context window, d_model for embedding).
**How to avoid:** Extract actual matrix dimensions from the ExperimentConfig: QK^T is [w, w], WvWo is [d_model, d_model], AVWo is [w, d_model].

### Pitfall 6: Probability Calibration Interpretation
**What goes wrong:** Treating raw SVD metric values as probabilities for reliability diagrams.
**Why it happens:** SVD metrics are continuous values, not probabilities.
**How to avoid:** Convert SVD metric values to pseudo-probabilities using empirical CDF (rank-based) before computing calibration. The metric value at percentile p is treated as predicting P(violation) = p. This is the natural non-parametric approach.

## Code Examples

### PR Curve with sklearn
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# labels: 1 = violation, 0 = control
# scores: metric value (higher = more likely violation)
precision, recall, thresholds = precision_recall_curve(labels, scores)
auprc = average_precision_score(labels, scores)

# No-skill baseline
prevalence = labels.mean()  # P(positive)
```

### Calibration Curve with sklearn
```python
from sklearn.calibration import calibration_curve

# Convert metric values to pseudo-probabilities via rank
from scipy.stats import rankdata
prob_pred = rankdata(scores) / len(scores)

# 10 equal-width bins
fraction_of_positives, mean_predicted_value = calibration_curve(
    labels, prob_pred, n_bins=10, strategy='uniform'
)

# ECE computation
bin_counts = np.histogram(prob_pred, bins=10, range=(0, 1))[0]
total = len(labels)
ece = sum(
    (bin_counts[i] / total) * abs(fraction_of_positives[i] - mean_predicted_value[i])
    for i in range(len(fraction_of_positives))
    if bin_counts[i] > 0
)
```

### CUDA Event Timing
```python
import torch

def benchmark_svd_method(matrix, method_fn, n_warmup=5, n_timed=20):
    """Benchmark an SVD method with CUDA events."""
    device = matrix.device

    # Warmup
    for _ in range(n_warmup):
        method_fn(matrix)

    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_timed):
        method_fn(matrix)
    end_event.record()

    torch.cuda.synchronize(device)
    elapsed_ms = start_event.elapsed_time(end_event) / n_timed
    return elapsed_ms
```

### SVD Accuracy Comparison
```python
def compare_svd_accuracy(matrix, full_U, full_S, full_Vh):
    """Compare randomized/values-only SVD against full SVD."""
    # Randomized SVD
    U_rand, S_rand, V_rand = torch.svd_lowrank(matrix, q=min(matrix.shape))

    # Values-only SVD
    S_vals = torch.linalg.svdvals(matrix)

    # Relative Frobenius error: ||S_full - S_approx||_F / ||S_full||_F
    frob_error_rand = torch.norm(full_S - S_rand) / torch.norm(full_S)
    frob_error_vals = torch.norm(full_S - S_vals) / torch.norm(full_S)

    # Singular value correlation
    sv_corr_rand = torch.corrcoef(torch.stack([full_S, S_rand]))[0, 1]
    sv_corr_vals = torch.corrcoef(torch.stack([full_S, S_vals]))[0, 1]

    return {
        "randomized": {"frob_error": frob_error_rand.item(), "sv_correlation": sv_corr_rand.item()},
        "values_only": {"frob_error": frob_error_vals.item(), "sv_correlation": sv_corr_vals.item()},
    }
```

### Collapsible HTML Section
```html
<details class="enrichment-section">
  <summary><h2 style="display:inline">Precision-Recall Curves</h2></summary>
  <div class="section-content">
    <!-- PR curve content here -->
  </div>
</details>
```

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | None (uses default discovery) |
| Quick run command | `pytest tests/test_pr_curves.py tests/test_calibration.py tests/test_svd_benchmark.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PRCL-01 | PR curves + AUPRC per metric per lookback | unit | `pytest tests/test_pr_curves.py -x` | Wave 0 |
| PRCL-02 | Reliability diagrams + ECE | unit | `pytest tests/test_calibration.py -x` | Wave 0 |
| PRCL-03 | PR + calibration in HTML reports | unit | `pytest tests/test_reporting.py -x` | Exists (extend) |
| OVHD-01 | SVD cost benchmarking with CUDA events | unit | `pytest tests/test_svd_benchmark.py -x` | Wave 0 |
| OVHD-02 | Full vs randomized vs values-only comparison | unit | `pytest tests/test_svd_benchmark.py -x` | Wave 0 |
| OVHD-03 | Cost summary table in HTML | unit | `pytest tests/test_reporting.py -x` | Exists (extend) |

### Sampling Rate
- **Per task commit:** `pytest tests/test_pr_curves.py tests/test_calibration.py tests/test_svd_benchmark.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green

### Wave 0 Gaps
- [ ] `tests/test_pr_curves.py` -- covers PRCL-01
- [ ] `tests/test_calibration.py` -- covers PRCL-02
- [ ] `tests/test_svd_benchmark.py` -- covers OVHD-01, OVHD-02

## Implementation Notes

### File Organization (New Files)
```
src/analysis/pr_curves.py        # PR curve + AUPRC computation
src/analysis/calibration.py      # Reliability diagrams + ECE computation
src/analysis/svd_benchmark.py    # SVD overhead benchmarking
src/visualization/pr_curves.py   # PR curve plotting
src/visualization/calibration.py # Reliability diagram plotting
src/visualization/svd_benchmark.py # SVD benchmark bar chart
```

### Files to Modify
```
src/visualization/render.py      # Add render hooks for new plot types
src/reporting/single.py          # Add figure collection + template vars
src/reporting/templates/single_report.html  # Add collapsible sections
src/results/schema.py            # Add validation for pr_curves block
```

### result.json Schema Additions

```json
{
  "metrics": {
    "pr_curves": {
      "config": {
        "min_events_per_class": 2
      },
      "by_r_value": {
        "5": {
          "n_violations": 50,
          "n_controls": 200,
          "by_metric": {
            "qkt.layer_0.grassmannian_distance": {
              "auprc_by_lookback": [0.85, 0.72, ...],
              "prevalence": 0.2
            }
          }
        }
      }
    },
    "calibration": {
      "by_metric": {
        "qkt.layer_0.grassmannian_distance": {
          "ece_by_lookback": [0.05, 0.08, ...],
          "n_bins": 10
        }
      }
    },
    "svd_benchmark": {
      "config": {
        "n_warmup": 5,
        "n_timed": 20,
        "device": "cuda"
      },
      "by_target": {
        "qkt": {
          "matrix_shape": [64, 64],
          "full_svd_ms": 0.5,
          "randomized_svd_ms": 0.3,
          "values_only_ms": 0.1,
          "randomized_frob_error": 0.001,
          "randomized_sv_correlation": 0.999,
          "values_only_sv_correlation": 1.0
        }
      },
      "total_eval_time_ms": 100.0,
      "svd_percentage": 45.0
    }
  }
}
```

## Open Questions

1. **Score direction for PR curves**
   - What we know: Different SVD metrics may increase or decrease before violations
   - What's unclear: Best approach to determine direction per metric
   - Recommendation: Use AUROC direction -- if AUROC > 0.5 for a metric, higher score predicts violation; otherwise negate. This is consistent with existing pipeline.

2. **Probability conversion for calibration**
   - What we know: SVD metrics are not naturally probabilities
   - What's unclear: Whether rank-based conversion is sufficient or logistic regression is better
   - Recommendation: Use empirical CDF (rank-based) -- simpler, non-parametric, doesn't overfit. Platt scaling would undermine pre-registration claims per Out of Scope in REQUIREMENTS.md.

3. **SVD benchmark without GPU**
   - What we know: Benchmark requires CUDA events
   - What's unclear: How to handle CPU-only environments in tests
   - Recommendation: CPU timing fallback with time.perf_counter() for tests; CUDA events for actual benchmarks. Skip benchmark tests if no CUDA available.

## Sources

### Primary (HIGH confidence)
- PyTorch documentation: torch.linalg.svd, torch.svd_lowrank, torch.linalg.svdvals, torch.cuda.Event
- scikit-learn documentation: precision_recall_curve, average_precision_score, calibration_curve
- Existing codebase: src/analysis/auroc_horizon.py, src/analysis/event_extraction.py, src/visualization/render.py, src/reporting/single.py

### Secondary (MEDIUM confidence)
- HTML5 details/summary element: MDN Web Docs -- standard collapsible sections

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in project or standard Python scientific stack
- Architecture: HIGH -- follows established codebase patterns exactly
- Pitfalls: HIGH -- based on actual codebase analysis and known SVD/ML pitfalls

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable domain, well-established libraries)
