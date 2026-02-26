# Phase 8: Visualization - Research

**Researched:** 2026-02-26
**Domain:** Publication-quality static figure generation from scientific analysis results
**Confidence:** HIGH

## Summary

Phase 8 generates all publication-quality static figures from the analysis pipeline output (result.json + token_metrics.npz). The codebase already has all data sources: training curves (loss, edge_compliance, rule_compliance) in result.json metrics.curves, SVD metric time series in token_metrics.npz, AUROC curves and predictive horizons from the analysis module, behavioral labels (4-class outcomes), and statistical controls (bootstrap CIs, Cohen's d, correlation matrices). The visualization module reads this data and renders it using matplotlib + seaborn.

matplotlib and seaborn are NOT currently installed. They must be added to pyproject.toml dependencies and pip-installed before any plotting code runs.

**Primary recommendation:** Create a `src/visualization/` module with a style module for consistent theming, individual plot functions per requirement (PLOT-01 through PLOT-06), and a `render_all()` orchestrator that reads result.json + token_metrics.npz and writes PNG/SVG to `results/{experiment_id}/figures/`.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use seaborn whitegrid as specified in requirements
- Consistent palette across all plot types
- Standard academic figure conventions

### Claude's Discretion
- Color palette selection (colorblind-safe recommended)
- Font family and sizes for labels, titles, legends
- Figure dimensions and aspect ratios
- Confidence band style (shaded regions vs error bars)
- Distribution plot type (violin, box, histogram)
- Heatmap colormap choice
- Subplot arrangement and multi-panel layout
- Whether figures include titles or are left bare for LaTeX captions
- File naming convention within figures/ directory
- Subplot letter labels (a, b, c) if multi-panel

User explicitly deferred all visual polish decisions -- will refine after initial implementation.

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PLOT-01 | Event-aligned SVD metric plots (position 0 = failure event, neg before, pos after) with confidence bands and correct-sequence baseline overlay | SVD metrics in token_metrics.npz keyed as `{target}.layer_{N}.{metric_name}`, event extraction via `AnalysisEvent` with `resolution_step`, failure_index for alignment |
| PLOT-02 | Training convergence curves (loss and compliance over steps) | result.json `metrics.curves.train_loss`, `metrics.curves.edge_compliance`, `metrics.curves.rule_compliance` |
| PLOT-03 | AUROC vs lookback distance j curves per SVD metric | `auroc_by_lookback` arrays per metric per r-value from `run_auroc_analysis()` output |
| PLOT-04 | Confusion matrix for 4-class behavioral outcomes | `rule_outcome` + `edge_valid` arrays in token_metrics.npz; 4 classes: edge_valid+rule_followed, edge_valid+rule_violated, edge_invalid+rule_followed, edge_invalid+rule_violated |
| PLOT-05 | Pre/post failure distribution comparison plots | SVD metric values split by position relative to failure_index; use violin/box plots |
| PLOT-06 | Predictive horizon heatmap across (r, w) parameter grid | Requires loading multiple result.json files across sweep configs; horizon values from `compute_predictive_horizon()` |
| PLOT-07 | All plots follow project style baseline (seaborn whitegrid, consistent palette) | Single style module with `apply_style()` and named palette constants |
| PLOT-08 | All figures saved as both PNG (300 dpi) and SVG to results/{experiment_id}/figures/ | `save_figure()` helper that calls `savefig` twice with format parameter |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | >=3.8 | Base plotting engine | Standard Python plotting; required by seaborn |
| seaborn | >=0.13 | Statistical visualization and theming | Required by PLOT-07 (whitegrid style); provides violin plots, heatmaps |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | >=2.0 (installed) | Array manipulation for plot data | Already used throughout project |
| scipy | >=1.14 (installed) | Statistical computations | Already used for bootstrap, rankdata |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib+seaborn | plotly | Interactive but requirements specify static figures with SVG/PNG |
| matplotlib+seaborn | altair | Declarative but less control over publication formatting |

**Installation:**
```bash
pip install matplotlib>=3.8 seaborn>=0.13
```

## Architecture Patterns

### Recommended Project Structure
```
src/
├── visualization/
│   ├── __init__.py
│   ├── style.py          # apply_style(), save_figure(), palette constants
│   ├── event_aligned.py  # PLOT-01: SVD metric event-aligned plots
│   ├── training.py       # PLOT-02: Training convergence curves
│   ├── auroc.py          # PLOT-03: AUROC vs lookback curves
│   ├── confusion.py      # PLOT-04: Confusion matrix
│   ├── distributions.py  # PLOT-05: Pre/post failure distributions
│   ├── heatmap.py        # PLOT-06: Predictive horizon heatmap
│   └── render.py         # Orchestrator: render_all() reads result data, calls all plots
```

### Pattern 1: Style-First Architecture
**What:** A single `apply_style()` function called at import time or start of rendering that sets seaborn whitegrid, configures matplotlib rcParams (font sizes, figure size, DPI), and defines the project color palette. Every plot function receives the palette as a parameter or imports it from style.py.
**When to use:** Always -- ensures PLOT-07 compliance across all figures.
**Example:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Project palette: colorblind-safe (tab10 subset or custom)
PALETTE = sns.color_palette("colorblind", n_colors=8)
VIOLATION_COLOR = PALETTE[3]  # red-ish
CONTROL_COLOR = PALETTE[0]    # blue-ish

def apply_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.figsize": (8, 5),
    })
```

### Pattern 2: Dual-Format Save Helper
**What:** A helper function that saves every figure as both PNG (300 dpi) and SVG, ensuring PLOT-08 compliance without duplicating savefig calls in every plot function.
**Example:**
```python
def save_figure(fig, output_dir: Path, name: str):
    """Save figure as PNG (300 dpi) and SVG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
```

### Pattern 3: Data Loading Separation
**What:** Each plot function receives pre-loaded data as arguments (numpy arrays, dicts), not file paths. The `render_all()` orchestrator handles loading result.json and token_metrics.npz once, then passes the relevant slices to each plot function. This makes plot functions independently testable.

### Anti-Patterns to Avoid
- **Embedding file I/O in plot functions:** Makes testing require real files; pass arrays instead
- **Hardcoding style per plot:** Style drift; use centralized apply_style()
- **Forgetting plt.close(fig):** Memory leaks when generating many figures
- **Using plt.show() in library code:** Blocks execution; only save

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Confidence bands on time series | Manual fill_between calculation | `ax.fill_between()` with pre-computed CI arrays | Correct alpha blending, proper z-ordering |
| Heatmap with annotations | Manual text placement | `sns.heatmap(annot=True)` | Handles cell text alignment, colorbar, formatting |
| Confusion matrix display | Custom grid plotting | `sns.heatmap()` on normalized matrix | Standard layout, proper coloring |
| Violin/box plots | Manual kernel density | `sns.violinplot()` or `sns.boxplot()` | Correct bandwidth, proper statistics display |

**Key insight:** seaborn handles the statistical visualization patterns needed here (heatmaps, violins, distribution comparisons) with less code and fewer bugs than manual matplotlib.

## Common Pitfalls

### Pitfall 1: NaN in Plot Data
**What goes wrong:** SVD metrics have NaN for positions < w (warmup) and potentially for degenerate matrices. Plotting NaN values causes gaps or crashes.
**Why it happens:** token_metrics.npz is NaN-filled by default; only positions >= w have valid data.
**How to avoid:** Always filter with `np.isfinite()` before plotting. For event-aligned plots, use `np.nanmean()` / `np.nanstd()` for aggregation.
**Warning signs:** Empty plots or matplotlib warnings about NaN.

### Pitfall 2: Event Alignment Off-by-One
**What goes wrong:** Event-aligned plots show wrong alignment if the resolution_step convention doesn't match array indexing.
**Why it happens:** resolution_step is 1-based step numbering while arrays are 0-indexed. The convention is: metric at position `resolution_step - j` for lookback j (already 0-indexed into the array).
**How to avoid:** Use the same indexing as `compute_auroc_curve()`: idx = event.resolution_step - j, where j ranges 1..r. Position 0 on the plot = resolution_step (the failure event).
**Warning signs:** Baseline and violation curves don't diverge at expected position.

### Pitfall 3: SVG Text Rendering
**What goes wrong:** SVG output uses system fonts that may not render correctly on other machines.
**Why it happens:** matplotlib embeds font references rather than paths.
**How to avoid:** Use `plt.rcParams['svg.fonttype'] = 'none'` to embed text as SVG text elements (not paths). For maximum portability, could use `'path'` to convert text to paths.
**Warning signs:** Missing or substituted fonts in SVG viewers.

### Pitfall 4: Figure Size for Multi-Panel Layouts
**What goes wrong:** Subplots are too cramped or labels overlap.
**Why it happens:** Using default figure size for multi-panel figures.
**How to avoid:** Scale figure size with number of subplots. Use `fig.tight_layout()` or `constrained_layout=True`.

### Pitfall 5: Heatmap with Single Data Point
**What goes wrong:** The (r, w) heatmap requires multiple sweep configurations but anchor config provides only one data point.
**Why it happens:** Phase 10 (sweep execution) hasn't run yet.
**How to avoid:** The heatmap function must handle a sparse grid gracefully -- display available data points even if grid is mostly empty. Success criteria says "at least the anchor config data point."

## Code Examples

### Event-Aligned SVD Plot (PLOT-01 core pattern)
```python
def plot_event_aligned(
    metric_values: np.ndarray,  # [n_sequences, max_steps]
    events: list[AnalysisEvent],
    failure_index: np.ndarray,
    window: int,  # how many steps before/after to show
    metric_name: str,
    ax: plt.Axes | None = None,
):
    """Plot SVD metric aligned to failure events.

    Position 0 = failure event (resolution_step).
    Negative positions = before failure.
    Positive positions = after failure.
    """
    if ax is None:
        fig, ax = plt.subplots()

    violations = [e for e in events if e.outcome == RuleOutcome.VIOLATED]
    controls = [e for e in events if e.outcome == RuleOutcome.FOLLOWED]

    positions = np.arange(-window, window + 1)

    # Collect values at each relative position
    for group, label, color in [
        (violations, "Violation", VIOLATION_COLOR),
        (controls, "Control", CONTROL_COLOR),
    ]:
        aligned = np.full((len(group), len(positions)), np.nan)
        for i, ev in enumerate(group):
            for j, pos in enumerate(positions):
                idx = ev.resolution_step + pos
                if 0 <= idx < metric_values.shape[1]:
                    aligned[i, j] = metric_values[ev.walk_idx, idx]

        mean = np.nanmean(aligned, axis=0)
        std = np.nanstd(aligned, axis=0)
        n = np.sum(np.isfinite(aligned), axis=0)
        se = std / np.sqrt(np.maximum(n, 1))

        ax.plot(positions, mean, label=label, color=color)
        ax.fill_between(positions, mean - 1.96 * se, mean + 1.96 * se,
                        alpha=0.2, color=color)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.7, label="Failure event")
    ax.set_xlabel("Position relative to failure event")
    ax.set_ylabel(metric_name)
    ax.legend()
```

### Training Convergence (PLOT-02 core pattern)
```python
def plot_training_curves(
    curves: dict,  # from result.json metrics.curves
    ax_loss: plt.Axes,
    ax_compliance: plt.Axes,
):
    """Plot training loss and compliance curves."""
    steps = range(len(curves["train_loss"]))
    ax_loss.plot(steps, curves["train_loss"], color=PALETTE[0])
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Cross-entropy loss")

    epochs = range(len(curves["edge_compliance"]))
    ax_compliance.plot(epochs, curves["edge_compliance"], label="Edge compliance")
    ax_compliance.plot(epochs, curves["rule_compliance"], label="Rule compliance")
    ax_compliance.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
    ax_compliance.axhline(0.80, color="gray", linestyle=":", alpha=0.5)
    ax_compliance.legend()
```

### AUROC Curve (PLOT-03 core pattern)
```python
def plot_auroc_curve(
    auroc_by_lookback: list[float],
    metric_name: str,
    horizon: int,
    threshold: float = 0.75,
    ax: plt.Axes | None = None,
):
    """Plot AUROC vs lookback distance j."""
    lookbacks = np.arange(1, len(auroc_by_lookback) + 1)
    ax.plot(lookbacks, auroc_by_lookback, marker="o", markersize=4)
    ax.axhline(threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")
    ax.axhline(0.5, color="lightgray", linestyle=":", alpha=0.5, label="Chance")
    if horizon > 0:
        ax.axvline(horizon, color="green", linestyle="--", alpha=0.5, label=f"Horizon (j={horizon})")
    ax.set_xlabel("Lookback distance j")
    ax.set_ylabel("AUROC")
    ax.set_title(metric_name)
    ax.legend(fontsize=8)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| matplotlib only | matplotlib + seaborn | 2012+ | Cleaner defaults, better statistical plots |
| `plt.style.use()` | `sns.set_theme()` | seaborn 0.11+ | Integrates with matplotlib rcParams |
| Manual DPI handling | `savefig(dpi=300)` | Always available | Standard practice for publication figures |

**Deprecated/outdated:**
- `seaborn.set()` → replaced by `sns.set_theme()` in seaborn 0.11+
- `seaborn.distplot()` → replaced by `sns.histplot()` / `sns.kdeplot()` in seaborn 0.11+

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.2 |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `.venv/bin/pytest tests/test_visualization.py -x` |
| Full suite command | `.venv/bin/pytest tests/ -x` |
| Estimated runtime | ~5 seconds |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PLOT-01 | Event-aligned SVD metric plot generation | unit | `.venv/bin/pytest tests/test_visualization.py::test_event_aligned_plot -x` | No -- Wave 0 gap |
| PLOT-02 | Training convergence curve generation | unit | `.venv/bin/pytest tests/test_visualization.py::test_training_curves -x` | No -- Wave 0 gap |
| PLOT-03 | AUROC vs lookback curve generation | unit | `.venv/bin/pytest tests/test_visualization.py::test_auroc_curve -x` | No -- Wave 0 gap |
| PLOT-04 | Confusion matrix generation | unit | `.venv/bin/pytest tests/test_visualization.py::test_confusion_matrix -x` | No -- Wave 0 gap |
| PLOT-05 | Distribution comparison plot generation | unit | `.venv/bin/pytest tests/test_visualization.py::test_distribution_plot -x` | No -- Wave 0 gap |
| PLOT-06 | Predictive horizon heatmap generation | unit | `.venv/bin/pytest tests/test_visualization.py::test_horizon_heatmap -x` | No -- Wave 0 gap |
| PLOT-07 | Style consistency (whitegrid, palette) | unit | `.venv/bin/pytest tests/test_visualization.py::test_style_application -x` | No -- Wave 0 gap |
| PLOT-08 | Dual-format save (PNG + SVG) | unit | `.venv/bin/pytest tests/test_visualization.py::test_save_dual_format -x` | No -- Wave 0 gap |

### Nyquist Sampling Rate
- **Minimum sample interval:** After every committed task run: `.venv/bin/pytest tests/test_visualization.py -x`
- **Full suite trigger:** Before final task of each plan wave
- **Phase-complete gate:** Full suite green before verification
- **Estimated feedback latency per task:** ~5 seconds

### Wave 0 Gaps (must be created before implementation)
- [ ] `tests/test_visualization.py` -- covers PLOT-01 through PLOT-08
- [ ] `src/visualization/__init__.py` -- package initialization
- [ ] Install: `pip install matplotlib>=3.8 seaborn>=0.13`

## Open Questions

1. **Exact (r, w) grid dimensions for heatmap**
   - What we know: Anchor config has r=57, w=64. Sweep in Phase 10 will vary these.
   - What's unclear: How many grid cells will be populated before Phase 10 runs.
   - Recommendation: Design heatmap to accept sparse data; fill single cell for anchor config. Use `sns.heatmap()` with NaN masking for empty cells.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/results/schema.py`, `src/analysis/auroc_horizon.py`, `src/analysis/statistical_controls.py`, `src/evaluation/pipeline.py`, `src/training/pipeline.py` -- all data structures verified
- Codebase: `pyproject.toml` -- confirmed matplotlib/seaborn not in dependencies

### Secondary (MEDIUM confidence)
- matplotlib/seaborn API conventions -- from training data, well-established stable APIs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- matplotlib + seaborn is the universal Python scientific plotting stack
- Architecture: HIGH -- clear data flow from existing modules to visualization functions
- Pitfalls: HIGH -- NaN handling and alignment conventions verified against actual code

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable domain, no fast-moving APIs)
