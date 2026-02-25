# Plotting Guide

All plots are generated from `result.json` files. Never generate plots directly
from in-memory experiment state — always write results first, then plot from the
stored JSON. This guarantees every figure is reproducible independently of the
experiment code.

---

## Conventions

### File Output

- Save all figures to `results/{experiment_id}/figures/`
- Save as both `.png` (300 dpi, for reports) and `.svg` (for editing)
- Filenames should be descriptive: `entropy_aligned_on_failure.png`, not `fig1.png`

```python
def save_fig(fig, name: str, experiment_id: str, results_dir: str = "results"):
    import os
    fig_dir = os.path.join(results_dir, experiment_id, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for ext in ["png", "svg"]:
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"),
                    dpi=300, bbox_inches="tight")
```

### Style Baseline

Apply this at the top of every plotting module. Do not use the default matplotlib style.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titleweight":  "bold",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.frameon":    False,
    "legend.fontsize":   9,
})
```

### Colour Palette

Use a consistent palette across all experiments so comparisons are readable.

```python
PALETTE = {
    "primary":      "#2E75B6",   # blue   — main series / correct
    "danger":       "#C00000",   # red    — failure / hallucination
    "secondary":    "#595959",   # grey   — baseline / reference
    "highlight":    "#F4A300",   # amber  — points of interest
    "positive":     "#1A6B3C",   # green  — improvement
    "ci_fill":      "#D6E4F0",   # light blue — confidence bands
}

# For multi-experiment comparisons use:
COMPARISON_PALETTE = sns.color_palette("colorblind", 8)
```

**Rule:** correct/baseline traces = `primary`, hallucination/failure traces = `danger`.
Never use red for anything other than failure.

---

## Plot Types

### 1. Scalar Metrics Bar Chart

```python
import json, matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_scalars(result_path: str):
    with open(result_path) as f:
        r = json.load(f)
    scalars = r["metrics"]["scalars"]

    fig, ax = plt.subplots(figsize=(max(4, len(scalars) * 1.2), 4))
    bars = ax.bar(scalars.keys(), scalars.values(),
                  color=PALETTE["primary"], width=0.5, zorder=3)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(r["description"])
    fig.tight_layout()
    return fig
```

### 2. Training / Evaluation Curves

```python
def plot_curve(result_path: str, curve_name: str):
    with open(result_path) as f:
        r = json.load(f)
    curve = r["metrics"]["curves"][curve_name]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["x"], curve["y"], color=PALETTE["primary"], linewidth=1.8)
    ax.set_xlabel(curve["x_label"])
    ax.set_ylabel(curve["y_label"])
    ax.set_title(curve_name.replace("_", " ").title())
    fig.tight_layout()
    return fig
```

### 3. Confusion Matrix

```python
import numpy as np
import seaborn as sns

def plot_confusion_matrix(result_path: str):
    with open(result_path) as f:
        r = json.load(f)
    cm_data = r["metrics"]["confusion_matrix"]
    matrix = np.array(cm_data["matrix"])
    labels = cm_data["labels"]

    # Normalise to proportions for the colour scale; show raw counts as text
    norm = matrix.astype(float) / matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(norm, annot=matrix, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Row proportion"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig
```

### 4. Event-Aligned Token Statistics (Hallucination Alignment)

This is the primary plot type for LLM hallucination analysis. It aligns all
sequences on their `failure_index` and plots a chosen token-level statistic
(e.g. entropy, log-probability) as a function of relative position.

**Alignment convention:** position 0 = failure event, negative = before, positive = after.
Sequences with `failure_index = null` are treated as the correct/baseline group.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_aligned_statistic(
    result_path: str,
    statistic: str = "token_entropy",   # or "token_logprobs"
    window: tuple = (-10, 10),           # positions relative to failure
    min_sequences: int = 5,              # skip positions with fewer sequences
    show_baseline: bool = True,          # overlay correct sequences
):
    with open(result_path) as f:
        r = json.load(f)

    w_pre, w_post = window
    positions = list(range(w_pre, w_post + 1))

    # Collect values at each relative position for each group
    failed_vals  = defaultdict(list)
    correct_vals = defaultdict(list)

    for seq in r["sequences"]:
        values = seq.get(statistic)
        if not values:
            continue

        if seq["label"] == "hallucinated" and seq["failure_index"] is not None:
            fi = seq["failure_index"]
            for pos in positions:
                abs_idx = fi + pos
                if 0 <= abs_idx < len(values):
                    failed_vals[pos].append(values[abs_idx])
        elif seq["label"] == "correct" and show_baseline:
            # For correct sequences use the midpoint as the pseudo-anchor
            anchor = len(values) // 2
            for pos in positions:
                abs_idx = anchor + pos
                if 0 <= abs_idx < len(values):
                    correct_vals[pos].append(values[abs_idx])

    def summarise(vals_dict):
        xs, means, lo, hi = [], [], [], []
        for pos in positions:
            v = vals_dict.get(pos, [])
            if len(v) < min_sequences:
                continue
            arr = np.array(v)
            xs.append(pos)
            means.append(arr.mean())
            se = arr.std() / np.sqrt(len(arr))
            lo.append(arr.mean() - 1.96 * se)
            hi.append(arr.mean() + 1.96 * se)
        return np.array(xs), np.array(means), np.array(lo), np.array(hi)

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Failure group
    xs, means, lo, hi = summarise(failed_vals)
    if len(xs):
        ax.plot(xs, means, color=PALETTE["danger"], linewidth=2,
                label="Hallucinated", zorder=3)
        ax.fill_between(xs, lo, hi, color=PALETTE["danger"],
                        alpha=0.15, zorder=2)

    # Correct baseline
    if show_baseline:
        xs, means, lo, hi = summarise(correct_vals)
        if len(xs):
            ax.plot(xs, means, color=PALETTE["primary"], linewidth=2,
                    linestyle="--", label="Correct", zorder=3)
            ax.fill_between(xs, lo, hi, color=PALETTE["ci_fill"],
                            alpha=0.4, zorder=2)

    # Failure event marker
    ax.axvline(0, color=PALETTE["danger"], linewidth=1.2,
               linestyle=":", alpha=0.7, label="Failure event (pos 0)")

    y_label = statistic.replace("_", " ").title()
    ax.set_xlabel("Token position relative to failure")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} Aligned on Hallucination Onset\n"
                 f"(n_failed={sum(len(v) for v in failed_vals.values() if len(v) >= min_sequences)})")
    ax.legend()
    ax.set_xticks(positions[::2])
    fig.tight_layout()
    return fig
```

### 5. Per-Sequence Token Heatmap

Useful for visualising a single generation — shows how a statistic evolves token
by token, with the failure point marked.

```python
def plot_sequence_heatmap(
    result_path: str,
    sequence_id: str,
    statistic: str = "token_entropy",
):
    with open(result_path) as f:
        r = json.load(f)

    seq = next((s for s in r["sequences"] if s["sequence_id"] == sequence_id), None)
    assert seq, f"Sequence {sequence_id} not found"

    values  = seq[statistic]
    tokens  = seq["tokens"]
    failure = seq.get("failure_index")

    fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.4), 1.8))
    data = np.array(values).reshape(1, -1)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.4,
                 label=statistic.replace("_", " ").title())

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])

    if failure is not None:
        ax.axvline(failure - 0.5, color=PALETTE["danger"],
                   linewidth=2, label="Failure")
        ax.legend(loc="upper left", fontsize=8)

    ax.set_title(f"Sequence {sequence_id} — {statistic.replace('_', ' ').title()}")
    fig.tight_layout()
    return fig
```

### 6. Distribution Comparison (Before / After Failure)

Compare the distribution of a statistic in the window before vs. after failure.

```python
def plot_pre_post_distribution(
    result_path: str,
    statistic: str = "token_entropy",
    pre_window: int = 5,
    post_window: int = 5,
):
    with open(result_path) as f:
        r = json.load(f)

    pre_vals, post_vals = [], []
    for seq in r["sequences"]:
        values = seq.get(statistic)
        fi = seq.get("failure_index")
        if not values or fi is None:
            continue
        pre_vals.extend(values[max(0, fi - pre_window):fi])
        post_vals.extend(values[fi:fi + post_window])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pre_vals,  bins=30, alpha=0.6, color=PALETTE["primary"],
            label=f"Pre-failure (n={len(pre_vals)})",  density=True)
    ax.hist(post_vals, bins=30, alpha=0.6, color=PALETTE["danger"],
            label=f"Post-failure (n={len(post_vals)})", density=True)
    ax.set_xlabel(statistic.replace("_", " ").title())
    ax.set_ylabel("Density")
    ax.set_title("Token Statistic Distribution: Pre vs. Post Failure")
    ax.legend()
    fig.tight_layout()
    return fig
```

---

## All-at-Once: Generate All Standard Figures

Call this after every experiment. It detects which plot types are applicable
based on what the result file contains.

```python
def generate_all_figures(result_path: str, results_dir: str = "results"):
    import json, os
    with open(result_path) as f:
        r = json.load(f)
    eid = r["experiment_id"]

    figures = {}

    if r["metrics"].get("scalars"):
        figures["scalars"] = plot_scalars(result_path)

    for curve_name in r["metrics"].get("curves", {}):
        figures[f"curve_{curve_name}"] = plot_curve(result_path, curve_name)

    if r["metrics"].get("confusion_matrix"):
        figures["confusion_matrix"] = plot_confusion_matrix(result_path)

    if r.get("sequences"):
        seqs_with_stat = [s for s in r["sequences"] if s.get("token_entropy")]
        if seqs_with_stat:
            figures["entropy_aligned"] = plot_aligned_statistic(
                result_path, statistic="token_entropy")
            figures["logprob_aligned"] = plot_aligned_statistic(
                result_path, statistic="token_logprobs")
            figures["entropy_pre_post"] = plot_pre_post_distribution(
                result_path, statistic="token_entropy")

    for name, fig in figures.items():
        save_fig(fig, name, eid, results_dir)
        plt.close(fig)

    print(f"Generated {len(figures)} figures for {eid}")
    return list(figures.keys())
```

---

## Adding a New Plot Type

1. Write the function following the patterns above — takes `result_path` as input,
   returns a `fig` object.
2. Add it to `generate_all_figures` with an appropriate guard condition.
3. Document it in this file under the relevant section.
4. Never hardcode data — always read from `result.json`.
# scaffold
# Reporting Guide

Reports are generated from stored `result.json` files and pre-generated figures.
They are never generated directly from live experiment state.

Two report types are supported:
- **Single-experiment report** — a full summary of one run, rendered as HTML
- **Comparison report** — a side-by-side view of multiple runs, for iteration analysis

---

## Single-Experiment Report

### Output

```
results/{experiment_id}/report.html
```

Self-contained HTML file — all figures are embedded as base64 so the file is
portable and can be emailed or archived without a separate figures folder.

### Generating

```python
# generate_report.py
python generate_report.py results/hallucination_baseline_20260223_142301/result.json
```

Or from within an experiment script:

```python
from reporting import generate_single_report
generate_single_report("results/hallucination_baseline_20260223_142301/result.json")
```

### Report Structure

Each single-experiment report contains the following sections in order:

1. **Header** — experiment ID, timestamp, description, tags, code hash
2. **Configuration** — rendered config block (key/value table)
3. **Scalar Metrics** — bar chart + table of all scalar values
4. **Curves** — one chart per entry in `metrics.curves`
5. **Confusion Matrix** — heatmap (if present)
6. **Statistical Tests** — table of all tests with significance flags
7. **Sequence Analysis** *(if sequences present)*
   - Hallucination rate and label breakdown
   - Entropy aligned on failure (with correct baseline)
   - Log-probability aligned on failure
   - Pre/post failure distribution comparison
   - Sample sequence heatmaps (up to 6: 3 hallucinated, 3 correct)
8. **Reproduction** — command to reproduce the run exactly

### Implementation

```python
import json, os, base64
from datetime import datetime

def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def generate_single_report(result_path: str, results_dir: str = "results"):
    from plotting import generate_all_figures   # generates and saves figures first

    with open(result_path) as f:
        r = json.load(f)

    eid = r["experiment_id"]
    fig_dir = os.path.join(results_dir, eid, "figures")

    # Ensure figures exist
    if not os.path.exists(fig_dir) or not os.listdir(fig_dir):
        generate_all_figures(result_path, results_dir)

    # Load figures as base64
    figures = {}
    for fname in os.listdir(fig_dir):
        if fname.endswith(".png"):
            key = fname.replace(".png", "")
            figures[key] = img_to_b64(os.path.join(fig_dir, fname))

    html = _render_single_report(r, figures)
    out_path = os.path.join(results_dir, eid, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report → {out_path}")
    return out_path
```

---

## Comparison Report

Compares two or more experiments side-by-side. Used when iterating on a method
and reviewing what changed between runs.

### Output

```
results/comparisons/{comparison_id}/comparison.html
```

`comparison_id` is auto-generated from the experiment IDs being compared:
`compare_{exp_a_slug}_vs_{exp_b_slug}_{timestamp}`

### Generating

```python
from reporting import generate_comparison_report

generate_comparison_report([
    "results/hallucination_baseline_20260223_142301/result.json",
    "results/hallucination_entropy_threshold_20260224_091500/result.json",
])
```

Or from the CLI:

```bash
python compare_results.py \
    results/hallucination_baseline_20260223_142301/result.json \
    results/hallucination_entropy_threshold_20260224_091500/result.json
```

### Comparison Report Structure

1. **Header** — list of experiments being compared, their timestamps and descriptions
2. **Scalar Metrics Table** — all experiments as columns, all scalar metrics as rows.
   Highest value per metric highlighted in green, lowest in red.
3. **Scalar Metrics Chart** — grouped bar chart across all experiments
4. **Curve Overlays** — for each curve present in any experiment, all experiments
   overlaid on the same axes (using `COMPARISON_PALETTE`)
5. **Confusion Matrices** — side-by-side (if present in any experiment)
6. **Aligned Sequence Plots** — entropy and log-prob aligned on failure, all
   experiments overlaid, using the same window
7. **Config Diff** — table showing config values that differ across experiments,
   making it easy to see what changed between runs
8. **Statistical Comparison** — if applicable, pairwise significance tests between
   the hallucination rates or key metrics of the experiments

### Scalar Comparison Table (reference layout)

| Metric           | baseline_0223 | entropy_thresh_0224 | Δ (best vs baseline) |
|------------------|:-------------:|:-------------------:|:--------------------:|
| auroc            | 0.847         | **0.881**           | +0.034 ↑             |
| f1               | 0.713         | **0.741**           | +0.028 ↑             |
| hallucination_rate | 0.179       | **0.148**           | −0.031 ↑             |

Bold = best. Δ column compares best performer to the first (baseline) experiment.

### Config Diff Table (reference layout)

| Parameter         | baseline_0223              | entropy_thresh_0224         |
|-------------------|----------------------------|-----------------------------|
| temperature       | 0.0                        | 0.0                         |
| detection_method  | —                          | entropy_threshold           |
| threshold         | —                          | 2.5                         |
| code_hash         | a3f9c1d                    | b7e2a4f                     |

Only rows where at least one value differs are shown.

---

## HTML Report Template

Use this baseline HTML structure. Embed all CSS inline so the report is
fully self-contained.

```python
REPORT_CSS = """
  body { font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto;
         color: #222; background: #fff; }
  h1   { color: #1F4E79; border-bottom: 2px solid #2E75B6; padding-bottom: 6px; }
  h2   { color: #2E75B6; margin-top: 36px; }
  h3   { color: #595959; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
  th   { background: #1F4E79; color: white; padding: 8px 12px; text-align: left; }
  td   { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #EBF3FB; }
  .metric-best  { font-weight: bold; color: #1A6B3C; }
  .metric-worst { color: #C00000; }
  .tag  { display: inline-block; background: #D6E4F0; color: #1F4E79;
          border-radius: 3px; padding: 2px 8px; font-size: 11px; margin: 2px; }
  .warn { background: #FFF0F0; border-left: 4px solid #C00000;
          padding: 10px 16px; color: #C00000; font-size: 13px; }
  .figure { margin: 20px 0; text-align: center; }
  .figure img { max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }
  .figure caption { font-size: 11px; color: #888; margin-top: 4px; display: block; }
  pre  { background: #F5F5F5; padding: 12px; border-radius: 4px;
         font-size: 12px; overflow-x: auto; }
  code { font-family: 'Courier New', monospace; background: #F5F5F5;
         padding: 1px 4px; border-radius: 2px; font-size: 12px; }
"""

def _html_figure(b64: str, caption: str) -> str:
    return f"""
    <div class="figure">
      <img src="data:image/png;base64,{b64}" alt="{caption}">
      <caption>{caption}</caption>
    </div>"""

def _html_scalars_table(experiments: list) -> str:
    """experiments: list of result dicts"""
    all_metrics = set()
    for r in experiments:
        all_metrics.update(r["metrics"].get("scalars", {}).keys())

    rows = []
    for metric in sorted(all_metrics):
        vals = [r["metrics"]["scalars"].get(metric) for r in experiments]
        valid = [v for v in vals if v is not None]
        best  = max(valid) if valid else None
        worst = min(valid) if valid else None
        cells = []
        for v in vals:
            if v is None:
                cells.append("<td>—</td>")
            elif v == best and len(valid) > 1:
                cells.append(f'<td class="metric-best">{v:.4f}</td>')
            elif v == worst and len(valid) > 1:
                cells.append(f'<td class="metric-worst">{v:.4f}</td>')
            else:
                cells.append(f"<td>{v:.4f}</td>")
        rows.append(f"<tr><td><code>{metric}</code></td>{''.join(cells)}</tr>")

    headers = "".join(f"<th>{r['experiment_id']}</th>" for r in experiments)
    return f"""
    <table>
      <tr><th>Metric</th>{headers}</tr>
      {''.join(rows)}
    </table>"""
```

---

## Reproducibility Block

Every report ends with a reproduction block. This should be auto-generated
from the stored config and code hash.

```
## Reproducing This Experiment

git checkout a3f9c1d
python run_experiment.py \
    --model meta-llama/Llama-3-8B-Instruct \
    --dataset gsm8k \
    --split test \
    --n_samples 500 \
    --seed 42 \
    --temperature 0.0
```

Generate it programmatically:

```python
def reproduction_block(r: dict) -> str:
    cfg = r["config"]
    args = " \\\n    ".join(f"--{k} {v}" for k, v in cfg.items()
                             if k not in ("code_hash", "description", "tags"))
    return f"""
## Reproducing This Experiment

```
git checkout {cfg.get('code_hash', 'unknown')}
python run_experiment.py \\
    {args}
```
"""
```

---

## Running Reports via Make

Add these targets to the project `Makefile`:

```makefile
RESULT ?= $(shell ls -t results/*/result.json 2>/dev/null | head -1)

report:
	python generate_report.py $(RESULT)

compare:
	@echo "Usage: make compare RESULTS='results/exp_a/result.json results/exp_b/result.json'"
	python compare_results.py $(RESULTS)

report-all:
	@for f in results/*/result.json; do python generate_report.py $$f; done
```

Usage:

```bash
make report                              # report for most recent experiment
make report RESULT=results/exp_abc/result.json
make compare RESULTS="results/a/result.json results/b/result.json"
make report-all                          # regenerate all reports
```

---

## Adding a New Section to Reports

1. Add the data to the relevant block in `result.json` (see `RESULTS_SCHEMA.md`)
2. If it needs a figure, add a plot function to `PLOTTING_GUIDE.md` and
   `generate_all_figures`
3. Add a rendering function in `reporting.py` that reads from the result dict
4. Insert the section call in `_render_single_report` and/or `_render_comparison`
5. Update this document

The guiding rule: reports are **views over stored data**, never the source of truth.
# Experiment Results Schema

All experiments **must** write results to a JSON file conforming to this schema.
Plots, reports, and comparisons are always generated from these files — never from
in-memory state or ad-hoc outputs. This is the contract that makes iteration fast.

---

## File Naming & Location

```
results/
  {experiment_id}/
    result.json          # canonical output — always present
    figures/             # generated plots (png, svg) — regenerable from result.json
    report.html          # generated report — regenerable from result.json
```

`experiment_id` format: `{slug}_{YYYYMMDD}_{HHMMSS}` e.g. `hallucination_baseline_20260223_142301`

Never overwrite a `result.json`. Each run produces its own timestamped directory.

---

## Top-Level Schema

```json
{
  "schema_version": "1.0",
  "experiment_id":  "hallucination_baseline_20260223_142301",
  "timestamp":      "2026-02-23T14:23:01Z",
  "description":    "Baseline hallucination detection on GSM8K using Llama-3-8B",
  "tags":           ["hallucination", "baseline", "llama3"],

  "config":    { ... },
  "metrics":   { ... },
  "sequences": [ ... ],
  "metadata":  { ... }
}
```

| Field            | Type     | Required | Notes                                              |
|------------------|----------|----------|----------------------------------------------------|
| `schema_version` | string   | ✓        | Bump minor on additive changes, major on breaks    |
| `experiment_id`  | string   | ✓        | Unique, matches directory name                     |
| `timestamp`      | ISO 8601 | ✓        | UTC, when the run completed                        |
| `description`    | string   | ✓        | One sentence. What was tested, on what, with what. |
| `tags`           | string[] | ✓        | Used for filtering in comparison reports           |
| `config`         | object   | ✓        | Full reproducibility config (see below)            |
| `metrics`        | object   | ✓        | All quantitative outputs (see below)               |
| `sequences`      | array    | —        | Token sequences; required for LLM/hallucination work |
| `metadata`       | object   | —        | Freeform provenance info                           |

---

## `config` Block

Everything needed to reproduce the run exactly.

```json
"config": {
  "model":      "meta-llama/Llama-3-8B-Instruct",
  "dataset":    "gsm8k",
  "split":      "test",
  "n_samples":  500,
  "seed":       42,
  "parameters": {
    "temperature": 0.0,
    "max_new_tokens": 256
  },
  "code_hash":  "a3f9c1d"
}
```

`code_hash` should be the short git SHA at time of run. Include it automatically via:

```python
import subprocess
config["code_hash"] = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"]
).decode().strip()
```

---

## `metrics` Block

### Scalars

Single values summarising the run.

```json
"metrics": {
  "scalars": {
    "auroc":           0.847,
    "f1":              0.713,
    "accuracy":        0.821,
    "hallucination_rate": 0.179
  }
}
```

### Curves

Ordered sequences of values over a shared index (steps, epochs, token position, etc.).
Always store both axis arrays — never assume the index is 0-based integers.

```json
"curves": {
  "train_loss": {
    "x_label": "step",
    "y_label": "cross_entropy_loss",
    "x": [0, 100, 200, 300],
    "y": [2.31, 1.87, 1.54, 1.41]
  },
  "token_entropy_by_position": {
    "x_label": "token_position",
    "y_label": "entropy_bits",
    "x": [0, 1, 2, 3, 4],
    "y": [1.2, 1.5, 2.1, 3.4, 3.1]
  }
}
```

### Confusion Matrix

```json
"confusion_matrix": {
  "labels": ["correct", "hallucinated"],
  "matrix": [
    [412, 45],
    [43,  0]
  ],
  "note": "rows=actual, cols=predicted"
}
```

### Statistical Tests

```json
"statistical_tests": [
  {
    "name":        "Mann-Whitney U: entropy at failure vs. baseline",
    "test":        "mann_whitney_u",
    "statistic":   14823.0,
    "p_value":     0.0031,
    "ci_lower":    null,
    "ci_upper":    null,
    "significant": true,
    "alpha":       0.05,
    "note":        "One-sided test, alternative=greater"
  },
  {
    "name":        "95% CI on hallucination rate",
    "test":        "wilson_interval",
    "statistic":   null,
    "p_value":     null,
    "ci_lower":    0.141,
    "ci_upper":    0.221,
    "significant": null,
    "alpha":       0.05,
    "note":        "n=500"
  }
]
```

---

## `sequences` Block

Used for token-level analysis, including event-aligned plotting of hallucinations.
Each entry is one generation (one prompt → one model output).

```json
"sequences": [
  {
    "sequence_id":    "seq_0001",
    "prompt":         "What is 17 * 24?",
    "generated_text": "17 * 24 = 408",

    "tokens": ["17", " *", " 24", " =", " 408"],

    "token_logprobs": [-0.12, -0.03, -0.08, -0.05, -2.41],

    "token_entropy":  [0.21, 0.08, 0.14, 0.11, 3.87],

    "failure_index":  4,

    "label":          "hallucinated",

    "scores": {
      "semantic_similarity": 0.23,
      "factual_consistency": 0.11
    },

    "metadata": {
      "ground_truth": "408",
      "model_answer": "408",
      "annotator":    "auto"
    }
  }
]
```

| Field            | Type      | Notes                                                              |
|------------------|-----------|--------------------------------------------------------------------|
| `sequence_id`    | string    | Unique within the experiment                                       |
| `tokens`         | string[]  | Decoded tokens in generation order                                 |
| `token_logprobs` | float[]   | Log probability of each generated token; same length as `tokens`  |
| `token_entropy`  | float[]   | Entropy of the predictive distribution at each position            |
| `failure_index`  | int\|null | Index into `tokens` of the first hallucination event. `null` if correct. Used as the alignment anchor for event-aligned plots. |
| `label`          | string    | `"correct"` \| `"hallucinated"` \| `"uncertain"`                  |
| `scores`         | object    | Any per-sequence scalar scores                                     |

### Alignment Convention

When plotting token statistics aligned on failure:

- Position `0` = `failure_index` (the failure event)  
- Negative positions = tokens *before* failure (`-1` = one token before, etc.)  
- Positive positions = tokens *after* failure

Sequences without a `failure_index` (i.e. `null`) are excluded from aligned plots
unless explicitly included as a "correct" baseline trace.

---

## `metadata` Block

Freeform but encouraged fields:

```json
"metadata": {
  "researcher":   "Parker",
  "institution":  "Hospital Mathematics Division",
  "data_version": "gsm8k-v1.1",
  "notes":        "First baseline run. Temperature 0 to minimise variance.",
  "duration_seconds": 847,
  "gpu":          "A100-40GB"
}
```

---

## Writing Results in Python

Use this helper so every experiment writes a consistent file:

```python
# results_writer.py
import json, os, subprocess
from datetime import datetime, timezone

def save_results(slug: str, config: dict, metrics: dict,
                 sequences: list = None, metadata: dict = None,
                 results_dir: str = "results") -> str:
    ts = datetime.now(timezone.utc)
    experiment_id = f"{slug}_{ts.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(results_dir, experiment_id)
    os.makedirs(out_dir, exist_ok=True)

    try:
        code_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        code_hash = "unknown"

    config["code_hash"] = code_hash

    result = {
        "schema_version": "1.0",
        "experiment_id":  experiment_id,
        "timestamp":      ts.isoformat(),
        "description":    config.pop("description", ""),
        "tags":           config.pop("tags", []),
        "config":         config,
        "metrics":        metrics,
        "sequences":      sequences or [],
        "metadata":       metadata or {},
    }

    path = os.path.join(out_dir, "result.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved → {path}")
    return experiment_id
```

### Usage

```python
from results_writer import save_results

save_results(
    slug="hallucination_baseline",
    config={
        "description": "Baseline hallucination detection on GSM8K",
        "tags": ["hallucination", "baseline"],
        "model": "meta-llama/Llama-3-8B-Instruct",
        "dataset": "gsm8k",
        "seed": 42,
    },
    metrics={
        "scalars": {"auroc": 0.847, "f1": 0.713},
        "curves": { ... },
        "confusion_matrix": { ... },
        "statistical_tests": [ ... ],
    },
    sequences=sequence_list,
    metadata={"researcher": "Parker", "duration_seconds": 847}
)
```

---

## Validation

Before generating any plot or report, validate the result file:

```python
# validate_result.py
import json

REQUIRED_TOP = {"schema_version", "experiment_id", "timestamp",
                "description", "tags", "config", "metrics"}

def validate(path: str):
    with open(path) as f:
        r = json.load(f)
    missing = REQUIRED_TOP - set(r.keys())
    assert not missing, f"Missing top-level fields: {missing}"
    assert "scalars" in r["metrics"], "metrics.scalars required"
    for seq in r.get("sequences", []):
        if seq.get("token_logprobs"):
            assert len(seq["token_logprobs"]) == len(seq["tokens"]), \
                f"token_logprobs length mismatch in {seq['sequence_id']}"
    print(f"✓ {path} is valid")
```
---

## Research Abstract
We want to build a framework to predict LLM errors at the token level using a controlled synthetic environment with known ground truth. The core scientific question is whether instability in the SVD of the QK^T attention matrix precedes and predicts rule violations in a transformer model, and how far ahead of the violation this signal is detectable.
We use a degree-corrected stochastic block model (DCSBM) to generate a directed graph where vertices represent tokens and directed edges represent one token following another. Walks on this graph are the training corpus and represent valid sequences. The graph has block structure where each block can be thought of as a concept or semantic cluster. We impose hidden rules on a subset of vertices called block jumpers. A block jumper vertex v_i in block b has an associated jump length r, meaning that after exactly r steps from v_i, the walk must land in a specific target block different from b. This rule is never encoded for the transformer - it must be learned implicitly from the corpus. The jump length r is the primary experimental variable and is swept relative to the context window size w. Valid paths from v_i to the target block must exist at length r but must not be the only paths, otherwise the rule is trivially learnable from graph topology alone. The in-group and out-group connectivity density parameters of the DCSBM are therefore essential experimental controls for ensuring the experiment is non-trivial.
The governing parameters and their intended sweep ranges are as follows. n is the number of vertices, swept over 200, 500, 1000, 2000. w is the transformer context window size, swept over 32, 64, 128, 256. The DCSBM block structure parameters include number of blocks swept over 4, 8, 16, in-group edge probability p_in swept over 0.15, 0.25, 0.40, and out-group edge probability p_out swept over 0.01, 0.03, 0.07. t is training corpus size, always kept at minimum two orders of magnitude larger than n with values swept over 50k, 200k, 1M, 5M walks. r is the jump length, swept as multiples of w at 0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w - we expect hallucination rate to increase monotonically with r and to show a step change as r crosses w. l is the walk length included in the training corpus, always set to at least 2w and swept at 2w, 4w, 8w. Number of block jumper vertices per block is swept over 1, 2, 5 as a fraction of block size. Each configuration is run with 3 random seeds. The full sweep is large and should be structured so that the core r vs w interaction experiment runs first with n=500, w=64, t=200k as the baseline configuration, with all other parameters varied around this anchor.
The transformer architecture is NanoGPT scale with d_model swept over 64, 128, 256, n_layers swept over 2, 4, 6, and exactly 1 attention head throughout. The single attention head constraint is intentional and essential - it keeps the QK^T matrix analysis unambiguous and interpretable. Context window w matches the parameter above.
Training sufficiency is a hard gate before any SVD analysis is run. A configuration is considered sufficiently trained only when edge compliance rate exceeds 95 percent and rule compliance rate exceeds 80 percent on held-out walks. If a configuration does not meet this gate after the allocated training budget, it is flagged and excluded from SVD analysis rather than producing noise results.
Evaluation uses three independent layers. First is training signal, which is standard cross-entropy on next token prediction used only to monitor convergence and determine when the sufficiency gate is met, not reported as a result metric. Second is behavioural compliance, where at each step of a generated walk the model output is checked for edge validity meaning the chosen next token corresponds to a valid directed edge in the DCSBM, and rule compliance meaning that at step r from a block jumper vertex the walk lands in the required target block. This produces four outcome classes per step combining edge valid or invalid with rule followed, rule violated, or rule not applicable. The hallucination label in the results schema is this four-class outcome. Third is predictive horizon, which is the core result metric. For each confirmed rule violation event, we look back j steps and ask whether SVD instability metrics were elevated at step t minus j. Sweeping j from 1 to r gives a predictive horizon curve per metric per configuration. The headline result for each metric is the AUROC at each value of j, and the furthest j at which AUROC exceeds 0.75 is the predictive horizon for that metric.
The SVD metrics to be collected at every token step from the QK^T matrix are as follows. Direction of the principal left and right singular vectors tracked as the angle between consecutive steps. Dominant subspace membership tracked as which indices belong to the top-k singular vectors and the set difference between consecutive steps. Principal angles between consecutive dominant subspaces as the canonical Grassmannian distance. Condition number as sigma_1 divided by sigma_n. Spectral gap as sigma_1 minus sigma_2 and generalised gap as sigma_k minus sigma_k+1 for k equal to 2, 4, 8. Singular value entropy computed as negative sum of p_i log p_i where p_i is sigma_i divided by the sum of all singular values, measuring effective rank. Stable rank computed as the squared Frobenius norm divided by the squared spectral norm. Participation ratio computed as the square of the L1 norm of singular values divided by the product of the squared L2 norm and n. Low-rank approximation error at rank k for k equal to 2, 4, 8 computed as the Frobenius norm of QK^T minus its rank-k approximation. Angular velocity of the principal vector as the rate of change of the principal angle between steps. Subspace drift as the Grassmannian distance between dominant subspaces at consecutive steps. Singular value velocity as the per-step change in each of the top-k singular values. Condition number velocity. Alignment of the dominant left singular vector with the token embedding of the current token. Alignment of the dominant right singular vector with the token embedding of the predicted next token. Coherence of the dominant subspace with the full embedding matrix measured as the maximum cosine similarity between any subspace basis vector and any token embedding. Rank of QK^T restricted to the current context window tokens. Variance of singular values across the context window.
All of these metrics are stored as token-level time series in the sequences block of the results JSON per the project schema. Each metric gets its own entry in token_metrics keyed by metric name. The failure_index field marks the confirmed rule violation event and is the alignment anchor for all event-aligned plots.
Results are stored per experimental configuration where each configuration is one combination of all governing parameters plus a random seed. Each configuration gets its own result.json. The comparison report across configurations is the primary deliverable and must support filtering by any subset of parameters and overlaying aligned metric curves across filtered configurations.
The compute budget is 100 USD on RunPod. Use RTX 3090 or RTX 4090 instances. The first run should be a single anchor configuration end-to-end to calibrate wall time before launching the full sweep. Training and SVD collection should be profiled separately. SVD collection at every token step for long sequences is O(d cubed) per step and must be optimised, using torch.linalg.svd with full_matrices=False and batching where possible. The parameter sweep should be structured as a job queue so that the most scientifically critical configurations run first and the budget can be cut at any point without losing the core result.

---

## Project Context

- **Initiative:** AI Health Research — Hospital Mathematics Division
- **Framework:** This project uses [GSD (Get Shit Done)](https://github.com/gsd-build/get-shit-done) for project scaffolding. Read the GSD repo before generating any structure and follow its conventions.
- **Verification Requirement:** All core mathematical logic must be extractable into a peer-review PDF (see below).

---

## Clarification Phase

Before scaffolding anything, conduct a structured clarifying interview. Ask one topic at a time. Do not proceed to GSD invocation until all of the following are resolved with enough specificity:

- [ ] Research question and hypothesis
- [ ] Data sources — type, structure, availability (synthetic vs. clinical vs. sensor)
- [ ] Mathematical / statistical methods expected or preferred
- [ ] Evaluation criteria and success metrics
- [ ] Constraints — privacy, compute budget, reproducibility requirements
- [ ] Required output artifacts (reports, model weights, visualizations, etc.)

Once all are answered, summarize your understanding and ask for confirmation before invoking GSD.

---

## GSD Invocation

After confirmation, invoke GSD to scaffold the project. The generated project must include:

- A `README.md` with the research question, methods summary, and reproduction steps
- A `requirements.txt` (Python 3.11+)
- A `Makefile` with at minimum: `make run`, `make test`, `make pdf`
- Source organized so that math-heavy modules are clearly separated from I/O, config, and orchestration code

---

## Math Verification PDF

After the project is scaffolded and core logic is implemented, produce a PDF for researcher sign-off.

**Process:**

1. Identify all source files containing core mathematical logic. Signals to look for:
   - Numerical methods, matrix operations, statistical models
   - Loss / objective functions, optimization routines
   - Probability distributions, signal processing, transforms
   - Files with math-heavy docstrings or equation comments

2. For each identified file, generate:
   - A 1–2 sentence plain-language summary of what the file does
   - The full code block (monospaced)
   - A LaTeX representation of the mathematics the code implements

   > Use an LLM call (Anthropic API) to generate LaTeX from code — do not use static heuristics. LaTeX accuracy is more important than generation speed.

3. Compile into a single PDF with the following structure:
   - Title page: project name, research question, date
   - Table of contents
   - One section per math-heavy file: summary → code → LaTeX → plain-language description
   - Appendix: list of all other source files (non-math) for completeness

4. The PDF is intended for peer review. Note clearly on the title page that LaTeX was AI-generated and requires researcher sign-off before being treated as ground truth.

**Tooling:** Use `pylatex` or raw LaTeX templating + `pdflatex` for compilation.

---

## Notes for Claude Code

- Read the GSD repo first. Follow its structure and conventions throughout.
- Do not write any code before the clarification phase is complete and confirmed.
- Keep the math extraction and PDF generation as a standalone `make pdf` step — it should be re-runnable at any point after implementation.
- If anything in this prompt is ambiguous, ask before proceeding.
- Always run python commands in a venv

Guiding Principals
## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.