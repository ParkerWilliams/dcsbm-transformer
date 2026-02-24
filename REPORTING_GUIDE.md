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
