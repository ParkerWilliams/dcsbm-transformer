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
