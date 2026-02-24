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
