# Architecture Patterns: v1.1 Integration

**Domain:** DCSBM transformer SVD hallucination prediction -- journal feedback features
**Researched:** 2026-02-26
**Scope:** Integration of 8 v1.1 features into existing v1.0 architecture

## Recommended Architecture

The v1.1 features fall into three integration categories: (A) features that modify existing core modules (multi-head, full spectrum), (B) features that extend existing analysis pipelines (PR curves, calibration, compliance curves, softmax bounds), and (C) features that are largely standalone (null model, overhead benchmarking, pre-registration). The key architectural risk is multi-head support, which requires coordinated changes across 6+ modules. All other features are additive and can be built incrementally without breaking backward compatibility.

### Integration Map: All 8 Features

```
FEATURE                     MODULES TOUCHED          CATEGORY
1. Null model baseline      config, graph, eval      New config flag, reuse pipeline
2. Softmax filtering bound  analysis (NEW module)    New standalone module
3. Multi-head ablation      config, model, eval,     Core module modification
                            analysis, viz, reporting
4. PR curves + calibration  analysis, viz            Extension of existing AUROC
5. Pre-registration         analysis, results        Extension of existing schema
6. Compliance curve         training, analysis (NEW) New sweep + analysis module
7. Full spectrum storage    eval, analysis, results  Extension of existing NPZ
8. SVD overhead benchmark   benchmarks (NEW package) Standalone tool
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `src/config/experiment.py` | Add `n_heads` flexibility, `null_model` flag | All modules |
| `src/model/attention.py` | Multi-head CausalSelfAttention | block.py, transformer.py |
| `src/model/transformer.py` | Pass n_heads through, update get_wvwo() | evaluation pipeline |
| `src/evaluation/pipeline.py` | Per-head SVD extraction, full spectrum storage | analysis modules |
| `src/analysis/auroc_horizon.py` | PR curves, calibration, pre-registration | results schema |
| `src/analysis/softmax_bound.py` | NEW: theoretical bound verification | reporting |
| `src/analysis/compliance_curve.py` | NEW: r/w sweep analysis | reporting, visualization |
| `src/analysis/spectrum.py` | NEW: curvature/torsion on full spectrum | results schema |
| `src/benchmarks/` | NEW package: SVD timing | standalone |

### Data Flow Changes from v1.0

```
v1.0 flow (unchanged for single-head):
  Config -> Graph -> Walks -> Train -> Eval+SVD -> Analysis -> Report

v1.1 additions:

  [Null model path]
  Config(null_model=True) -> Graph(n_jumpers_per_block=0) -> Walks -> Train ->
    Eval+SVD -> Baseline distributions (stored in result.json)

  [Multi-head path]
  Config(n_heads=2|4) -> Model(MultiHeadAttention) -> Train ->
    Eval(per-head QKT extraction) -> SVD(per-head) -> Analysis(signal concentration)

  [Full spectrum path]
  Eval -> token_metrics.npz now includes sigma_vectors per step ->
    Analysis(curvature, torsion) -> result.json

  [Compliance curve path]
  r_sweep configs -> multiple training runs -> compliance_curve analysis ->
    sharp transition plot

  [Benchmark path]
  Model + synthetic inputs -> timed SVD -> overhead_report.json (standalone)
```

## Feature 1: Null Model Baseline

### Where it fits

The null model is NOT a new model type. It is a standard experiment run with `n_jumpers_per_block=0`, which eliminates all block jumper rules. The model trains on clean sequences where no rule violations are possible. The resulting Grassmannian drift distribution serves as the null hypothesis: "this is how much subspace rotation happens when there is nothing to predict."

### Architecture decision: Config flag vs. new walk type

**Use a config flag, not a new walk type.** The existing pipeline already handles the case where `n_jumpers_per_block=0` -- the jumper designation returns an empty list, walks are pure random walks, and behavioral classification labels everything as NOT_APPLICABLE. The only change needed is:

1. **`src/config/experiment.py`**: Remove or relax the `n_jumpers_per_block` minimum if one exists. Add a `tags` value convention: configs with `tags=("null_model",)` are recognized as baseline runs.

2. **`src/graph/jumpers.py`**: Already handles `n_jumpers_per_block=0` gracefully (the loop runs zero times, returns empty list). No change needed.

3. **`src/evaluation/pipeline.py`**: Already handles empty jumper maps. SVD metrics are still collected. Behavioral classification produces all NOT_APPLICABLE outcomes. No change needed.

4. **`src/analysis/null_baseline.py`** (NEW): Loads token_metrics.npz from null model runs, extracts Grassmannian distance distributions, computes summary statistics (mean, std, percentiles), and provides comparison functions for real experiment results.

### Modifications required

| File | Change | Breaking? |
|------|--------|-----------|
| `src/config/experiment.py` | None (already supports n_jumpers_per_block=0 if > 0 check doesn't exist) | No |
| `src/analysis/null_baseline.py` | NEW file: extract null distributions | No |
| `src/results/schema.py` | Add `null_baseline` section to result.json | No (additive) |

### Data contract

```python
# null_baseline section in result.json
"null_baseline": {
    "source_experiment_id": "null_n500_w64_seed42",
    "grassmannian_distance": {
        "qkt": {"mean": 0.12, "std": 0.04, "p95": 0.19, "p99": 0.23},
        "avwo": {"mean": 0.08, "std": 0.03, "p95": 0.13, "p99": 0.16}
    },
    "per_layer": {
        "layer_0": {...},
        "layer_1": {...},
        ...
    }
}
```

## Feature 2: Softmax Filtering Bound

### Where it fits

This is a mathematical analysis feature that verifies a theoretical bound: given perturbation epsilon in QKT, how much does the output AVWo change after softmax filtering? It has TWO parts: (a) LaTeX derivation in the math PDF, and (b) empirical verification using collected SVD data.

### Architecture decision: New analysis module

**Create `src/analysis/softmax_bound.py`.** This is cleanly separate from existing modules because it asks a different question (perturbation propagation) than existing analysis (predictive discrimination).

### Modifications required

| File | Change | Breaking? |
|------|--------|-----------|
| `src/analysis/softmax_bound.py` | NEW: epsilon-bound computation and empirical verification | No |
| `src/reporting/math_pdf.py` | Add MATH_SECTIONS entry for the softmax bound derivation | No (additive) |
| `src/evaluation/pipeline.py` | May need to store paired (QKT_perturbed, AVWo) at same step | Possibly -- see below |

### Key concern: Data collection

The empirical verification needs paired measurements: "at step t, the QKT perturbation was X and the AVWo perturbation was Y." The existing pipeline already stores both QKT and AVWo Grassmannian distances at each step, keyed by `qkt.layer_N.grassmannian_distance` and `avwo.layer_N.grassmannian_distance`. These are already aligned by step index in token_metrics.npz.

Therefore: **No new data collection is needed.** The softmax bound verification can work entirely from existing NPZ data by correlating the QKT and AVWo perturbation magnitudes at each step and checking whether the empirical ratio stays within the theoretical bound.

```python
# In src/analysis/softmax_bound.py
def verify_softmax_bound(
    qkt_grassmannian: np.ndarray,  # [n_sequences, n_steps]
    avwo_grassmannian: np.ndarray, # [n_sequences, n_steps]
    d_model: int,
    theoretical_bound_fn: Callable,
) -> dict:
    """Check empirical AVWo perturbation <= bound(QKT perturbation)."""
```

## Feature 3: Multi-Head Ablation (1h/2h/4h)

### Where it fits

This is the most invasive feature. The current architecture **enforces** `n_heads=1` via a hard check in `ExperimentConfig.__post_init__`. The single-head assumption is baked into:

1. `CausalSelfAttention` -- no head dimension, Q/K/V are each [B, T, D]
2. `ForwardOutput` -- QKT is [B, n_layers, T, T] (no head dimension)
3. `fused_evaluate` -- indexes QKT as [B, layer, T, T]
4. SVD metrics -- compute on [T, T] matrices
5. Grassmannian distance tracking -- keyed by (target, layer)
6. NPZ keys -- `qkt.layer_0.grassmannian_distance`
7. AUROC analysis -- reads NPZ keys directly
8. All visualization -- expects single QKT per layer

### Architecture decision: Head-aware attention with backward-compatible ForwardOutput

**Approach:** Modify `CausalSelfAttention` to support multiple heads. Add a head dimension to ForwardOutput. Make the evaluation pipeline iterate over heads. Keep the NPZ key convention backward-compatible by including head index.

**Critical constraint:** The d_model must be divisible by n_heads. For anchor config d_model=128: 2 heads = d_head=64, 4 heads = d_head=32. Both are valid.

### Detailed changes

#### `src/config/experiment.py`
```python
# REMOVE the n_heads != 1 check in __post_init__
# REPLACE with:
if self.model.n_heads not in (1, 2, 4):
    raise ValueError("n_heads must be 1, 2, or 4")
if self.model.d_model % self.model.n_heads != 0:
    raise ValueError(f"d_model ({self.model.d_model}) must be divisible by n_heads ({self.model.n_heads})")
```

#### `src/model/attention.py`
The entire class needs rewriting to support multi-head. Key change: Q, K, V go from [B, T, D] to [B, n_heads, T, d_head]. The output shape stays [B, T, D] (heads are concatenated and projected).

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        # W_q, W_k, W_v, W_o dimensions unchanged (d_model -> d_model)
        # But internally we reshape to [B, n_heads, T, d_head]

    def forward(self, x, extract=False):
        # Q, K, V: [B, T, D] -> [B, n_heads, T, d_head]
        # QKT per head: [B, n_heads, T, T]
        # Attention per head: [B, n_heads, T, T]
        # Values per head: [B, n_heads, T, d_head]
        # Output: concat heads -> [B, T, D] -> W_o -> [B, T, D]
        if extract:
            return y, AttentionInternals(
                qkt=qkt_target,        # NOW [B, n_heads, T, T]
                attention_weights=attn, # NOW [B, n_heads, T, T]
                values=v,               # NOW [B, n_heads, T, d_head]
            )
```

#### `src/model/types.py`
```python
@dataclass
class AttentionInternals:
    qkt: torch.Tensor           # [B, n_heads, T, T] (was [B, T, T])
    attention_weights: torch.Tensor  # [B, n_heads, T, T] (was [B, T, T])
    values: torch.Tensor        # [B, n_heads, T, d_head] (was [B, T, D])

@dataclass
class ForwardOutput:
    logits: torch.Tensor
    qkt: torch.Tensor | None = None  # [B, n_layers, n_heads, T, T] (was [B, n_layers, T, T])
    attention_weights: torch.Tensor | None = None  # [B, n_layers, n_heads, T, T]
    values: torch.Tensor | None = None  # [B, n_layers, n_heads, T, d_head]
    ...
```

#### `src/model/transformer.py`
```python
# In TransformerLM.__init__:
# Pass n_heads to TransformerBlock -> CausalSelfAttention
self.blocks = nn.ModuleList([
    TransformerBlock(d_model, n_heads, max_seq_len, dropout)
    for _ in range(n_layers)
])

# In forward():
# Stack now includes head dimension:
qkt = torch.stack(all_qkt, dim=1)  # [B, n_layers, n_heads, T, T]

# get_wvwo() returns [n_layers, n_heads, d_head, d_head] for multi-head
# or [n_layers, d_model, d_model] for single-head
```

#### `src/evaluation/pipeline.py`
This is the most complex change. The inner loop must iterate over heads:

```python
for layer_idx in range(n_layers):
    for head_idx in range(n_heads):
        qkt_matrix = output.qkt[:, layer_idx, head_idx]  # [B, T, T]
        # SVD on per-head QKT...
        key = f"qkt.layer_{layer_idx}.head_{head_idx}.{metric_name}"
```

For single-head (n_heads=1), this produces keys like `qkt.layer_0.head_0.grassmannian_distance`, which is backward-incompatible with v1.0 keys like `qkt.layer_0.grassmannian_distance`.

**Backward compatibility strategy:** When n_heads=1, emit BOTH key formats:
- `qkt.layer_0.grassmannian_distance` (v1.0 compatible)
- `qkt.layer_0.head_0.grassmannian_distance` (v1.1 format)

This ensures v1.0 analysis code continues to work on single-head results while multi-head results use the new key format.

#### `src/analysis/signal_concentration.py` (NEW)
Per-head signal concentration analysis: which heads carry the predictive signal?

```python
def compute_signal_concentration(
    per_head_aurocs: dict[int, dict[str, np.ndarray]],  # head_idx -> metric -> auroc_curve
) -> dict:
    """Determine which heads concentrate the predictive SVD signal."""
    # Compare max AUROC across heads for each metric
    # Report: "Head 0 carries 80% of Grassmannian signal, Head 1 carries 20%"
```

#### Impact on downstream modules

| Module | Change needed |
|--------|--------------|
| `src/analysis/auroc_horizon.py` | Parse head index from NPZ keys; aggregate or per-head AUROC |
| `src/analysis/statistical_controls.py` | Handle per-head metrics in correlation matrices |
| `src/analysis/event_extraction.py` | No change (works on behavioral labels, not SVD) |
| `src/visualization/auroc.py` | Per-head AUROC plot overlays |
| `src/visualization/event_aligned.py` | Per-head metric trajectories |
| `src/reporting/single.py` | Multi-head section in report |
| `src/reporting/math_pdf.py` | Update attention.py section for multi-head math |

### Multi-head WvWo changes

For multi-head, `get_wvwo()` must return per-head OV circuits. With multi-head attention:
- W_v projects [D] -> [D], then reshape to [n_heads, d_head]
- W_o projects from concatenated heads [D] -> [D]

The per-head OV circuit is: `W_v_head_h @ W_o_slice_h`, where W_o_slice_h is the slice of W_o that maps head h's output back to the residual stream. This gives [d_head, d_model] per head, not a square matrix.

**Decision:** For multi-head WvWo, compute the full OV product per head as `W_v[:, h*d_head:(h+1)*d_head].T @ W_o[h*d_head:(h+1)*d_head, :]`, yielding [d_head, d_model]. SVD metrics that require square matrices (like read_write_alignment) must handle rectangular input or be skipped for WvWo in multi-head mode.

## Feature 4: PR Curves + Calibration

### Where it fits

These plug directly into the existing AUROC pipeline. PR curves and calibration use the SAME violation/control event extraction and the SAME metric values at each lookback distance. They just compute different statistics.

### Architecture decision: Extend auroc_horizon.py

**Add functions to `src/analysis/auroc_horizon.py`**, not a new module. PR curves and calibration are conceptually part of the same predictive evaluation as AUROC.

### Modifications required

| File | Change | Breaking? |
|------|--------|-----------|
| `src/analysis/auroc_horizon.py` | Add `compute_pr_curve()`, `compute_calibration()` | No |
| `src/analysis/statistical_controls.py` | Add PR-AUC and calibration to enriched results | No |
| `src/visualization/` | New plot types: pr_curve.py, calibration.py | No (additive) |
| `src/results/schema.py` | Add pr_auc and calibration sections | No (additive) |

### Data contract

```python
# Added to by_metric in result.json predictive_horizon
"pr_curve": {
    "precision_by_lookback": [[...], ...],   # [r_value][n_thresholds]
    "recall_by_lookback": [[...], ...],
    "pr_auc_by_lookback": [0.72, 0.68, ...], # one PR-AUC per lookback j
    "max_pr_auc": 0.72,
    "max_pr_auc_lookback": 1
},
"calibration": {
    "bin_edges": [0.0, 0.1, 0.2, ...],
    "bin_fractions": [0.05, 0.12, ...],     # actual positive rate per bin
    "bin_mean_predicted": [0.05, 0.15, ...],
    "expected_calibration_error": 0.034
}
```

### Implementation notes

PR curves need a continuous score, not just binary groups. The existing pipeline provides metric values (Grassmannian distance, etc.) at each lookback -- these ARE the continuous scores. Higher Grassmannian distance = higher predicted probability of violation. The PR curve computation:

```python
def compute_pr_curve(
    violation_events: list[AnalysisEvent],
    control_events: list[AnalysisEvent],
    metric_array: np.ndarray,
    r_value: int,
) -> dict:
    """Compute precision-recall curve at each lookback j."""
    # For each lookback j:
    #   scores = metric values at resolution_step - j
    #   labels = 1 for violation, 0 for control
    #   PR curve from sklearn.metrics.precision_recall_curve
```

Calibration requires binning predicted probabilities. Since raw SVD metrics are not probabilities, use isotonic regression or Platt scaling to convert metric values to calibrated probabilities, then compute reliability diagrams.

## Feature 5: Pre-Registration Framework

### Where it fits

Pre-registration is a methodological framework, not a software feature. It manifests as:
1. A frozen hypothesis specification (already exists: PRIMARY_METRICS in auroc_horizon.py)
2. A held-out evaluation protocol
3. Documentation of what was decided before seeing results

### Architecture decision: Extend config + results schema

**Minimal code changes.** The pre-registration framework is mostly about discipline, not architecture. The code changes are:

| File | Change | Breaking? |
|------|--------|-----------|
| `src/config/experiment.py` | Add `held_out_fraction: float = 0.2` to TrainingConfig | No |
| `src/evaluation/pipeline.py` | Split eval walks into exploratory/confirmatory sets | No |
| `src/analysis/auroc_horizon.py` | Add `dataset_split` parameter ("exploratory" or "confirmatory") | No |
| `src/results/schema.py` | Add `pre_registration` section with hypothesis, split info | No |

### Data contract

```python
"pre_registration": {
    "primary_hypothesis": "Grassmannian distance of QKT predicts rule violations",
    "primary_metric": "qkt.grassmannian_distance",
    "secondary_metrics": ["qkt.spectral_gap_1_2", "qkt.spectral_entropy", ...],
    "held_out_fraction": 0.2,
    "exploratory_n": 800,
    "confirmatory_n": 200,
    "exploratory_results": {...},  # full analysis on 80% of eval data
    "confirmatory_results": {...}  # blinded analysis on 20% held-out
}
```

## Feature 6: Sharp Compliance Curve

### Where it fits

The compliance curve shows how rule compliance degrades as r/w increases. This requires running multiple training experiments with different r values and plotting the resulting compliance rates. It is NOT an extension of the sufficiency gate -- it is a sweep analysis.

### Architecture decision: New analysis module + config sweep utility

The compliance curve needs data from multiple experiments, each with a different r value. The existing `compute_r_values()` in jumpers.py already defines the discrete r set. The compliance curve analysis reads completed result.json files and extracts compliance rates.

| File | Change | Breaking? |
|------|--------|-----------|
| `src/analysis/compliance_curve.py` | NEW: load multiple result.json, extract compliance, plot transition | No |
| `src/visualization/compliance.py` | NEW: sharp transition plot | No |

### Key insight: Reuse existing architecture

The compliance curve does NOT require any new training infrastructure. Each r value is already a separate experiment config. The sweep runs them independently. The compliance curve analysis is a post-hoc aggregation:

```python
def compute_compliance_curve(
    result_dirs: list[Path],
) -> dict:
    """Extract r/w ratio vs compliance from multiple experiments."""
    points = []
    for result_dir in result_dirs:
        result = load_result(result_dir / "result.json")
        r = result["config"]["training"]["r"]
        w = result["config"]["training"]["w"]
        edge = result["metrics"]["scalars"]["final_edge_compliance"]
        rule = result["metrics"]["scalars"]["final_rule_compliance"]
        points.append({"r_over_w": r / w, "edge": edge, "rule": rule})
    return {"compliance_curve": sorted(points, key=lambda p: p["r_over_w"])}
```

The r values from jumpers.py ({0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w}) already provide the x-axis points. Each is a separate training run with its own result.json.

## Feature 7: Full Spectrum Trajectory

### Where it fits

Currently, the evaluation pipeline computes SVD and immediately reduces singular values to scalar metrics (stable_rank, spectral_entropy, etc.). The full spectrum feature stores the raw singular value vector sigma_1...sigma_k at each step, enabling post-hoc analysis of spectral curve shape changes.

### Architecture decision: Extend NPZ storage, new analysis module

**Store full spectrum in token_metrics.npz alongside scalar metrics.** This is additive -- existing scalar metric keys are unchanged.

| File | Change | Breaking? |
|------|--------|-----------|
| `src/evaluation/pipeline.py` | Store `qkt.layer_N.spectrum` arrays in NPZ | No (additive) |
| `src/analysis/spectrum.py` | NEW: curvature, torsion, spectral curve analysis | No |
| `src/results/schema.py` | Add spectrum trajectory section | No (additive) |

### Storage format

```python
# In token_metrics.npz, add per-layer per-target spectrum arrays:
# Key: "qkt.layer_0.spectrum"
# Shape: [n_sequences, n_steps, k] where k = min(T, D) singular values
# Key: "avwo.layer_0.spectrum"
# Shape: [n_sequences, n_steps, k]
```

### Storage overhead estimate

For anchor config: n_sequences=1000, n_steps=256, k=min(64,128)=64:
- Per target per layer: 1000 * 256 * 64 * 4 bytes = ~62 MB
- 3 targets * 4 layers = 12 arrays = ~750 MB total

This is large. **Mitigation:** Only store spectrum for QKT (the primary target), only for layers that show signal (configurable, default all layers), and use float16 for storage:
- QKT only, all 4 layers: ~250 MB (float16: ~125 MB)
- Acceptable for research runs on RTX 3090/4090 with 24GB VRAM

### Curvature and torsion computation

```python
# In src/analysis/spectrum.py
def spectral_curvature(spectra: np.ndarray) -> np.ndarray:
    """Curvature of the spectral curve sigma(t).

    spectra: [n_steps, k] -- singular values at each step
    Returns: [n_steps] -- curvature at each step (finite difference)
    """
    # First derivative: d_sigma/dt (velocity of spectral change)
    d1 = np.diff(spectra, axis=0)  # [n_steps-1, k]
    # Second derivative: d^2_sigma/dt^2 (acceleration)
    d2 = np.diff(d1, axis=0)  # [n_steps-2, k]
    # Curvature = ||d2|| / ||d1||^3 (per-step scalar)
    d1_norm = np.linalg.norm(d1[:-1], axis=-1)
    d2_norm = np.linalg.norm(d2, axis=-1)
    return d2_norm / (d1_norm**3 + 1e-12)
```

## Feature 8: SVD Computational Overhead

### Where it fits

This is a standalone benchmarking tool that measures how much time SVD adds to the evaluation pipeline. It does NOT need to be integrated into the production evaluation code.

### Architecture decision: Standalone benchmarks package

**Create `src/benchmarks/svd_overhead.py`.** This is a measurement tool, not a pipeline component. It runs synthetic inputs through the SVD computation path and reports timing.

| File | Change | Breaking? |
|------|--------|-----------|
| `src/benchmarks/__init__.py` | NEW package | No |
| `src/benchmarks/svd_overhead.py` | NEW: timing benchmarks | No |

### What it measures

1. **Per-step SVD time** for different matrix sizes (32x32, 64x64, 128x128, 256x256)
2. **Per-step metric computation time** (all 8 scalar metrics from a pre-computed SVD)
3. **Total evaluation overhead** as fraction of forward pass time
4. **Cheaper approximation candidates**: `torch.svd_lowrank(q=8)` vs `torch.linalg.svd(full_matrices=False)`

### Output format

```python
# overhead_report.json
{
    "device": "cuda:0 (RTX 3090)",
    "matrix_sizes": {
        "64x64": {
            "svd_ms": 0.12,
            "metrics_ms": 0.03,
            "total_ms": 0.15,
            "forward_pass_ms": 1.2,
            "overhead_fraction": 0.125
        },
        "128x128": {...}
    },
    "approximations": {
        "svd_lowrank_q8": {
            "time_ms": 0.05,
            "speedup": 2.4,
            "metric_error": {
                "stable_rank": 0.001,
                "grassmannian_distance": 0.003,
                ...
            }
        }
    }
}
```

## Backward Compatibility with v1.0 result.json

### Schema version

Bump `schema_version` from `"1.0"` to `"1.1"`. The v1.1 schema is a strict superset of v1.0:
- All v1.0 fields remain unchanged
- New sections are additive (null_baseline, pr_curve, calibration, pre_registration, spectrum_trajectory, overhead)
- v1.0 readers that check for required fields will not break (no required fields removed)

### NPZ key compatibility

| v1.0 Key Format | v1.1 Key Format (single-head) | v1.1 Key Format (multi-head) |
|-----------------|------------------------------|------------------------------|
| `qkt.layer_0.grassmannian_distance` | Same (preserved) | `qkt.layer_0.head_0.grassmannian_distance` |
| `avwo.layer_0.stable_rank` | Same (preserved) | `avwo.layer_0.head_0.stable_rank` |
| N/A | `qkt.layer_0.spectrum` | `qkt.layer_0.head_0.spectrum` |

For single-head runs, both v1.0 and v1.1 key formats are emitted. For multi-head runs, only v1.1 format is used (there is no sensible v1.0 equivalent for per-head data).

### result.json compatibility

```python
# validate_result() in schema.py
def validate_result(result: dict) -> list[str]:
    version = result.get("schema_version", "1.0")
    if version == "1.0":
        # v1.0 validation (unchanged)
        ...
    elif version == "1.1":
        # v1.0 validation + v1.1 additions
        ...
```

## Build Order Considering Feature Dependencies

### Dependency graph

```
Level 0 (no v1.1 dependencies):
  [8] SVD overhead benchmarks   -- standalone, can build anytime
  [2] Softmax filtering bound   -- only needs existing NPZ data
  [5] Pre-registration framework -- config + schema only

Level 1 (depends on Level 0 for data):
  [1] Null model baseline       -- needs config relaxation, new analysis module
  [4] PR curves + calibration   -- extends existing AUROC pipeline
  [7] Full spectrum storage     -- extends evaluation pipeline

Level 2 (depends on Level 1):
  [6] Compliance curve          -- needs multiple experiment results (from null + real runs)

Level 3 (depends on all others for validation):
  [3] Multi-head ablation       -- most invasive, touches everything, should be last
```

### Recommended build sequence

1. **SVD overhead benchmarks** -- standalone, no risk, provides data for the paper immediately
2. **Pre-registration framework** -- minimal code, mostly config + schema, sets up methodology
3. **Softmax filtering bound** -- new analysis module, uses existing data, adds math PDF section
4. **Null model baseline** -- small config change + new analysis module, validates signal
5. **PR curves + calibration** -- extends AUROC pipeline, uses same event extraction
6. **Full spectrum storage** -- extends eval pipeline, new analysis module
7. **Compliance curve** -- post-hoc analysis of multiple runs, needs data from steps 1-6
8. **Multi-head ablation** -- LAST because it modifies core types; all other features should be stable before this invasive change

### Rationale for this order

- **Multi-head last:** Every other feature can be built and tested against the single-head architecture. Once working, multi-head changes propagate through the entire stack. If multi-head is built first, every subsequent feature must be developed against both single-head and multi-head codepaths simultaneously, doubling testing complexity.

- **Null model early:** Provides the baseline distribution that other analyses reference. Without it, the softmax bound verification and PR curves lack context.

- **Overhead benchmarks first:** Zero risk, immediate value, and the results inform whether full spectrum storage is feasible within the compute budget.

## New vs. Modified Components Summary

### New files (7)

| File | Purpose | Depends on |
|------|---------|------------|
| `src/analysis/null_baseline.py` | Extract/compare null distributions | token_metrics.npz |
| `src/analysis/softmax_bound.py` | Theoretical bound verification | token_metrics.npz |
| `src/analysis/spectrum.py` | Full spectrum curvature/torsion | token_metrics.npz (spectrum keys) |
| `src/analysis/compliance_curve.py` | r/w sweep compliance analysis | Multiple result.json files |
| `src/analysis/signal_concentration.py` | Per-head signal analysis | Multi-head AUROC results |
| `src/benchmarks/__init__.py` | Package marker | None |
| `src/benchmarks/svd_overhead.py` | SVD timing benchmarks | Model, torch.linalg.svd |

### Modified files (10)

| File | Change scope | Risk |
|------|-------------|------|
| `src/config/experiment.py` | Remove n_heads=1 constraint, add n_heads validation | MEDIUM |
| `src/model/attention.py` | Multi-head reshape, per-head extraction | HIGH |
| `src/model/block.py` | Pass n_heads through | LOW |
| `src/model/transformer.py` | Pass n_heads, update get_wvwo(), update stacking | MEDIUM |
| `src/model/types.py` | Add head dimension to AttentionInternals/ForwardOutput | MEDIUM |
| `src/evaluation/pipeline.py` | Per-head SVD, full spectrum storage, dual key format | HIGH |
| `src/analysis/auroc_horizon.py` | PR curves, calibration, head-aware key parsing | MEDIUM |
| `src/analysis/statistical_controls.py` | Handle per-head metrics | MEDIUM |
| `src/results/schema.py` | Schema v1.1 additions, validate_result v1.1 branch | LOW |
| `src/reporting/math_pdf.py` | Softmax bound derivation section | LOW |

### Unchanged files (many)

The following modules need NO changes for v1.1:
- `src/graph/` (entire package -- graph generation is orthogonal)
- `src/walk/` (entire package -- walk generation is orthogonal)
- `src/training/trainer.py` (training loop unchanged)
- `src/training/data.py` (data loading unchanged)
- `src/training/checkpoint.py` (checkpoint format unchanged)
- `src/analysis/event_extraction.py` (behavioral events, not SVD)
- `src/reproducibility/` (entire package)
- `src/config/hashing.py`, `src/config/serialization.py`, `src/config/defaults.py`

## Patterns to Follow

### Pattern 1: Feature Flags via Config Tags

**What:** Use `ExperimentConfig.tags` to mark experiments as null models, ablation runs, etc. rather than adding boolean flags to config classes.

**When:** For metadata that does not change the computation (it only changes how results are interpreted).

**Example:**
```python
null_config = ExperimentConfig(
    graph=GraphConfig(n_jumpers_per_block=0),
    tags=("null_model", "v1.1"),
)
```

**Why:** Tags are already in the schema and result.json. Adding boolean flags to frozen dataclasses requires schema migration. Tags are flexible and backward-compatible.

### Pattern 2: Dual Key Emission for Backward Compatibility

**What:** When a key format changes (e.g., adding head index), emit both old and new key formats for the default case (single head).

**When:** Any time an NPZ or result.json key format changes in a way that would break downstream readers.

**Example:**
```python
# In evaluation/pipeline.py
if n_heads == 1:
    # Emit v1.0 compatible key
    svd_metric_arrays[f"qkt.layer_{layer_idx}.{metric_name}"] = ...
# Always emit v1.1 key
svd_metric_arrays[f"qkt.layer_{layer_idx}.head_{head_idx}.{metric_name}"] = ...
```

### Pattern 3: Post-Hoc Aggregation Over In-Pipeline Computation

**What:** For features that need data from multiple experiments (compliance curve, null model comparison), do NOT try to combine results during the pipeline. Write individual result.json files, then aggregate in a separate analysis step.

**When:** Any analysis that crosses experiment boundaries.

**Why:** Keeps each pipeline run independent and reproducible. Aggregation scripts are simpler to debug and re-run.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Multi-Head Refactor

**What:** Changing all modules to support multi-head simultaneously in one large PR.

**Why bad:** If attention.py, types.py, transformer.py, pipeline.py, and analysis all change together, a bug anywhere breaks everything. Testing is impossible because there are no stable intermediate states.

**Instead:** Build multi-head in stages: (1) attention.py + types.py + tests, (2) transformer.py + block.py + tests, (3) pipeline.py + tests, (4) analysis + tests. Each stage has a passing test suite before the next begins.

### Anti-Pattern 2: Storing Full Spectrum Unconditionally

**What:** Always storing sigma_1...sigma_k at every step for every target and layer.

**Why bad:** 750 MB per experiment in float32. With 8+ experiment configs, this exceeds storage budget and slows NPZ I/O.

**Instead:** Make spectrum storage configurable via a flag in ExperimentConfig or ExtractionMode. Default: store only for QKT target. Use float16 for storage.

### Anti-Pattern 3: Computing Calibration During Evaluation

**What:** Running isotonic regression or Platt scaling inside the fused evaluation loop.

**Why bad:** Calibration requires the full dataset of scores and labels. It cannot be computed incrementally during generation.

**Instead:** Calibration is a post-hoc analysis step. Collect raw metric values during evaluation, then fit calibration models in the analysis phase.

## Sources

- Existing codebase analysis (HIGH confidence -- direct code reading)
- `src/config/experiment.py` lines 85-86: `n_heads != 1` validation
- `src/model/attention.py`: single-head architecture with explicit Q/K/V
- `src/model/types.py`: ForwardOutput shape documentation
- `src/evaluation/pipeline.py`: SVD collection loop structure, NPZ key format
- `src/analysis/auroc_horizon.py`: AUROC computation and PRIMARY_METRICS
- `src/results/schema.py`: result.json structure and validation
- `.planning/v1.0-MILESTONE-AUDIT.md`: known integration gaps (NPZ key format conflict)
- PyTorch `torch.linalg.svd`: supports batched input with head dimension (HIGH confidence)
- sklearn.metrics.precision_recall_curve: standard PR curve computation (HIGH confidence)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Null model integration | HIGH | Minimal changes, existing pipeline handles n_jumpers=0 |
| Softmax bound integration | HIGH | Pure analysis module, no pipeline changes |
| Multi-head integration | MEDIUM | Invasive changes, many modules affected, shape mismatches possible |
| PR curves + calibration | HIGH | Well-understood extensions of existing AUROC pipeline |
| Pre-registration | HIGH | Minimal code, mostly methodology |
| Compliance curve | HIGH | Post-hoc aggregation, no pipeline changes |
| Full spectrum storage | MEDIUM | Storage overhead needs empirical validation; float16 precision may affect metrics |
| SVD overhead benchmarks | HIGH | Standalone tool, no integration risk |
| Build order | HIGH | Dependency chain is clear; multi-head last is critical |
| Backward compatibility | MEDIUM | Dual key emission strategy is sound but adds code complexity |
