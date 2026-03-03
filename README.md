# DCSBM Transformer

Predicting transformer rule violations from SVD instability in attention matrices, using synthetic token sequences from degree-corrected stochastic block models.

## Research Question

Can SVD instability metrics extracted from the $QK^\top$ attention matrix predict when a transformer will violate a learned structural rule — before the violation occurs?

We train a transformer on next-token prediction over random walks on a DCSBM graph, where designated "jumper" vertices impose reachability rules. During evaluation, we track singular value decomposition metrics at every generation step and measure whether these metrics diverge before rule violations, quantified as a **predictive horizon** (AUROC > 0.75 at lookback distance $j$).

## Background

### Degree-Corrected Stochastic Block Model

The graph is a directed DCSBM (Karrer & Newman, 2011) with $K$ blocks of equal size. Edge probabilities are:

$$P_{ij} = \theta_i \, \theta_j \, \omega_{b_i, b_j}$$

where $\omega_{ab} = p_\text{in}$ if $a = b$, $\omega_{ab} = p_\text{out}$ otherwise, and $\theta_i$ are degree-correction parameters sampled from a Zipf distribution ($\alpha = 1.0$), normalized per block.

### Jumper Rules

Each block contains designated **jumper** vertices. A jumper $v$ in block $b_s$ carries a rule: *"if visited at step $t$, the walk must be in target block $b_t$ at step $t + r_v$."* The jump length $r_v$ is drawn from a fixed set of scale factors applied to the context window:

$$r \in \left\{ \lfloor s \cdot w \rceil \;\middle|\; s \in \{0.5,\; 0.7,\; 0.9,\; 1.0,\; 1.1,\; 1.3,\; 1.5,\; 2.0\} \right\}$$

Jumper assignments are validated for **non-triviality**: the target block must be reachable in $r$ steps, and at least one non-target block must also be reachable (so compliance is not guaranteed by graph structure alone).

### Walk Generation with Path Splicing

Training walks are generated with **guaranteed compliance** via pre-computed viable paths. For each jumper $v$ with rule length $r_v$ and target block $b_t$, we pre-compute a pool of 200 random $r_v$-step walks from $v$ that end in $b_t$. During walk generation, encountering $v$ triggers a random splice from this pool, replacing probabilistic guided stepping with deterministic insertion.

## Pipeline

```
run_experiment.py --config config.json
```

| Stage | Description |
|-------|-------------|
| 1. Seed | Deterministic seeding (torch + numpy + python) |
| 2. Graph | DCSBM generation with connectivity validation and retry |
| 3. Walks | Two-phase corpus: jumper-seeded guided + random-start batch |
| 4. Model | NanoGPT-scale transformer ($d_\text{model}$=128, 4 layers) |
| 5. Training | Next-token prediction with sufficiency gate (edge >= 0.95, rule >= 0.80) |
| 6. Evaluation | Autoregressive generation with fused SVD extraction + behavioral labeling |
| 7. Analysis | AUROC predictive horizon, bootstrap CIs, Holm-Bonferroni correction |
| 8. Visualization | Horizon curves, spectrum trajectories, compliance plots |
| 9. Report | Self-contained HTML with embedded figures |

## SVD Metrics

Three matrix targets are decomposed at every generation step $t \geq w$:

| Target | Matrix | Description |
|--------|--------|-------------|
| `qkt` | $QK^\top / \sqrt{d_k}$ | Attention scores (zero-filled causal mask) |
| `avwo` | $A \cdot V \cdot W_o$ | Net attention output (per-head OV circuit applied) |
| `wvwo` | $W_v \cdot W_o$ | Static OV circuit weights (computed once) |

### Computed Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Stable rank | $\lVert M \rVert_F^2 / \lVert M \rVert_2^2 = \sum s_i^2 / s_1^2$ | Effective dimensionality |
| Spectral entropy | $-\sum p_i \log p_i$ where $p_i = s_i / \sum s_j$ | Singular value concentration |
| Spectral gap | $s_k - s_{k+1}$ for $k \in \{1, 2, 4\}$ | Separation between components |
| Condition number | $s_1 / s_n$ | Matrix sensitivity |
| Grassmannian distance | Geodesic on $\mathrm{Gr}(k, n)$ between consecutive top-$k$ subspaces | Subspace rotation rate |
| Rank-1 residual | $\lVert M - s_1 u_1 v_1^\top \rVert_F / \lVert M \rVert_F$ | Energy beyond top component |
| Read-write alignment | $\lvert \cos \angle(u_1, v_1) \rvert$ | OV circuit directional coupling |

### Primary Metrics (Pre-registered, Holm-Bonferroni Corrected)

- `qkt.grassmannian_distance`
- `qkt.spectral_gap_1_2`
- `qkt.spectral_entropy`
- `avwo.stable_rank`
- `avwo.grassmannian_distance`

## Predictive Horizon

For each jump length $r$ and lookback distance $j \in \{1, \ldots, r\}$:

$$\mathrm{AUROC}(j) = P\!\left(X_\text{violated}^{(t-j)} > X_\text{followed}^{(t-j)}\right)$$

The **predictive horizon** is the maximum $j$ where $\mathrm{AUROC}(j) > 0.75$, measuring how many tokens in advance SVD instability distinguishes violations from compliant walks.

Statistical controls include BCa bootstrap confidence intervals ($n = 10{,}000$ resamples), Cohen's $d$ effect sizes, Spearman correlation redundancy analysis, and exploratory/confirmatory split assignment.

## Configuration

Default anchor configuration (`config.json`):

```json
{
  "graph": { "n": 500, "K": 4, "p_in": 0.25, "p_out": 0.03, "n_jumpers_per_block": 2 },
  "model": { "d_model": 128, "n_layers": 4, "n_heads": 1, "dropout": 0.0 },
  "training": { "w": 64, "walk_length": 256, "corpus_size": 200000, "batch_size": 64, "max_steps": 50000 },
  "seed": 42
}
```

Multi-head ablation: set `n_heads` to 1, 2, or 4 (with `d_model` scaled to keep $d_k = 128$).

## Installation

```bash
git clone https://github.com/ParkerWilliams/dcsbm-transformer.git
cd dcsbm-transformer
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

## Usage

```bash
# Full experiment
python run_experiment.py --config config.json --verbose

# Dry run (show plan without executing)
python run_experiment.py --config config.json --dry-run
```

### Output

```
results/{experiment_id}/
├── result.json                 # All metrics, config, metadata
├── token_metrics.npz           # Per-sequence, per-step SVD metrics
├── spectrum_trajectories.npz   # Top-k singular value trajectories
├── config.json                 # Input config copy
├── figures/                    # PNG (300 dpi) + SVG
└── report.html                 # Self-contained HTML report
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
src/
├── config/          # ExperimentConfig dataclasses, hashing, serialization
├── graph/           # DCSBM generation, jumper designation, validation, caching
├── walk/            # Walk generation (path splicing), compliance, corpus assembly
├── model/           # Transformer (multi-head attention, residual blocks)
├── training/        # Training loop, data loading, checkpointing
├── evaluation/      # Fused SVD extraction, behavioral labeling, split assignment
├── analysis/        # AUROC horizon, statistical controls, signal concentration
├── visualization/   # Figure rendering from result.json
├── reporting/       # HTML report generation (Jinja2 templates)
├── results/         # Experiment ID generation, result storage
└── reproducibility/ # Seeding, git hash tracking
```
