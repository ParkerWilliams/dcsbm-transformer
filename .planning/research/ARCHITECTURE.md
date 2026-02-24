# Architecture Patterns

**Domain:** Research framework -- DCSBM transformer SVD hallucination prediction
**Researched:** 2026-02-24

## Recommended Architecture

The system is a linear research pipeline with a parameter sweep orchestrator wrapping around it. There are seven distinct modules, each with a single responsibility, communicating through well-defined data contracts (files on disk or typed Python objects). The critical design constraint is that SVD collection is the computational bottleneck and must be tightly integrated with model inference, not bolted on after the fact.

### High-Level Data Flow

```
[1. Config]
    |
    v
[2. Graph Gen] --> adjacency matrix + metadata (pickle/npz)
    |
    v
[3. Walk Gen] --> walk corpus (tokenized sequences, .pt files)
    |
    v
[4. Transformer Training] --> model checkpoint (.pt)
    |                          + training curves (JSON)
    |
    v
[5. Sufficiency Gate] -- FAIL --> flag config, skip SVD
    |
    | PASS
    v
[6. Evaluation + SVD Collection] --> per-step behavioral labels
    |                                 + per-step SVD metrics
    |                                 (combined into sequences[])
    v
[7. Analysis + Reporting] --> result.json
                              + figures/
                              + report.html
```

The sweep orchestrator wraps steps 2-7 and manages the job queue, priority ordering, and budget tracking.

### Component Boundaries

| Component | Responsibility | Communicates With | Key Interface |
|-----------|---------------|-------------------|---------------|
| `config` | Define, validate, serialize experiment configurations | All modules read config | `ExperimentConfig` dataclass |
| `graph` | DCSBM generation, block jumper designation, adjacency storage | Walk Gen reads adjacency + block info | `DCBSMGraph` object with adj matrix, block assignments, jumper rules |
| `walks` | Random walk sampling, corpus construction, train/eval split | Training reads walk corpus | `.pt` files of tokenized walk tensors |
| `model` | NanoGPT-scale transformer definition, single-head attention | Training uses model, Eval hooks into model | `nn.Module` with explicit QKV access |
| `training` | Training loop, loss tracking, checkpoint saving, sufficiency gate | Reads walks + model, writes checkpoints + training curves | Checkpoint `.pt` + `training_log.json` |
| `evaluation` | Behavioral eval + SVD metric extraction in a single forward pass | Reads checkpoint + eval walks + graph rules, writes sequences | `sequences[]` conforming to result schema |
| `analysis` | Predictive horizon computation, statistical tests, plotting, reporting | Reads result.json, writes figures + reports | `result.json` conforming to schema |
| `sweep` | Job queue management, priority ordering, budget tracking | Wraps all other modules, reads/writes sweep state | `sweep_state.json` + job queue |

### Data Flow Detail

**Phase 1: Configuration**
- Input: sweep parameters (YAML or Python dict)
- Output: list of `ExperimentConfig` objects, each fully specifying one run
- Config includes: n, w, t, d_model, n_layers, r, p_in, p_out, n_blocks, n_jumpers, walk_length, seed
- Config is frozen and serialized before any computation begins

**Phase 2: Graph Generation**
- Input: `ExperimentConfig` (graph params only)
- Output: `DCBSMGraph` containing:
  - Adjacency matrix (sparse, directed) -- `scipy.sparse.csr_matrix`
  - Block assignments: `np.ndarray[int]` of length n
  - Degree corrections: `np.ndarray[float]` of length n
  - Block jumper rules: `dict[int, JumperRule]` mapping vertex -> (jump_length, target_block)
  - Saved as `.npz` + `.json` metadata for reproducibility
- Graph is generated ONCE per unique (n, n_blocks, p_in, p_out, n_jumpers, seed) tuple
- Multiple configs sharing the same graph params should reuse the cached graph

**Phase 3: Walk Generation**
- Input: `DCBSMGraph` + walk params (t, walk_length)
- Output: tokenized walk tensors split into train/eval
  - `train_walks.pt`: tensor of shape `(num_train_walks, walk_length)`
  - `eval_walks.pt`: tensor of shape `(num_eval_walks, walk_length)`
  - `walk_metadata.json`: which walks contain jumper vertices, positions of jumper activations
- Walk generation is a random process: from each starting vertex, follow outgoing edges weighted by degree correction
- Walks are integer-encoded (vertex IDs are token IDs; vocabulary size = n)
- Walk corpus is generated ONCE per unique (graph_id, t, walk_length, seed) tuple

**Phase 4: Training**
- Input: `train_walks.pt` + model config (d_model, n_layers, w)
- Output:
  - Model checkpoint: `model.pt` (state_dict + optimizer_state + epoch)
  - Training log: `training_log.json` with loss curves per step
- Standard next-token prediction with cross-entropy loss
- Vocabulary size = n (each vertex is a token)
- Model architecture: embedding(n, d_model) -> n_layers x TransformerBlock -> linear(d_model, n)
- Each TransformerBlock: single-head causal self-attention + FFN + LayerNorm

**Phase 5: Sufficiency Gate**
- Input: trained model + `eval_walks.pt` + `DCBSMGraph`
- Output: PASS/FAIL + compliance metrics
- Edge compliance: fraction of generated next-tokens that correspond to valid edges
- Rule compliance: fraction of jumper-activated positions where the model lands in the correct target block
- Thresholds: edge > 95%, rule > 80%
- FAIL: config is logged with compliance metrics but excluded from SVD analysis

**Phase 6: Evaluation + SVD Collection (the expensive step)**
- Input: trained model + `eval_walks.pt` + `DCBSMGraph` + `ExperimentConfig`
- Output: `sequences[]` array conforming to result schema
- This is where the computational cost lives. At every token step during eval:
  1. Run forward pass, extract Q and K matrices from the single attention head
  2. Compute QK^T (shape: `(seq_pos, seq_pos)` or more precisely `(current_ctx_len, current_ctx_len)`)
  3. Run `torch.linalg.svd(QKT, full_matrices=False)`
  4. Compute all ~20 SVD metrics from the singular values and vectors
  5. Classify behavioral outcome (edge valid/invalid x rule followed/violated/NA)
  6. Store per-step metrics + labels
- The QK^T matrix at step t has shape `(min(t, w), min(t, w))` -- it grows with context up to w

**Phase 7: Analysis + Reporting**
- Input: `result.json` (complete)
- Output: figures/, report.html, comparison reports
- Predictive horizon computation: for each metric, compute AUROC at each lookback j
- Statistical tests: Mann-Whitney U, Wilson intervals
- Plotting: all plot types from the plotting guide
- Report generation: HTML with embedded base64 figures

## Patterns to Follow

### Pattern 1: Config-Driven Everything

**What:** Every module takes an `ExperimentConfig` (or a subset of it) as its primary input. No hardcoded parameters anywhere.

**When:** Always. Every function that could conceivably vary between runs takes its parameters from config.

**Example:**
```python
@dataclass(frozen=True)
class GraphConfig:
    n: int
    n_blocks: int
    p_in: float
    p_out: float
    n_jumpers_per_block: int
    seed: int

@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_layers: int
    context_window: int  # w
    vocab_size: int      # = n

@dataclass(frozen=True)
class ExperimentConfig:
    graph: GraphConfig
    model: ModelConfig
    walk_length: int         # l
    corpus_size: int         # t
    jump_length: int         # r
    seeds: list[int]         # random seeds for replication
    experiment_slug: str
```

### Pattern 2: Forward Hook for QK^T Extraction

**What:** Use PyTorch forward hooks to extract Q, K matrices from the attention layer without modifying the model architecture. This keeps the model code clean and the extraction concern separate.

**When:** During evaluation (Phase 6). Do NOT use hooks during training -- they add overhead and are unnecessary.

**Example:**
```python
class QKTExtractor:
    """Attaches to a single-head attention layer and captures QK^T."""

    def __init__(self, attention_layer: nn.Module):
        self.qkt = None
        self._hook = attention_layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # Assumes the attention layer exposes Q, K internally
        # For NanoGPT-style: override forward to store Q, K as attributes
        Q = module._last_Q  # set during forward
        K = module._last_K
        self.qkt = Q @ K.transpose(-2, -1)

    def remove(self):
        self._hook.remove()
```

**Alternative (preferred for this project):** Since we control the model code and have a single attention head, it is cleaner to have the attention layer return QK^T as part of its output during eval mode rather than using hooks. This avoids the fragility of hooks.

```python
class SingleHeadAttention(nn.Module):
    def forward(self, x, return_qkt=False):
        Q = self.W_Q(x)  # (B, T, d)
        K = self.W_K(x)
        V = self.W_V(x)
        qkt = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)
        attn = F.softmax(qkt.masked_fill(self.causal_mask[:T, :T] == 0, float('-inf')), dim=-1)
        out = attn @ V
        if return_qkt:
            return out, qkt  # return raw (pre-softmax) QK^T
        return out
```

### Pattern 3: Lazy SVD Batching

**What:** Instead of computing SVD independently at every token position, batch SVD computations where possible. Since `torch.linalg.svd` can operate on batched input, accumulate QK^T matrices across positions and SVD them together.

**When:** During evaluation, when processing a single walk sequence of length L.

**Example:**
```python
def collect_svd_metrics_batched(model, walk_tensor, config):
    """Process one walk, collecting SVD metrics at every step."""
    model.eval()
    w = config.model.context_window
    L = walk_tensor.shape[0]

    # Collect QK^T at each step
    qkt_list = []
    with torch.no_grad():
        for t in range(1, L):
            ctx = walk_tensor[max(0, t - w):t].unsqueeze(0)  # (1, ctx_len)
            _, qkt = model(ctx, return_qkt=True)  # (1, ctx_len, ctx_len)
            qkt_list.append(qkt.squeeze(0))  # (ctx_len, ctx_len)

    # Batch SVD for same-size QK^T matrices
    # Group by context length (all steps where ctx_len == w can be batched)
    from itertools import groupby
    size_groups = {}
    for idx, qkt in enumerate(qkt_list):
        sz = qkt.shape[0]
        size_groups.setdefault(sz, []).append((idx, qkt))

    svd_results = [None] * len(qkt_list)
    for sz, group in size_groups.items():
        indices, matrices = zip(*group)
        batch = torch.stack(matrices)  # (batch_size, sz, sz)
        U, S, Vh = torch.linalg.svd(batch, full_matrices=False)
        for i, idx in enumerate(indices):
            svd_results[idx] = (U[i], S[i], Vh[i])

    return svd_results
```

**Why this matters:** For a walk of length L with context window w, steps 1 through w-1 each have a different-sized QK^T matrix (1x1, 2x2, ..., (w-1)x(w-1)). Steps w through L all have the same size (wxw). The wxw batch is where most of the computation lives, and batching it is a significant win.

### Pattern 4: Result Schema as the Single Source of Truth

**What:** All downstream code (plotting, reporting, comparison) reads exclusively from `result.json`. No module passes in-memory objects to the reporting layer. Write result first, then read it back for analysis.

**When:** Always. This is a hard rule from the spec.

**Why:** Reproducibility. Any figure or report can be regenerated from the stored JSON without re-running the experiment. This also means the evaluation module must write a complete `result.json` before any analysis begins.

### Pattern 5: Graph and Walk Caching

**What:** Graph generation and walk generation are deterministic given the same parameters and seed. Cache results on disk keyed by a hash of the relevant config subset.

**When:** During parameter sweeps. Many configs share the same graph (varying only r, d_model, n_layers) or the same walks (varying only model architecture).

**Example:**
```python
import hashlib, json

def config_hash(config_subset: dict) -> str:
    """Deterministic hash of config parameters."""
    canonical = json.dumps(config_subset, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]

def get_or_generate_graph(graph_config: GraphConfig, cache_dir: str = "cache/graphs"):
    h = config_hash(asdict(graph_config))
    path = os.path.join(cache_dir, h)
    if os.path.exists(path):
        return DCBSMGraph.load(path)
    graph = generate_dcsbm(graph_config)
    graph.save(path)
    return graph
```

### Pattern 6: Evaluation as a Single Fused Pass

**What:** Behavioral evaluation and SVD metric collection happen in the SAME forward pass. Do not run the model twice (once for eval, once for SVD). The evaluation module produces the complete `sequences[]` array in one pass per walk.

**When:** Always during Phase 6.

**Why:** The forward pass is the expensive part (GPU compute). Running it twice would double the wall time for no reason. The model forward returns both the prediction logits (for behavioral classification) and the QK^T matrix (for SVD analysis) simultaneously.

## Anti-Patterns to Avoid

### Anti-Pattern 1: In-Memory Pipeline

**What:** Passing large tensors between pipeline stages through Python objects in a single script.

**Why bad:** Memory pressure (graph + walks + model + SVD results all in RAM simultaneously), no ability to resume from a failed run, no caching, no parallelism.

**Instead:** Each stage writes its output to disk. The next stage reads from disk. This enables: (a) caching and reuse across configs, (b) resume from any failed stage, (c) independent profiling of each stage, (d) running stages on different hardware if needed.

### Anti-Pattern 2: SVD After-the-Fact

**What:** Training the model, then doing a separate "SVD analysis pass" that re-runs inference just to collect QK^T.

**Why bad:** Doubles the inference compute. The QK^T matrix is available during the same forward pass that produces predictions.

**Instead:** Use Pattern 6 (fused evaluation pass). The model's forward method returns both logits and QK^T when `return_qkt=True`.

### Anti-Pattern 3: Giant result.json

**What:** Storing all SVD metrics for all token positions for all eval walks in a single result.json, producing multi-GB files.

**Why bad:** JSON parsing becomes the bottleneck. Loading a 2GB JSON into memory to plot one figure is wasteful.

**Instead:** Use a hybrid storage strategy:
- `result.json` stores: config, scalar metrics, curves, confusion matrix, statistical tests, predictive horizon data, and SUMMARY statistics of SVD metrics (means, stds, distributions at key positions).
- `token_metrics.npz` (companion file): stores the full per-step SVD metric tensors as numpy arrays, referenced by sequence_id. This file is only loaded when needed for detailed per-token analysis.
- Plotting functions check for `token_metrics.npz` and fall back to summary data in `result.json`.

### Anti-Pattern 4: Monolithic Sweep Script

**What:** A single `run_sweep.py` that loops over all configs sequentially, with no ability to pause, resume, reprioritize, or monitor progress.

**Why bad:** With a $100 budget, you need to be able to stop at any point and have complete results for the most important configs. A monolithic loop means partial results are useless.

**Instead:** Use a job queue pattern. Each config is a job. Jobs are prioritized (anchor config first, then r-sweep, then secondary sweeps). The sweep runner picks the next highest-priority incomplete job, runs it, writes results, and updates the queue state. You can stop and restart at any time.

### Anti-Pattern 5: Recomputing SVD Metrics in Analysis

**What:** Analysis code that re-derives SVD metrics from raw singular values stored in result.json.

**Why bad:** Different definitions of the "same" metric lead to subtle bugs. The metric computation should happen exactly once, during evaluation.

**Instead:** Compute all ~20 metrics during the evaluation pass and store them by name. Analysis code reads metric values directly; it never touches singular values or vectors.

## Module Dependency Graph (Build Order)

The build order is determined by what each module needs to function and be testable independently.

```
Level 0 (no dependencies):
  config        -- dataclasses, validation, serialization
  graph         -- DCSBM generation (only needs numpy/scipy)

Level 1 (depends on Level 0):
  walks         -- needs graph
  model         -- needs config (for architecture params)

Level 2 (depends on Level 1):
  training      -- needs model + walks

Level 3 (depends on Level 2):
  evaluation    -- needs trained model + walks + graph + SVD metric computation

Level 4 (depends on Level 3):
  analysis      -- needs result.json from evaluation
  sweep         -- wraps levels 0-4
```

### Suggested Build Sequence

1. **config** -- Define all dataclasses first. Everything else imports these. Testable with pure unit tests (validation, serialization, hashing).

2. **graph** -- DCSBM generation is standalone math. Test: generate a graph, verify block structure, edge density matches p_in/p_out, jumper rules are satisfiable (valid paths of length r exist to target block). No PyTorch dependency.

3. **model** -- Define the NanoGPT-scale transformer. Test: forward pass with random input, verify output shape, verify QK^T extraction works with `return_qkt=True`, verify single-head constraint.

4. **walks** -- Walk generation on the graph. Test: generate walks, verify all transitions are valid edges, verify walk length, verify jumper vertices appear at expected frequency.

5. **training** -- Training loop. Test: train on a tiny graph (n=20, w=8) for a few epochs, verify loss decreases, verify checkpoint save/load round-trips.

6. **evaluation** -- The most complex module. Build in sub-stages:
   a. Behavioral classification (edge validity + rule compliance) -- test independently
   b. QK^T extraction and SVD computation -- test with random matrices
   c. SVD metric computation (~20 metrics) -- test each metric against hand-computed values
   d. Fused evaluation pass -- integration test combining a-c
   e. Result JSON writer -- test schema compliance

7. **analysis** -- Predictive horizon, plotting, reporting. Test: create synthetic result.json with known patterns, verify AUROC computation, verify plots generate without errors.

8. **sweep** -- Job queue, priority ordering, budget tracking. Test: create a small sweep (3 configs), verify execution order matches priority, verify resume after interruption.

## Detailed Module Specifications

### `config/` Module

```
config/
  __init__.py
  schema.py          # ExperimentConfig, GraphConfig, ModelConfig, etc.
  validation.py      # Validate parameter ranges, cross-parameter constraints
  sweep.py           # Generate config grid from sweep ranges
  hashing.py         # Deterministic config hashing for caching
```

Key cross-parameter constraints to validate:
- `walk_length >= 2 * context_window`
- `corpus_size >= 100 * n`
- `jump_length >= 1`
- Valid paths of length r must exist (checked after graph generation, not here)

### `graph/` Module

```
graph/
  __init__.py
  dcsbm.py           # DCSBM adjacency matrix generation
  jumpers.py         # Block jumper designation and rule validation
  pathcheck.py       # Verify valid paths of length r exist to target block
  io.py              # Save/load graph to/from disk
```

The DCSBM generation:
1. Assign n vertices to n_blocks blocks (balanced or configurable)
2. Generate degree corrections from a Pareto or log-normal distribution
3. For each vertex pair (i, j), edge exists with probability: `theta_i * theta_j * B[block(i), block(j)]` where B is the block interaction matrix with p_in on diagonal, p_out off-diagonal
4. Edges are directed (the adjacency matrix is not symmetric)

### `model/` Module

```
model/
  __init__.py
  transformer.py     # NanoGPT-scale transformer with single-head attention
  attention.py       # SingleHeadAttention with return_qkt option
```

Architecture for anchor config (d_model=128, n_layers=4, w=64, n=500):
- Token embedding: (500, 128)
- Position embedding: (64, 128)
- 4 transformer blocks, each:
  - SingleHeadAttention(d_model=128)
  - FFN(d_model=128, d_ff=512) -- 4x expansion
  - LayerNorm
- Output linear: (128, 500)
- Total parameters: approximately 500*128 + 64*128 + 4*(128*128*3 + 128*128 + 128*512*2 + 128*4) + 128*500 = ~860K parameters

### `training/` Module

```
training/
  __init__.py
  trainer.py         # Training loop with logging
  sufficiency.py     # Gate evaluation (edge compliance, rule compliance)
  checkpoint.py      # Save/load model checkpoints
```

### `evaluation/` Module

```
evaluation/
  __init__.py
  behavioral.py      # Edge validity + rule compliance classification
  svd_metrics.py     # All ~20 SVD metric computations
  collector.py       # Fused evaluation pass combining behavioral + SVD
  writer.py          # Write result.json + token_metrics.npz
```

The SVD metrics module is the most mathematically dense. Each metric is a pure function: `(U, S, Vh, prev_U, prev_S, prev_Vh, embeddings) -> float`. Test each independently.

### `analysis/` Module

```
analysis/
  __init__.py
  horizon.py         # Predictive horizon AUROC computation
  statistics.py      # Statistical tests (Mann-Whitney, Wilson, etc.)
  plotting.py        # All plot types from plotting guide
  reporting.py       # HTML report generation
  comparison.py      # Multi-experiment comparison
```

### `sweep/` Module

```
sweep/
  __init__.py
  queue.py           # Job queue with priority ordering
  runner.py          # Execute jobs, track budget
  state.py           # Persist sweep state for resume
```

## Scalability Considerations

| Concern | Anchor Config (n=500, w=64) | Large Config (n=2000, w=256) | Full Sweep (~200 configs) |
|---------|----------------------------|------------------------------|---------------------------|
| Graph memory | ~1MB sparse | ~16MB sparse | Cached, not simultaneous |
| Walk corpus (t=200k) | ~50MB | ~200MB | Cached per unique graph+params |
| QK^T matrix per step | 64x64 = 4KB | 256x256 = 64KB | Only one config runs at a time |
| SVD per step | ~0.1ms (64x64) | ~5ms (256x256) | Dominates wall time for large w |
| SVD total (200k eval steps) | ~20 seconds | ~17 minutes | Budget-critical |
| Token metrics storage | ~100MB per config | ~400MB per config | Use .npz, not JSON |
| result.json (summary) | ~5MB | ~20MB | Manageable |
| Model parameters | ~860K | ~14M | Fits on any GPU |
| Training time (200k walks) | ~10 minutes (est.) | ~2 hours (est.) | Budget-critical |
| Total per config | ~30 minutes | ~3 hours | Must prioritize |

### SVD Optimization Strategy

The SVD at every token step is the defining computational challenge. For the anchor config (w=64), a 64x64 SVD is fast (~0.1ms on GPU). For w=256, a 256x256 SVD takes ~5ms. Over 200k evaluation steps, this is 17 minutes just for SVD.

Optimizations in priority order:

1. **Use `full_matrices=False`** -- reduces SVD output size and computation. Since we only need top-k singular values/vectors for most metrics, this is sufficient.

2. **Batch same-size matrices** -- all steps after the context window fills (step w onwards) have the same QK^T size. Batch them for `torch.linalg.svd`. This is the biggest single win because GPU utilization improves dramatically with batching.

3. **Keep everything on GPU** -- never transfer QK^T to CPU for SVD. `torch.linalg.svd` works on CUDA tensors. All metric computation should also stay on GPU.

4. **Precompute reusable quantities** -- many metrics share intermediate values (e.g., normalized singular values p_i = sigma_i / sum(sigma) are used by entropy, participation ratio, and stable rank). Compute once.

5. **Consider truncated SVD for large w** -- for w=256, if we only need top-k singular values/vectors (e.g., k=8), `torch.svd_lowrank` is much faster than full SVD. However, some metrics (condition number, full entropy, rank) need all singular values. Strategy: compute full SVD for the "full" metrics, but if budget is tight, use truncated SVD and flag which metrics are exact vs. approximate.

6. **Eval walk sampling** -- instead of evaluating ALL walks, evaluate a statistically sufficient subset. For AUROC computation with target significance, ~2000-5000 failure events are sufficient. This can dramatically reduce the number of eval steps.

## File System Layout

```
dcsbm-transformer/
  config/
    __init__.py
    schema.py
    validation.py
    sweep.py
    hashing.py
  graph/
    __init__.py
    dcsbm.py
    jumpers.py
    pathcheck.py
    io.py
  model/
    __init__.py
    transformer.py
    attention.py
  training/
    __init__.py
    trainer.py
    sufficiency.py
    checkpoint.py
  evaluation/
    __init__.py
    behavioral.py
    svd_metrics.py
    collector.py
    writer.py
  analysis/
    __init__.py
    horizon.py
    statistics.py
    plotting.py
    reporting.py
    comparison.py
  sweep/
    __init__.py
    queue.py
    runner.py
    state.py
  tests/
    test_config.py
    test_graph.py
    test_model.py
    test_walks.py
    test_training.py
    test_svd_metrics.py
    test_evaluation.py
    test_analysis.py
    test_sweep.py
  cache/
    graphs/           # cached graph files
    walks/            # cached walk corpora
  results/
    {experiment_id}/
      result.json
      token_metrics.npz
      figures/
      report.html
  run_experiment.py    # single-config entry point
  run_sweep.py         # sweep entry point
  Makefile
  requirements.txt
```

## Critical Interface Contracts

### Between graph and walks
```python
# graph provides:
class DCBSMGraph:
    adj: scipy.sparse.csr_matrix      # (n, n) directed adjacency
    blocks: np.ndarray                  # (n,) block assignment
    degree_corrections: np.ndarray      # (n,) degree correction factors
    jumper_rules: dict[int, JumperRule] # vertex -> (jump_length, target_block)
    n: int
    n_blocks: int

    def neighbors(self, v: int) -> np.ndarray:
        """Return out-neighbors of vertex v."""

    def is_valid_edge(self, u: int, v: int) -> bool:
        """Check if edge u->v exists."""
```

### Between model and evaluation
```python
# model provides:
class Transformer(nn.Module):
    def forward(self, x: torch.Tensor, return_qkt: bool = False):
        """
        x: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
            qkt: (batch, seq_len, seq_len) if return_qkt=True, else None
        """
```

### Between evaluation and analysis
```python
# evaluation writes result.json conforming to schema v1.0 with extensions:
# sequences[i].token_metrics: dict[str, list[float]]
#   where keys are metric names and values are per-step metric values
# sequences[i].behavioral_labels: list[str]
#   where each entry is one of: "edge_valid_rule_followed",
#   "edge_valid_rule_violated", "edge_valid_rule_na",
#   "edge_invalid_rule_na"
```

## Sources

- Project specification: `combined-spec.md` in repository root (HIGH confidence -- primary source)
- PROJECT.md: `.planning/PROJECT.md` (HIGH confidence -- defines constraints and scope)
- PyTorch SVD documentation: `torch.linalg.svd` supports batched input and CUDA tensors (HIGH confidence -- well-established PyTorch API)
- NanoGPT architecture patterns: single-file transformer implementations with explicit Q, K, V access are standard practice (HIGH confidence -- widely adopted pattern)
- SVD computational complexity: O(min(m,n)^2 * max(m,n)) for an m x n matrix; for square wxw matrix this is O(w^3) (HIGH confidence -- standard numerical linear algebra)

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Module boundaries | HIGH | Follows directly from spec; each stage has clear inputs/outputs |
| Data flow | HIGH | Linear pipeline is the natural structure for this problem |
| SVD optimization | MEDIUM | Batching strategy is sound but actual speedups need profiling; truncated SVD tradeoffs need validation on real data |
| Storage strategy (hybrid JSON + npz) | MEDIUM | Deviates from spec's pure-JSON approach; may need refinement based on actual file sizes |
| Build order | HIGH | Dependency chain is unambiguous |
| Model parameter estimates | MEDIUM | Back-of-envelope; actual count depends on implementation details (bias terms, LayerNorm params, etc.) |
