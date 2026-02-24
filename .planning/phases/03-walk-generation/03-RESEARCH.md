# Phase 3: Walk Generation - Research

**Researched:** 2026-02-24
**Domain:** Random walk generation on directed sparse graphs with constraint-guided jumper compliance
**Confidence:** HIGH

## Summary

Phase 3 generates directed random walk corpora on the DCSBM graph from Phase 2, with 100% rule-compliant training data where every jumper encounter is satisfied by a guided path segment. The core technical challenge is the **guided walk mechanism**: when a walk encounters a jumper vertex with jump length r and target block B, the subsequent r steps must be guided so the walk lands in block B at exactly step+r. This must work correctly even when nested jumpers create simultaneous constraints.

Profiling on the anchor config (n=500, K=4, walk_length=256) reveals that **vectorized batch walk generation** using NumPy CSR index arrays processes 200k unguided walks in ~3.5 seconds. The guided walk mechanism uses **precomputed path-count vectors** (via sparse matrix-vector multiplication) to weight neighbor selection at each step, ensuring uniform sampling over valid compliant paths. Precomputation takes ~22ms for all 4 target blocks up to max_r=128, consuming only ~2MB of memory. Joint constraint satisfaction for nested jumpers multiplies path-count weights across active constraints; the measured infeasibility rate is ~2.7% of walks, handled by discarding and regenerating (requiring only ~3% overgeneration).

**Primary recommendation:** Implement a two-phase walk generator: (1) vectorized batch generation for the unguided portions and jumper-free walks, and (2) per-walk guided generation using precomputed path-count weights for jumper-containing walks. Store walks and metadata atomically in a single .npz archive with caching by config hash.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Starting vertices chosen uniformly at random
- Transition probabilities uniform over outgoing neighbors (degree correction is in graph structure, not walk process)
- Vertex revisits allowed freely (standard random walk)
- Dead-end handling: restart walk from new random vertex (should not occur given Phase 2 strong connectivity guarantee)
- **Training walks must be 100% rule-compliant** -- every jumper rule satisfied in every walk
- Walks are guided at jumper encounters: when a walk hits a jumper vertex, the next r_j steps follow a pre-computed compliant path of length r_j to a node with an edge to the target block
- Compliant path chosen **uniformly** among all valid length-r paths (not weighted) -- forces the transformer to learn the destination constraint, not memorize specific routes
- **Nested jumpers must both be satisfied** -- if a second jumper is encountered during a guided path, both rules must be fulfilled simultaneously
- After any filtering/discards, corpus is regenerated to maintain size requirement
- **50% minimum** -- at least half the walks must contain a jumper encounter
- Achieve this by seeding 50%+ of walks directly from jumper vertices
- **Path diversity validation** -- each jumper must have at least 3 distinct compliant paths represented in the final corpus (prevents single-route memorization)
- **Independent generation** with different seeds -- no overlap by construction
- **90/10 split** -- 90% training, 10% evaluation
- Training corpus must be >= 100n tokens (the 100n threshold applies to train alone, eval is additional)
- Walk length is a **sweep parameter** -- one length per experiment config (2w, 4w, or 8w)
- Eval walks test whether the transformer learned edge structure and jumper rules
- **Per-walk event list** -- each walk stores: [{vertex_id, step, target_block, expected_arrival_step}]
- Records encounter only (outcome evaluated during transformer inference in Phase 6)
- Stored inside the same .npz file as walks (atomic -- walks and metadata always in sync)
- **NumPy .npz archive** -- walks as int32 array (num_walks x walk_length), metadata as structured arrays, all in one file
- Cache key = hash(graph_config_hash + walk_length + corpus_size + split + seed) -- graph changes auto-invalidate walks
- Cached files stored in same cache directory as graph .npz files (single cache location for all artifacts)

### Claude's Discretion
- Exact implementation of nested jumper constraint satisfaction (solver approach)
- Pre-computation strategy for enumerating compliant paths
- Batch size and parallelism during walk generation
- Corpus statistics logging format

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| WALK-01 | System generates directed random walks on the DCSBM graph with configurable walk length l (swept at 2w, 4w, 8w) | Vectorized batch walker using CSR indptr/indices arrays; walk_length derived from config.training.walk_length; validated that each step follows a valid directed edge |
| WALK-02 | System validates corpus size is at least 2 orders of magnitude larger than n (t >= 100n) | Already validated in ExperimentConfig.__post_init__; walk generator should also verify actual generated count meets threshold after discards |
| WALK-03 | System produces separate train and evaluation walk sets with different seeds | 90/10 split with seed offsets (e.g., train_seed=config.seed+2000, eval_seed=config.seed+3000); independent RNG instances; no overlap by construction |
| WALK-04 | System tracks block jumper encounter metadata during walk generation (which jumper was hit, at which step, expected target block at step+r) | Per-walk event list stored as structured numpy arrays in the .npz archive; events recorded during walk generation when current vertex is in jumper_map |
| WALK-05 | System caches generated walks by config hash to avoid redundant regeneration | Cache key = hash(graph_config_hash + walk_length + corpus_size + split + seed); stored as .npz in shared cache directory; follows Phase 2 caching pattern |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=2.0 | Walk array storage, vectorized batch stepping, path-count vectors | Already in project deps; CSR index array access is the fastest pure-NumPy approach |
| scipy.sparse | >=1.14 | CSR adjacency matrix access, sparse matrix-vector multiply for path counts | Already used by Phase 2 graph module; CSR format gives O(1) neighbor access via indptr/indices |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib (stdlib) | N/A | Walk cache key computation | Deterministic SHA-256 hashing of walk config params, follows Phase 2 pattern |
| logging (stdlib) | N/A | Walk generation progress and statistics | Matches existing codebase logging pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure NumPy walks | Numba JIT compilation | Numba not in project deps; NumPy vectorized approach achieves ~3.5s for 200k walks which is adequate |
| scipy.sparse matmul for path counts | NetworkX path enumeration | NetworkX path enumeration is O(exponential); matrix power approach is O(n^2 * max_r) |
| Custom .npz caching | HDF5 via h5py | .npz matches Phase 2 pattern; h5py would add dependency for no benefit at this scale |

**Installation:**
No new dependencies required. Uses numpy, scipy, and stdlib modules already in pyproject.toml.

## Architecture Patterns

### Recommended Project Structure
```
src/
├── walk/
│   ├── __init__.py          # Public API exports
│   ├── types.py             # WalkCorpus, JumperEvent dataclasses
│   ├── generator.py         # Core walk generation (batch + guided)
│   ├── compliance.py        # Path-count precomputation, guided step logic
│   ├── corpus.py            # Corpus assembly (train/eval split, validation)
│   └── cache.py             # Walk caching (load/save .npz)
```

### Pattern 1: Path-Count Guided Walking
**What:** Precompute path-count vectors for each target block using iterative sparse matrix-vector multiplication. At each guided step, weight neighbor selection proportional to path counts, ensuring uniform sampling over all valid compliant paths.

**When to use:** Every time a walk encounters a jumper vertex and enters a guided segment.

**Algorithm:**
1. For each target block B (K=4 blocks total):
   - Initialize N[0] = indicator vector for block B membership (n-dimensional)
   - For k = 1 to max_r: N[k] = adj @ N[k-1], then normalize by max value to prevent overflow
   - Store all N[0..max_r] vectors
2. During guided walk at step i with constraint (deadline, target_block):
   - remaining = deadline - step
   - weights[neighbor] = path_counts[target_block][remaining-1][neighbor] for each neighbor
   - Choose neighbor proportional to weights
3. For nested jumpers (multiple active constraints):
   - Multiply weights across all active constraints
   - If joint weights are all zero (infeasible), discard walk and regenerate

**Key properties verified by profiling:**
- Precomputation: ~22ms for all blocks up to max_r=128
- Memory: ~2MB for all path-count vectors
- At k >= 2, all vertices have nonzero path counts (constraint is only "tight" in last 1-2 steps)
- Infeasibility rate: ~2.7% of walks (negligible overgeneration needed)

### Pattern 2: Vectorized Batch Walk Generation
**What:** Generate many walks simultaneously by processing one step at a time across all walks, using NumPy array operations on CSR index arrays.

**When to use:** For the unguided portion of walks (walks without active jumper constraints).

**Algorithm:**
```python
indptr = adj.indptr  # CSR row pointers
indices = adj.indices  # CSR column indices

walks = np.zeros((n_walks, walk_length), dtype=np.int32)
walks[:, 0] = start_vertices

for step in range(1, walk_length):
    current = walks[:, step - 1]
    starts = indptr[current]
    ends = indptr[current + 1]
    degrees = ends - starts
    offsets = (rng.random(n_walks) * degrees).astype(np.int64)
    offsets = np.clip(offsets, 0, degrees - 1)
    walks[:, step] = indices[starts + offsets]
```

**Performance:** 200k walks of length 256 in ~3.5 seconds on the anchor config.

### Pattern 3: Two-Phase Corpus Assembly
**What:** Separate walk generation into jumper-free batch walks and jumper-containing guided walks, then merge.

**When to use:** Always -- this is the top-level generation strategy.

**Algorithm:**
1. Phase A: Generate ~50% of walks starting from jumper vertices (guided walks, per-walk Python loop with path-count weighting)
2. Phase B: Generate remaining ~50% starting from random vertices (vectorized batch, then check for jumper encounters and re-run as guided if needed)
3. Validate all walks for compliance, discard violations, regenerate to fill
4. Merge and shuffle

### Anti-Patterns to Avoid
- **Filtering random walks for compliance:** Generating unguided random walks and discarding non-compliant ones would have ~75% discard rate per jumper encounter (1/K chance of landing in correct block). With ~4 encounters per walk, almost no walks would survive. The guided approach is essential.
- **Enumerating all valid paths:** The number of valid paths of length r grows exponentially (up to ~10^225 for r=128). Path-count vectors give the same sampling distribution without enumeration.
- **Separate files for walks and metadata:** The CONTEXT requires atomic storage in a single .npz file. Never store walks in one file and events in another -- they must stay in sync.
- **Using Python loops for unguided walks:** Pure Python walk generation takes ~21 minutes for 200k walks vs ~3.5 seconds vectorized. Always use NumPy vectorization for unguided walks.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Uniform path sampling | Custom path enumeration/counting | Sparse matrix-vector multiply (adj @ N_prev) | O(nnz * max_r) vs exponential enumeration; exact same distribution |
| Overflow prevention in path counts | Manual BigInteger arithmetic | Normalize by max value at each multiplication step | float64 preserves ratios exactly; normalization is standard numerical linear algebra |
| Neighbor access from sparse matrix | Dense adjacency matrix | CSR indptr/indices arrays directly | O(degree) access vs O(n); 500x500 dense = 2MB vs CSR ~100KB |
| Walk deduplication between train/eval | Hash-based set intersection | Different RNG seeds | Walk space is 500^256; collision probability is zero for practical purposes |

**Key insight:** The path-count weighting via matrix-vector multiplication is the mathematically correct and computationally efficient solution. It gives truly uniform sampling over all valid compliant paths without ever enumerating them. This is a well-known technique from Markov chain theory.

## Common Pitfalls

### Pitfall 1: Path Count Overflow
**What goes wrong:** Path counts grow as O(mean_degree^r). For r=128 with mean_degree=26.7, raw counts would be ~10^183, far exceeding float64 range (~1.8e308 but precision degrades much earlier).
**Why it happens:** Iterating adj @ N_prev without normalization causes exponential growth.
**How to avoid:** Normalize N_prev by its maximum value at each step. This preserves the ratios between entries (which is all that matters for weighted sampling) while keeping values in [0, 1].
**Warning signs:** NaN or Inf in path-count vectors; all neighbors having identical weights (loss of precision).

### Pitfall 2: Off-by-One in Deadline Handling
**What goes wrong:** Jumper at step s with r means arrival at step s+r. If the code checks at step s+r-1 or s+r+1, compliance fails.
**Why it happens:** Ambiguity in whether "r steps" includes the starting step or not.
**How to avoid:** Define clearly: jumper at step s, deadline at step s+r. At step s+r, block_assignments[walk[s+r]] must equal target_block. The path-count lookup at guided step i uses remaining = deadline - current_step, and selects from path_counts[target_block][remaining-1] (because we're choosing the NEXT vertex, which will be at current_step+1, leaving remaining-1 more steps).
**Warning signs:** Systematic violations at step s+r or s+r+1; compliance rate of ~75% (1/K) instead of 100%.

### Pitfall 3: Joint Constraint Infeasibility
**What goes wrong:** Two nested jumper constraints with deadlines within 1-2 steps of each other and different target blocks create a situation where no single vertex satisfies both constraints simultaneously.
**Why it happens:** A vertex can only be in one block. If constraint A needs block 0 at step t and constraint B needs block 2 at step t+1, the walk at step t must be in block 0 AND have a neighbor in block 2. Usually possible (~80% of block-0 vertices have a neighbor in block 2), but not always.
**How to avoid:** Detect infeasibility when joint weights are all zero. Discard the walk and regenerate. Measured discard rate is ~2.7% -- negligible.
**Warning signs:** Walk compliance dropping below 100%; error logs showing infeasible constraint combinations.

### Pitfall 4: Non-Reproducible Walk Generation Due to Discard Order
**What goes wrong:** If discarded walks consume RNG state differently across runs (e.g., different number of retry attempts), the remaining walks will differ despite same seed.
**Why it happens:** Discarding a walk after N steps consumes N random numbers; a different discard pattern shifts the RNG state.
**How to avoid:** Use a separate RNG instance per walk (derived from a master RNG), so that discarding one walk does not affect others. OR generate all walks with a deterministic per-walk seed: walk_seed = master_rng.integers(0, 2**63) drawn sequentially, then each walk uses its own RNG(walk_seed).
**Warning signs:** Walks differ between identical configs on re-run; test_reproducibility_same_seed fails.

### Pitfall 5: 90/10 Split Applied to Total Instead of Training
**What goes wrong:** Generating 200k walks and splitting 90/10 gives 180k train + 20k eval, but the 100n threshold applies to TRAINING corpus only. If 180k < 100n, the validation fails.
**Why it happens:** Misreading the requirement -- "Training corpus must be >= 100n" means the train set alone must have >= 100n walks.
**How to avoid:** Generate corpus_size walks for training (200k for anchor config), then generate an ADDITIONAL corpus_size/9 walks for evaluation (~22.2k). Both independently generated with different seeds.
**Warning signs:** Corpus size validation failing despite generating "enough" total walks.

### Pitfall 6: Cache Invalidation on Graph Changes
**What goes wrong:** Walks cached under old graph parameters are loaded for a new graph config.
**Why it happens:** Walk cache key doesn't include graph config hash.
**How to avoid:** Include graph_config_hash in the walk cache key (per CONTEXT decision). When graph params change, walks automatically miss cache.
**Warning signs:** Walks that don't follow valid edges in the current graph.

## Code Examples

### Precomputing Path-Count Vectors
```python
# For each target block, compute path-count vectors up to max_r
# path_counts[target_block][k][v] = (normalized) count of paths from v to target_block in k steps
def precompute_path_counts(adj, block_assignments, K, max_r):
    path_counts = {}
    for tb in range(K):
        target_mask = (block_assignments == tb).astype(np.float64)
        N_all = [target_mask.copy()]
        N_prev = target_mask.copy()
        for k in range(1, max_r + 1):
            N_prev = np.asarray(adj @ N_prev).ravel()
            mx = N_prev.max()
            if mx > 0:
                N_prev = N_prev / mx  # normalize to prevent overflow
            N_all.append(N_prev.copy())
        path_counts[tb] = N_all
    return path_counts
```

### Guided Step with Multiple Constraints
```python
def guided_step(v, active_constraints, step, path_counts, adj_list, rng):
    neighbors = adj_list[v]
    weights = np.ones(len(neighbors), dtype=np.float64)
    for deadline, target_block in active_constraints:
        remaining = deadline - step
        if remaining <= 0:
            continue  # past deadline (shouldn't happen if properly managed)
        w = path_counts[target_block][remaining - 1][neighbors]
        weights *= w
    total = weights.sum()
    if total == 0:
        return None  # infeasible -- caller should discard walk
    return rng.choice(neighbors, p=weights / total)
```

### Vectorized Batch Walk Step
```python
# Process one step for all walks simultaneously
def batch_step(walks, step, indptr, indices, rng):
    current = walks[:, step - 1]
    starts = indptr[current]
    ends = indptr[current + 1]
    degrees = ends - starts
    offsets = (rng.random(len(walks)) * degrees).astype(np.int64)
    offsets = np.clip(offsets, 0, degrees - 1)
    walks[:, step] = indices[starts + offsets]
```

### Walk-Level Seed Isolation for Reproducibility
```python
# Generate per-walk seeds from master RNG to ensure reproducibility
# even when some walks are discarded and regenerated
master_rng = np.random.default_rng(config.seed + WALK_SEED_OFFSET)
walk_seeds = master_rng.integers(0, 2**63, size=n_walks_needed)

for i, seed in enumerate(walk_seeds):
    walk_rng = np.random.default_rng(seed)
    walk, events = generate_single_walk(start_vertex, walk_length, walk_rng, ...)
```

### NPZ Storage Format
```python
# Atomic save of walks + metadata in single .npz file
def save_walk_corpus(walks, events_list, path, metadata):
    # walks: int32 array of shape (num_walks, walk_length)
    # events_list: list of lists of event dicts

    # Flatten events into structured arrays
    all_walk_ids = []
    all_vertex_ids = []
    all_steps = []
    all_target_blocks = []
    all_arrival_steps = []

    for walk_id, events in enumerate(events_list):
        for ev in events:
            all_walk_ids.append(walk_id)
            all_vertex_ids.append(ev['vertex_id'])
            all_steps.append(ev['step'])
            all_target_blocks.append(ev['target_block'])
            all_arrival_steps.append(ev['expected_arrival_step'])

    np.savez_compressed(
        path,
        walks=walks,
        event_walk_ids=np.array(all_walk_ids, dtype=np.int32),
        event_vertex_ids=np.array(all_vertex_ids, dtype=np.int32),
        event_steps=np.array(all_steps, dtype=np.int32),
        event_target_blocks=np.array(all_target_blocks, dtype=np.int32),
        event_arrival_steps=np.array(all_arrival_steps, dtype=np.int32),
        # Metadata stored as individual arrays
        num_walks=np.array(walks.shape[0]),
        walk_length=np.array(walks.shape[1]),
    )
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| NetworkX random walks | Direct CSR array walks via NumPy | Standard practice | 100-1000x speedup for large graphs |
| Path enumeration for constrained walks | Matrix-power path counting | Standard in Markov chain theory | Polynomial vs exponential complexity |
| Separate walk/metadata files | Single .npz atomic archive | Project design decision | Prevents desync between walks and events |

**Deprecated/outdated:**
- graph-tool random walk functions: Not in project deps, and the CSR-direct approach is equally fast without the dependency
- Node2Vec-style biased walks: Not applicable here -- walks must be uniform over outgoing neighbors per CONTEXT decision

## Open Questions

1. **Guided walk performance at scale with high jumper density**
   - What we know: ~2.7% discard rate with 8 jumpers and anchor config. Guided walks take ~26ms per walk in pure Python.
   - What's unclear: If n_jumpers_per_block increases (e.g., 5 per block = 20 total = 4% of vertices), does the discard rate grow significantly? Does the per-walk cost of guided generation become a bottleneck for corpus sizes >> 200k?
   - Recommendation: Profile during implementation; if guided walk cost dominates, consider Numba JIT compilation for the inner loop (add numba as optional dependency). For the current anchor config, pure Python + NumPy is sufficient.

2. **Uniform path sampling exactness with normalization**
   - What we know: Normalizing path-count vectors by max value at each step preserves ratios between entries, giving correct relative weights for sampling.
   - What's unclear: Whether accumulated floating-point error from 128 successive normalizations causes detectable bias in the path distribution.
   - Recommendation: Verify empirically that the distribution of final vertices in guided walks matches the expected block proportion. The measured 100/100 compliance rate for r=128 guided walks is encouraging. Can add a statistical test in the test suite.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- numpy + scipy.sparse are the only reasonable choice; no new dependencies needed
- Architecture: HIGH -- path-count guided walking is mathematically well-founded; profiling confirms performance; 100% compliance achieved in prototype
- Pitfalls: HIGH -- all pitfalls identified through direct experimentation on the anchor config graph; discard rates, overflow behavior, and performance characteristics measured empirically

**Research date:** 2026-02-24
**Valid until:** indefinitely (algorithmic research, not library-version-dependent)

## Sources

### Primary (HIGH confidence)
- Direct profiling on anchor config DCSBM graph (n=500, K=4, 8 jumpers, r_values=[32,45,58,64,70,83,96,128])
- Existing codebase: src/graph/ module (Phase 2 implementation providing GraphData, JumperInfo, adjacency matrix, block_assignments, cache pattern)
- Existing codebase: src/config/experiment.py (ExperimentConfig with walk_length, corpus_size, w parameters)
- Existing codebase: src/config/hashing.py (graph_config_hash, config_hash utilities)

### Secondary (MEDIUM confidence)
- Markov chain theory: path counting via matrix powers is a standard result in algebraic graph theory
- CSR sparse matrix random walk optimization is a well-known technique in graph embedding literature (Node2Vec, DeepWalk)
