# Phase 2: DCSBM Graph Generation - Research

**Researched:** 2026-02-24
**Domain:** Degree-corrected stochastic block model graph generation, validation, and caching
**Confidence:** HIGH

## Summary

Phase 2 implements a custom DCSBM directed graph generator following Karrer & Newman (2011). The core task is sampling edges from a block-structured probability matrix with per-vertex degree correction parameters drawn from a power-law (Zipf) distribution. The graph must satisfy strong connectivity, minimum expected degree, and edge density constraints. Block jumper vertices with variable r values are designated and validated for non-triviality (paths exist but are not unique). Generated graphs are cached by config hash to avoid redundant computation across sweep configs.

The implementation is straightforward numerical Python: numpy for random sampling and matrix operations, scipy.sparse for the adjacency representation, and standard library tools for caching. No external graph library (networkx, igraph) is needed for generation itself, though scipy's sparse graph algorithms provide strong connectivity checking. The key complexity is in the validation/retry loop and the non-triviality verification for block jumpers.

**Primary recommendation:** Implement edge sampling via direct Bernoulli draws from the DCSBM probability matrix using numpy vectorized operations. Use scipy.sparse.csgraph for strong connectivity verification. Store graphs as scipy CSR sparse matrices with metadata dicts for block assignments and jumper designations. Cache using pickle with gzip compression keyed by graph_config_hash.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Power-law distribution for theta_i parameters (Zipf's law), mimicking token frequency distribution
- Fixed exponent alpha=1.0 (classic Zipf), not swept -- keeps one fewer dimension in the parameter grid
- Normalize theta_i values so expected total degree matches the uncorrected SBM expectation (p_in/p_out control density, theta_i only reshapes distribution)
- Single theta_i per vertex for both in-degree and out-degree (no separate theta_in/theta_out)
- Jumper count per block: fraction of block size, swept at {1%, 2%, 5%}
- Floor rounding with minimum 1 jumper per block (guarantees coverage across all blocks)
- Target blocks assigned randomly per jumper (uniform from blocks != own block). Different jumpers in the same block may have different targets
- **Variable r per jumper within a single graph**: each graph contains jumpers at ALL r values from the discrete set {0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w}, distributed uniformly across r values
- This replaces the config-level r sweep entirely -- r analysis happens at filtering time, not config time
- All r values are rounded to nearest integer after computing from w
- Equal-sized blocks (n/K vertices per block)
- Anchor config: K=4 blocks (~125 vertices per block with n=500)
- K swept over {4, 8, 16} per spec
- Anchor density: p_in=0.25, p_out=0.03 (ratio ~8:1)
- No self-loops
- On validation failure: retry with incremented seed, maximum 10 attempts
- Generation attempt counter keeps config hash stable across retries
- Edge density tolerance: 2 sigma from expected value under the DCSBM model
- Non-triviality check: if a jumper vertex fails, reassign to a different vertex in the same block (retry up to block_size/2 times before full graph regeneration)

### Claude's Discretion
- Graph data structure choice (sparse vs dense representation)
- Caching serialization format
- Edge sampling algorithm (direct Bernoulli sampling vs adjacency matrix construction)
- Internal logging verbosity during generation
- Exact implementation of the power-law theta_i sampling

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| GRPH-01 | DCSBM directed graphs with configurable n, K, p_in, p_out, and degree correction per Karrer & Newman 2011 | Standard Stack (numpy/scipy), Architecture Pattern 1 (DCSBM edge sampling), Code Examples (theta sampling, edge probability) |
| GRPH-02 | Block jumper vertices with configurable jump length r and target block, enforcing the "after r steps" rule | Architecture Pattern 2 (block jumper designation), variable-r-per-graph design from CONTEXT.md |
| GRPH-03 | Graph validation: strongly connected, min expected degree >= 3, edge density matching p_in/p_out | Architecture Pattern 3 (validation gates), scipy.sparse.csgraph for connectivity |
| GRPH-04 | Non-triviality verification: valid paths of length r exist but are not the only paths at that length | Architecture Pattern 4 (non-triviality via BFS/matrix power), Pitfall 3 (path counting) |
| GRPH-05 | Cache generated graphs by config hash | Architecture Pattern 5 (pickle+gzip caching), existing graph_config_hash from Phase 1 |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >= 2.0 | Random sampling, array operations, Bernoulli edge draws | Already in project deps; vectorized edge sampling is orders of magnitude faster than Python loops |
| scipy | >= 1.14 | Sparse matrices (CSR), strong connectivity (csgraph), matrix power for path verification | Industry standard for sparse graph algorithms; csgraph.connected_components with connection='strong' |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pickle + gzip | stdlib | Graph cache serialization | Caching generated graphs to disk with compression |
| logging | stdlib | Generation diagnostics | Retry attempts, validation failures, timing |
| pathlib | stdlib | Cache directory management | Cache path construction from config hash |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.sparse CSR | networkx DiGraph | networkx is ~10x slower for generation at n=500, adds heavy dependency, but has richer algorithms. Not needed here since we only need connectivity and path counting. |
| scipy.sparse CSR | Dense numpy array | Dense is simpler but wastes memory. At n=500, dense is 2MB (fine), but at n=8000+ (if scaled), sparse wins. Sparse also makes adjacency iteration faster. |
| Direct Bernoulli | Erdos-Renyi then reject | Bernoulli per-edge is clean and matches the DCSBM model exactly. ER+reject doesn't apply to block models. |

**Installation:**
```bash
pip install scipy>=1.14
```
(numpy already in pyproject.toml; scipy is the only new dependency)

## Architecture Patterns

### Recommended Project Structure
```
src/
├── config/              # [Phase 1 - exists]
├── reproducibility/     # [Phase 1 - exists]
├── results/             # [Phase 1 - exists]
└── graph/               # [Phase 2 - NEW]
    ├── __init__.py      # Public API re-exports
    ├── dcsbm.py         # DCSBM generator: generate_dcsbm_graph()
    ├── degree_correction.py  # Theta sampling and normalization
    ├── jumpers.py       # Block jumper designation and non-triviality
    ├── validation.py    # Connectivity, density, degree checks
    └── cache.py         # Graph caching by config hash
```

### Pattern 1: DCSBM Edge Sampling (Karrer & Newman 2011)
**What:** For a directed DCSBM with K blocks, sample each potential edge (i,j) independently with probability:

    P(i -> j) = theta_i * theta_j * omega[b_i, b_j]

where `omega[b_i, b_j]` is `p_in` if `b_i == b_j`, else `p_out`, and `theta_i` are degree correction parameters.

**When to use:** Every graph generation call.
**Implementation strategy:**
1. Assign vertices to blocks: vertex i belongs to block `i // (n // K)`
2. Sample theta_i from power-law (Zipf with alpha=1.0), normalize
3. Build the full probability matrix P[i,j] = theta_i * theta_j * omega[block[i], block[j]]
4. Zero the diagonal (no self-loops)
5. Sample edges: `adjacency = (numpy.random.random((n, n)) < P).astype(np.int8)`
6. Convert to scipy.sparse.csr_matrix

**Key normalization:** theta values must be normalized so that:
    E[degree_i] = theta_i * sum_j(theta_j * omega[b_i, b_j])
The total expected edges should match the uncorrected SBM expectation. This means:
    For each block pair (a, b): sum over i in a, j in b of theta_i * theta_j * omega[a,b]
    should equal |block_a| * |block_b| * omega[a,b]
    which requires: (sum of theta in block a) * (sum of theta in block b) = |block_a| * |block_b|
    So normalize theta within each block to sum to block_size.

```python
# Theta sampling and normalization
def sample_theta(n: int, K: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Sample degree correction parameters from Zipf distribution."""
    block_size = n // K
    theta = np.zeros(n)
    for b in range(K):
        start = b * block_size
        end = start + block_size
        # Zipf: theta_i proportional to 1/rank^alpha
        ranks = np.arange(1, block_size + 1)
        raw = 1.0 / (ranks ** alpha)
        rng.shuffle(raw)  # Randomize which vertex gets which rank
        # Normalize so sum = block_size (preserves expected total degree)
        theta[start:end] = raw * (block_size / raw.sum())
    return theta
```

### Pattern 2: Block Jumper Designation with Variable r
**What:** Designate a fraction of vertices in each block as jumpers, each with a specific r value and target block. The key CONTEXT.md decision: each graph contains jumpers at ALL r values from {0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w}.

**Implementation strategy:**
1. Compute total jumpers per block: `max(1, floor(jumper_fraction * block_size))`
2. Compute r values: `[round(scale * w) for scale in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]]`
3. Distribute r values across jumpers uniformly (cycle through r values if more jumpers than r values)
4. For each jumper, assign a random target block != own block
5. Return a list of JumperInfo dataclasses: (vertex_id, source_block, target_block, r)

```python
@dataclass(frozen=True, slots=True)
class JumperInfo:
    vertex_id: int
    source_block: int
    target_block: int
    r: int  # jump length (steps to reach target block)
```

### Pattern 3: Validation Gates with Retry
**What:** After generating a graph, validate it meets all constraints. On failure, retry with incremented seed.

**Checks in order (cheapest first):**
1. **No self-loops:** Diagonal is zero (enforced during generation, but verify)
2. **Strong connectivity:** `scipy.sparse.csgraph.connected_components(adj, directed=True, connection='strong')` returns n_components == 1
3. **Minimum expected degree:** For each vertex, expected degree = sum of probabilities in its row >= 3. Note: this is the EXPECTED degree from the probability matrix, not the realized degree.
4. **Edge density:** For each block pair (a,b), compute observed density = edges(a,b) / (|a| * |b|). Check |observed - expected| < 2 * sigma where sigma = sqrt(expected * (1-expected) / (|a|*|b|)) for the average edge probability in that block pair.

**Retry logic:**
```python
for attempt in range(max_retries):
    rng = np.random.default_rng(seed + attempt)
    graph = generate_raw_graph(config, rng)
    errors = validate_graph(graph, config)
    if not errors:
        return graph
raise GraphGenerationError(f"Failed after {max_retries} attempts: {errors}")
```

### Pattern 4: Non-Triviality Verification (GRPH-04)
**What:** For each jumper vertex v in block b with target block t and jump length r, verify:
1. There EXISTS at least one path of length r from v that ends in block t
2. There EXISTS at least one path of length r from v that does NOT end in block t

This means block t is reachable in r steps but is not the ONLY possible destination at distance r.

**Implementation:** Use sparse matrix power. Let A be the adjacency matrix. A^r[v, :] gives the number of paths of length r from v to each vertex. Check:
- `any(A^r[v, vertices_in_block_t] > 0)` -- paths to target exist
- `any(A^r[v, vertices_not_in_block_t] > 0)` -- paths to non-target exist

**Efficiency concern:** Computing A^r for the full matrix is expensive for large r. For n=500 and r up to 128 (2.0 * 64), we need matrix power up to 128. However, we only need row v of A^r, so we can compute it iteratively:
```python
def paths_from_vertex(adj_csr, vertex, r):
    """Compute number of paths of length r from a single vertex."""
    # Start with indicator vector for vertex
    vec = scipy.sparse.csr_matrix(([1], ([0], [vertex])), shape=(1, adj_csr.shape[0]))
    for _ in range(r):
        vec = vec @ adj_csr
        # Clip to binary to avoid overflow (we only need reachability)
        vec.data[:] = np.minimum(vec.data, 1)
    return vec.toarray().ravel()
```

This computes reachability in r steps from a single vertex in O(r * nnz) time where nnz is the number of edges. For n=500 with density ~0.08 (weighted average of p_in and p_out), nnz ~ 20,000, so each vertex check is ~r * 20K = ~2.5M operations for r=128. With ~8-40 jumpers per graph, total is manageable.

**Important:** Clip to binary reachability (0/1) during iteration to prevent integer overflow. We only need to know IF paths exist, not HOW MANY.

### Pattern 5: Graph Caching
**What:** Cache generated graphs to disk using the graph_config_hash from Phase 1. Two configs differing only in seed should share the same graph cache (since graph_config_hash excludes seed).

**Wait -- correction:** The graph depends on the random seed used for generation. Two configs with different seeds produce different graphs even with the same graph parameters. The graph_config_hash from Phase 1 hashes only GraphConfig (n, K, p_in, p_out, n_jumpers_per_block). For caching to work correctly, we need to include the master seed in the cache key.

**Revised approach:** Cache key = `graph_config_hash(config)` + `_s{seed}`. This way:
- Same graph params + same seed = cache hit (exact same graph)
- Same graph params + different seed = cache miss (different graph needed)
- The graph_config_hash part enables grouping/identification of graph families

Actually, re-reading GRPH-05: "caches generated graphs by config hash to avoid redundant regeneration across sweep configs sharing the same graph parameters." This implies that sweep configs that share the SAME graph parameters (including the same seed from the 3-seed replication) should share cached graphs. The graph_config_hash already captures graph params; we just need to also include seed since the same graph params with different seeds produce different random graphs.

**Cache structure:**
```
.cache/
└── graphs/
    └── {graph_config_hash}_{seed}/
        ├── adjacency.npz       # scipy sparse matrix
        ├── metadata.json       # block assignments, generation params, attempt count
        └── jumpers.json        # jumper designations
```

### Anti-Patterns to Avoid
- **Generating as dense then converting to sparse:** Build the probability matrix dense (unavoidable for Bernoulli sampling at n=500), but convert to sparse immediately after sampling. Do NOT store or pass around the dense adjacency.
- **Using networkx for generation:** Adds a heavy dependency for a simple sampling operation. numpy + scipy is sufficient and faster.
- **Computing full A^r matrix for non-triviality:** Only need specific rows. Use vector-matrix multiplication iteratively.
- **Floating-point theta without normalization check:** After normalization, assert that per-block theta sums are within 1e-10 of block_size.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Strong connectivity | BFS/DFS from scratch | `scipy.sparse.csgraph.connected_components(directed=True, connection='strong')` | Tarjan's algorithm is subtle; scipy's implementation is C-optimized |
| Sparse matrix operations | Custom adjacency list | `scipy.sparse.csr_matrix` | CSR format has optimized matrix-vector products needed for path computation |
| Power-law sampling | Manual inverse CDF | Direct rank-based computation `1/rank^alpha` | Zipf is deterministic given ranks; only randomization is which vertex gets which rank |
| Config hashing | New hash function | `graph_config_hash()` from Phase 1 | Already tested and canonical |

**Key insight:** The mathematical model is well-defined (Karrer & Newman 2011). The implementation complexity is in validation and retry logic, not in the generation itself.

## Common Pitfalls

### Pitfall 1: Degree Correction Normalization Drift
**What goes wrong:** Theta values are sampled but not properly normalized, causing the actual edge density to deviate significantly from p_in/p_out.
**Why it happens:** The normalization must ensure that per-block-pair expected edge counts match the uncorrected SBM. If theta is normalized globally instead of per-block, blocks with different theta distributions will have different expected densities.
**How to avoid:** Normalize theta per-block so each block's theta values sum to block_size. Verify with assertion after normalization.
**Warning signs:** Edge density validation failing consistently; density always too high or too low.

### Pitfall 2: Strong Connectivity Failures at Low p_out
**What goes wrong:** With p_out=0.03 and n=500/K=4 (125 per block), the expected number of out-edges between blocks is ~0.03 * 125 * 125 * theta_product ~ 469 per block pair (before degree correction). This is generally sufficient, but with degree correction, low-theta vertices may have very few inter-block edges, creating bottlenecks.
**Why it happens:** Power-law degree correction concentrates edges on high-theta vertices, potentially leaving low-theta vertices disconnected.
**How to avoid:** The retry mechanism handles this. With 10 retries and seed incrementing, strongly connected graphs should be found within a few attempts for the anchor config.
**Warning signs:** Retry count consistently at maximum; indicates p_out may be too low for given n/K.

### Pitfall 3: Non-Triviality Path Counting Overflow
**What goes wrong:** When computing A^r using matrix multiplication, path counts grow exponentially. For r=128 and average degree ~40, path counts reach ~40^128, far exceeding int64 range.
**Why it happens:** Matrix power counts ALL paths, not just simple paths. Path counts explode combinatorially.
**How to avoid:** Clip to binary reachability (0/1) at each multiplication step. We only need to know IF paths exist, not how many. Use `vec.data[:] = np.minimum(vec.data, 1)` after each step.
**Warning signs:** Integer overflow warnings; NaN or Inf in path count arrays.

### Pitfall 4: Block Assignment Off-by-One with Uneven Division
**What goes wrong:** When n is not evenly divisible by K, the last block gets fewer or more vertices.
**Why it happens:** Integer division `n // K` truncates.
**How to avoid:** CONTEXT.md locks equal-sized blocks. Assert `n % K == 0` in validation and raise a clear error if not satisfied. This is a precondition, not something to handle gracefully.
**Warning signs:** Block size arrays with unequal values.

### Pitfall 5: Variable r Exceeding Walk Length
**What goes wrong:** With r_scale=2.0 and w=64, r=128. If walk_length=128 (2*w), there's exactly one step where a jumper rule could be checked (at position 0 with check at position 128). For walk_length=256 (4*w), there are 128 valid positions.
**Why it happens:** Large r values relative to walk length reduce the number of positions where rules can be evaluated.
**How to avoid:** This is a downstream concern (Phase 3/6), but the graph generator should store the r values and let downstream consumers decide which are usable given their walk_length. Document this boundary clearly.
**Warning signs:** Very few jumper events in evaluation data for large r values.

### Pitfall 6: Cache Invalidation on Config Change
**What goes wrong:** Changing n_jumpers_per_block in GraphConfig changes the graph_config_hash, but existing cached graphs with the old jumper count are stale.
**Why it happens:** The graph_config_hash includes all GraphConfig fields, so any change invalidates the cache. But jumper designation is separate from edge generation.
**How to avoid:** Consider separating the cache into two layers: (1) adjacency matrix cached by edge-generation params only (n, K, p_in, p_out, alpha, seed), and (2) jumper designation cached by full graph config. This allows re-using adjacency matrices when only jumper params change. However, this adds complexity. For v1, cache the full result by graph_config_hash. Optimization can come later.
**Warning signs:** Cache misses when only jumper parameters changed.

## Code Examples

### DCSBM Edge Probability Matrix
```python
def build_probability_matrix(
    n: int, K: int, p_in: float, p_out: float, theta: np.ndarray
) -> np.ndarray:
    """Build the DCSBM edge probability matrix.

    P[i,j] = theta[i] * theta[j] * omega[block[i], block[j]]
    where omega is p_in for same-block, p_out for different-block.
    """
    block_size = n // K
    blocks = np.arange(n) // block_size  # block assignment

    # Build omega matrix (K x K)
    omega = np.full((K, K), p_out)
    np.fill_diagonal(omega, p_in)

    # Expand to n x n
    block_probs = omega[blocks][:, blocks]  # (n, n)

    # Apply degree correction
    P = np.outer(theta, theta) * block_probs

    # Clip to [0, 1] (degree correction can push above 1)
    np.clip(P, 0.0, 1.0, out=P)

    # No self-loops
    np.fill_diagonal(P, 0.0)

    return P
```

### Sparse Adjacency Sampling
```python
def sample_adjacency(P: np.ndarray, rng: np.random.Generator) -> scipy.sparse.csr_matrix:
    """Sample a directed adjacency matrix from probability matrix P."""
    n = P.shape[0]
    uniform = rng.random((n, n))
    edges = (uniform < P).astype(np.float64)
    np.fill_diagonal(edges, 0)  # redundant safety
    return scipy.sparse.csr_matrix(edges)
```

### Strong Connectivity Check
```python
from scipy.sparse.csgraph import connected_components

def check_strong_connectivity(adj: scipy.sparse.csr_matrix) -> bool:
    """Check if directed graph is strongly connected."""
    n_components, _ = connected_components(adj, directed=True, connection='strong')
    return n_components == 1
```

### Non-Triviality Check for Single Jumper
```python
def check_non_trivial(
    adj: scipy.sparse.csr_matrix,
    vertex: int,
    target_block_vertices: np.ndarray,
    r: int,
) -> bool:
    """Check that paths of length r from vertex reach BOTH target and non-target blocks."""
    n = adj.shape[0]
    all_vertices = set(range(n))
    target_set = set(target_block_vertices)
    non_target_set = all_vertices - target_set

    # Compute reachability at distance r
    vec = scipy.sparse.csr_matrix(([1.0], ([0], [vertex])), shape=(1, n))
    for _ in range(r):
        vec = vec @ adj
        # Clip to binary reachability
        vec.data[:] = np.minimum(vec.data, 1.0)

    reachable = set(vec.nonzero()[1])

    reaches_target = bool(reachable & target_set)
    reaches_non_target = bool(reachable & non_target_set)

    return reaches_target and reaches_non_target
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| networkx SBM generator | Custom numpy/scipy generation | Current best practice for research code | networkx SBM generators lack degree correction; custom is required |
| Dense adjacency throughout | Sparse CSR for operations | scipy 1.0+ | Memory efficient, fast matrix-vector products for path computation |
| numpy.random legacy API | numpy.random.Generator (PCG64) | numpy 2.0 | Better statistical properties, explicit RNG state, compatible with seed management from Phase 1 |

**Deprecated/outdated:**
- `numpy.random.randint` / `numpy.random.random`: Use `rng = np.random.default_rng(seed)` with Generator API
- `scipy.sparse.csgraph` in older scipy versions: Current versions (1.14+) are stable and well-tested

## Open Questions

1. **Jumper fraction in GraphConfig**
   - What we know: CONTEXT.md specifies jumper_fraction swept at {1%, 2%, 5%}, but the current GraphConfig has `n_jumpers_per_block: int = 2`. The fraction-based approach is the sweep parameter; the absolute count is the anchor default.
   - What's unclear: Should GraphConfig store fraction or absolute count? Current anchor: 2 jumpers per block = 2/125 = 1.6%, close to the 1% and 2% sweep values.
   - Recommendation: Keep `n_jumpers_per_block` as the config field (absolute count computed from fraction during sweep). The existing field works. Add a utility function `jumpers_from_fraction(block_size, fraction)` that computes `max(1, floor(fraction * block_size))`.

2. **Variable r and TrainingConfig.r**
   - What we know: CONTEXT.md says each graph has jumpers at ALL r values. But TrainingConfig has a single `r: int = 57`.
   - What's unclear: The TrainingConfig.r field may need to become optional or represent a different concept (e.g., the primary r for evaluation focus).
   - Recommendation: During Phase 2, the graph generator computes all r values from `w` (the context window). The r values are `[round(scale * w) for scale in R_SCALES]` where `R_SCALES = (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0)`. The generator reads `w` from `config.training.w`. TrainingConfig.r is left as-is for Phase 2; it may be repurposed or removed in later phases.

3. **Degree correction alpha parameter storage**
   - What we know: CONTEXT.md locks alpha=1.0, not swept.
   - What's unclear: Should alpha be stored in GraphConfig for explicitness?
   - Recommendation: Add `degree_correction_alpha: float = 1.0` to GraphConfig for documentation and future flexibility, but do NOT sweep it. This is a minor config addition.

## Sources

### Primary (HIGH confidence)
- Karrer, B. & Newman, M.E.J. (2011) "Stochastic blockmodels with a growing number of classes" - defines the DCSBM model
- scipy.sparse.csgraph documentation - connected_components with directed=True, connection='strong' uses Tarjan's algorithm
- numpy.random.Generator documentation - PCG64 default, reproducible seeding

### Secondary (MEDIUM confidence)
- scipy.sparse.csr_matrix documentation - CSR format for efficient row slicing and matrix-vector products
- Python pickle + gzip for serialization - standard pattern for numpy/scipy object caching

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - numpy and scipy are the canonical tools for this exact problem
- Architecture: HIGH - DCSBM generation is well-defined mathematically; implementation patterns are straightforward
- Pitfalls: HIGH - degree correction normalization and path counting overflow are well-known issues in the SBM literature

**Research date:** 2026-02-24
**Valid until:** 2026-06-24 (stable domain, no fast-moving dependencies)
