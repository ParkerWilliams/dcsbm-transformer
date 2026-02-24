# Phase 2: DCSBM Graph Generation - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate valid degree-corrected stochastic block model (DCSBM) directed graphs with block jumper rules, validation gates, and caching. Graphs serve as the foundation for training data — walks (Phase 3) and model training (Phase 5) consume these graphs. All design decisions aim to mimic the structure of real token adjacency graphs.

</domain>

<decisions>
## Implementation Decisions

### Degree Correction Model
- Power-law distribution for θ_i parameters (Zipf's law), mimicking token frequency distribution
- Fixed exponent α=1.0 (classic Zipf), not swept — keeps one fewer dimension in the parameter grid
- Normalize θ_i values so expected total degree matches the uncorrected SBM expectation (p_in/p_out control density, θ_i only reshapes distribution)
- Single θ_i per vertex for both in-degree and out-degree (no separate θ_in/θ_out)

### Block Jumper Designation
- Jumper count per block: fraction of block size, swept at {1%, 2%, 5%}
- Floor rounding with minimum 1 jumper per block (guarantees coverage across all blocks)
- Target blocks assigned randomly per jumper (uniform from blocks ≠ own block). Different jumpers in the same block may have different targets
- **Variable r per jumper within a single graph**: each graph contains jumpers at ALL r values from the discrete set {0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w}, distributed uniformly across r values
- This replaces the config-level r sweep entirely — r analysis happens at filtering time, not config time. Massively reduces the config grid
- All r values are rounded to nearest integer after computing from w

### Block Structure
- Equal-sized blocks (n/K vertices per block) — simplest, avoids confounding jumper analysis with block size variation
- Anchor config: K=4 blocks (~125 vertices per block with n=500)
- K swept over {4, 8, 16} per spec
- Anchor density: p_in=0.25, p_out=0.03 (ratio ~8:1, clear but non-trivial block structure)
- No self-loops

### Validation & Retry Policy
- On validation failure (not strongly connected, min degree < 3, density out of tolerance): retry with incremented seed
- Maximum 10 retry attempts before hard failure with diagnostic info
- Generation attempt counter keeps config hash stable across retries
- Edge density tolerance: 2σ from expected value under the DCSBM model
- Non-triviality check (GRPH-04): if a jumper vertex fails, reassign to a different vertex in the same block (retry up to block_size/2 times before falling back to full graph regeneration)

### Claude's Discretion
- Graph data structure choice (sparse vs dense representation)
- Caching serialization format
- Edge sampling algorithm (direct Bernoulli sampling vs adjacency matrix construction)
- Internal logging verbosity during generation
- Exact implementation of the power-law θ_i sampling

</decisions>

<specifics>
## Specific Ideas

- "All decisions should be made to mimic the structure of a token adjacency graph" — this is the guiding design principle
- The variable-r-per-graph design is a deliberate departure from the original spec's r sweep. Instead of separate configs per r, each graph embeds the full r spectrum. Analysis-time filtering by r replaces the r dimension of the config sweep. This simplifies the sweep grid significantly while providing more data per r value
- Power-law degree correction at α=1.0 directly models the Zipfian token frequency distribution seen in natural language

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-dcsbm-graph-generation*
*Context gathered: 2026-02-24*
