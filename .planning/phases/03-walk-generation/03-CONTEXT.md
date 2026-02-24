# Phase 3: Walk Generation - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate directed random walk corpora on DCSBM graphs with complete jumper-event metadata, producing 100% rule-compliant training data and separate evaluation data. The walks serve as transformer training and evaluation sequences — the core research question is whether a single-head transformer can learn contextual rules (edge structure and jumper rules) from this data, and what its limitations are. Walk length is a sweep parameter (2w, 4w, 8w per config).

</domain>

<decisions>
## Implementation Decisions

### Walk Sampling Strategy
- Starting vertices chosen uniformly at random
- Transition probabilities uniform over outgoing neighbors (degree correction is in graph structure, not walk process)
- Vertex revisits allowed freely (standard random walk)
- Dead-end handling: restart walk from new random vertex (should not occur given Phase 2 strong connectivity guarantee)

### Training Corpus Rule Compliance (CRITICAL)
- **Training walks must be 100% rule-compliant** — every jumper rule satisfied in every walk
- Walks are guided at jumper encounters: when a walk hits a jumper vertex, the next r_j steps follow a pre-computed compliant path of length r_j to a node with an edge to the target block
- Compliant path chosen **uniformly** among all valid length-r paths (not weighted) — forces the transformer to learn the destination constraint, not memorize specific routes
- **Nested jumpers must both be satisfied** — if a second jumper is encountered during a guided path, both rules must be fulfilled simultaneously
- After any filtering/discards, corpus is regenerated to maintain size requirement

### Jumper Representation
- **50% minimum** — at least half the walks must contain a jumper encounter
- Achieve this by seeding 50%+ of walks directly from jumper vertices
- **Path diversity validation** — each jumper must have at least 3 distinct compliant paths represented in the final corpus (prevents single-route memorization)

### Train/Eval Corpus Design
- **Independent generation** with different seeds — no overlap by construction
- **90/10 split** — 90% training, 10% evaluation
- Training corpus must be >= 100n tokens (the 100n threshold applies to train alone, eval is additional)
- Walk length is a **sweep parameter** — one length per experiment config (2w, 4w, or 8w)
- Eval walks test whether the transformer learned edge structure and jumper rules

### Jumper Event Metadata
- **Per-walk event list** — each walk stores: [{vertex_id, step, target_block, expected_arrival_step}]
- Records encounter only (outcome evaluated during transformer inference in Phase 6)
- Stored inside the same .npz file as walks (atomic — walks and metadata always in sync)

### Walk Storage and Caching
- **NumPy .npz archive** — walks as int32 array (num_walks × walk_length), metadata as structured arrays, all in one file
- Cache key = hash(graph_config_hash + walk_length + corpus_size + split + seed) — graph changes auto-invalidate walks
- Cached files stored in same cache directory as graph .npz files (single cache location for all artifacts)

### Claude's Discretion
- Exact implementation of nested jumper constraint satisfaction (solver approach)
- Pre-computation strategy for enumerating compliant paths
- Batch size and parallelism during walk generation
- Corpus statistics logging format

</decisions>

<specifics>
## Specific Ideas

- "The training data corpus should contain only rule-following walks. This is the entire point — to see if a transformer can learn this contextual rule."
- The distributional shift from guided walks at jumpers is intentional — the rule IS the signal the transformer must learn
- Jumper validation (multiple paths of length r_j exist) is already handled by Phase 2 (GRPH-04); walk generation can assume validated jumpers

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-walk-generation*
*Context gathered: 2026-02-24*
