# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-24)

**Core value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.
**Current focus:** Phase 5 - Training Pipeline (complete)

## Current Position

Phase: 5 of 10 (Training Pipeline) -- COMPLETE
Plan: 2 of 2 in current phase (all complete)
Status: Phase 5 execution complete, verified
Last activity: 2026-02-25 -- Completed 05-02: Sufficiency gate and training pipeline

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 10
- Average duration: 3.7 min
- Total execution time: ~0.62 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 2/2 | 9 min | 4.5 min |
| 2 | 3/3 | 8 min | 2.7 min |
| 3 | 2/2 | 9 min | 4.5 min |
| 4 | 1/1 | 4 min | 4.0 min |
| 5 | 2/2 | 7 min | 3.5 min |

**Recent Trend:**
- Last 5 plans: 03-01 (4 min), 03-02 (5 min), 04-01 (4 min), 05-01 (2 min), 05-02 (5 min)
- Trend: Stable

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [02-03]: Cache key = graph_config_hash + _s{seed} to enable per-seed caching
- [02-03]: Jumper seed offset by +1000 from graph seed to avoid correlation
- [02-03]: Convert numpy int64 to Python int for JSON serialization compatibility
- [02-02]: Global r-value cycling across all blocks (not per-block) ensures all 8 r-scales are represented
- [02-02]: Binary clipping at each iteration step prevents integer overflow in path counting
- [02-02]: Reassignment tries alternative vertices in same block before skipping
- [02-01]: Zipf alpha=1.0 with per-block normalization so theta sums to block_size
- [02-01]: Retry up to 10 times on validation failure (disconnected graph, etc.)
- [01-02]: Seeds set in strict order: random -> numpy -> torch -> cuda -> cudnn -> deterministic_algorithms -> cublas
- [01-02]: Git hash dirty detection checks both staged and unstaged changes
- [01-01]: Used dacite strict=True for config deserialization to catch schema drift early
- [01-01]: SweepConfig structure defined but execution deferred to Phase 10
- [Phase 03]: Convert numpy int types to Python int in JumperEvent for isinstance compatibility — numpy int32 is not recognized as Python int by isinstance checks
- [Phase 03]: Train seed offset +2000, eval seed offset +3000 from config.seed — Avoids correlation with graph seed and jumper seed (+1000)
- [Phase 03]: Events stored as parallel arrays in NPZ for efficient serialization — Flat arrays with walk_id grouping enables O(n) reconstruction
- [04-01]: Dual masking convention: zero-fill for SVD QK^T target, -inf for softmax attention
- [04-01]: Separate W_q/W_k/W_v/W_o linear layers (not fused) for extraction clarity
- [04-01]: Residual stream includes pre-block embedding state as index 0 (n_layers+1 total states)
- [04-01]: WvWo computed as W_v.weight.T @ W_o.weight per layer (nn.Linear convention)
- [05-01]: WalkDataset chunks walks into non-overlapping subsequences of size w+1 (context+target)
- [05-01]: Cosine schedule with 10% linear warmup, min_lr_ratio=0.1
- [05-01]: Gradient clipping max_norm=1.0, AdamW weight_decay=0.01
- [05-02]: Edge compliance checks CSR adjacency via indptr/indices lookup
- [05-02]: Rule compliance uses jumper_map dict for O(1) jumper vertex lookup
- [05-02]: Gate thresholds as module constants: EDGE_COMPLIANCE_THRESHOLD=0.95, RULE_COMPLIANCE_THRESHOLD=0.80
- [05-02]: Self-loops included in complete graph fixtures for test correctness
- [05-02]: Pipeline generates experiment_id for checkpoint directory naming

### Pending Todos

None yet.

### Blockers/Concerns

- Phase 6 is the largest phase (12 requirements) due to fused evaluation constraint; may need careful plan decomposition
- Phase 3 research flag: SVD memory footprint for w=256 needs profiling on anchor config before sweep planning
- pylatex stability on RunPod needs verification before Phase 9 math PDF work

## Session Continuity

Last session: 2026-02-25
Stopped at: Completed Phase 5 (05-01-PLAN.md and 05-02-PLAN.md)
Resume file: None
