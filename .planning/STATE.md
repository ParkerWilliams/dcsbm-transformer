# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.
**Current focus:** Milestone v1.1 Journal Feedback — defining requirements

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-02-26 — Milestone v1.1 Journal Feedback started

## Performance Metrics

**Velocity (v1.0):**
- Total plans completed: 20
- Average duration: 3.7 min
- Total execution time: ~1.25 hours

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
- [06-01]: EPS=1e-12 for all SVD metric denominator guards, CONDITION_CAP=1e6
- [06-01]: Grassmannian distance default k=2 subspace dimension
- [06-01]: compute_all_metrics conditionally includes spectral gaps based on S.shape[-1]
- [06-02]: Active jumper constraints tracked as (deadline_step, target_block) tuples
- [06-02]: failure_index records step index t where rule_outcome[t]==VIOLATED
- [06-03]: AVWo = (A @ V) @ W_o.weight.T matching actual residual stream contribution
- [06-03]: read_write_alignment only for square matrices (WvWo); skipped for QK^T, AVWo
- [06-03]: WvWo SVD computed once and broadcast (static weight matrices)
- [06-03]: NaN for WvWo Grassmannian distance (static, no step-to-step change)
- [07-01]: resolution_step = encounter_step + r; rule_outcome at resolution_step - 1
- [07-01]: Only violations contaminate subsequent encounters (FOLLOWED does not)
- [07-01]: is_first_violation matched via failure_index[walk] == outcome_idx
- [07-01]: PRIMARY_METRICS frozenset of 5 pre-registered metrics (target.metric_name pattern)
- [07-01]: Event count tiers: skip (0-1), low_n (2-4), moderate_n (5-9), full (10+)
- [07-01]: Shuffle control uses max AUROC across lookback distances
- [07-02]: BCa bootstrap with automatic fallback to percentile for degenerate AUROC distributions
- [07-02]: Holm-Bonferroni applies to exactly 5 primary metrics (correction factor at most 5, not 21)
- [07-02]: Cohen's d returns NaN for pooled_std < 1e-12 or insufficient samples (< 2 per group)
- [07-02]: Measurement correlation uses resolution_step - 1 as representative event position
- [07-02]: Predictive correlation replaces NaN AUROC with 0.5 for correlation computation
- [07-02]: Headline comparison identifies QK^T/AVWo by key prefix, filters to primary metrics
- [08-01]: Colorblind-safe palette from seaborn 'colorblind' preset (8 colors)
- [08-01]: Matplotlib Agg backend for headless rendering
- [08-01]: SVG fonttype='none' for portable text embedding
- [08-02]: Box plots as default distribution type
- [08-02]: YlOrRd colormap for horizon heatmap
- [08-02]: render_all wraps each plot in try/except for fault isolation
- [09-01]: Figure categorization by filename prefix for automatic template slot assignment
- [09-01]: Config tables structured as Model/Training/Data categories with N/A defaults
- [09-01]: Statistical tests extracted from metrics.statistical_controls.headline_comparison.primary_metrics
- [09-03]: Jinja2 LaTeX-safe delimiters (BLOCK/VAR) prevent brace conflicts with LaTeX
- [09-03]: Two-pass pdflatex with graceful fallback to .tex on FileNotFoundError or timeout
- [09-03]: Symbols defined in context within each section's math block, no separate glossary
- [09-02]: Renamed metric row key to experiment_values to avoid Jinja2 dict.values() method collision
- [09-02]: Sparkline implemented as horizontal bar chart using matplotlib with project colorblind-safe PALETTE
- [09-02]: Verdict uses higher-is-better default, with loss/error keywords triggering lower-is-better comparison
- [09-02]: Curve overlay grid uses side-by-side subplots with imshow of original figure PNGs

### Pending Todos

- v1.0 integration gaps deferred: run_experiment.py stub, set_seed not called in production, predictive_horizon not written to result.json
- v1.0 Phase 10 (Sweep Infrastructure) not implemented: MGMT-02, MGMT-03, MGMT-04, MGMT-06

### Blockers/Concerns

- Phase 3 research flag: SVD memory footprint for w=256 needs profiling on anchor config before sweep planning
- pylatex stability on RunPod needs verification before Phase 9 math PDF work
- Multi-head ablation requires relaxing the single-head constraint in TransformerLM

## Session Continuity

Last session: 2026-02-26
Stopped at: Starting milestone v1.1 Journal Feedback
Resume file: None
Next action: Define v1.1 requirements
