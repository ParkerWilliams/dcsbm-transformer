# Phase 17: E2E Pipeline Wiring - Context

**Gathered:** 2026-02-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire `run_experiment.py` to chain all phases (graph generation → walks → training → evaluation → analysis → visualization → reporting) into a single executable pipeline. Close P0/P1/P2 integration gaps from v1.0 milestone audit: reproducibility seeding, predictive horizon persistence, NPZ key consistency, visualization exports, and Makefile stubs.

</domain>

<decisions>
## Implementation Decisions

### Pipeline Execution Feedback
- Stage banners by default: print a line when each stage starts/finishes (e.g., "=== Graph Generation === ... done in 2.3s")
- Add `--verbose` flag mapping to DEBUG-level logging for detailed per-stage output
- Print final summary with per-stage timings and output file paths on completion
- Enhance `--dry-run` to show full pipeline plan: all stages that would execute + expected output paths

### Error Handling
- Fail fast on first error — stop immediately, let the Python traceback propagate naturally
- Save partial results before failing — write whatever completed successfully to disk
- Use Python `logging` module: INFO level by default, DEBUG for --verbose

### Result File Organization
- Output to `results/{experiment_id}/` directory — auto-created, self-contained
- Predictive horizon analysis results written to `predictive_horizon` block in result.json (same file as training/eval results)
- Same experiment ID → same directory; re-running overwrites previous results
- Flat output directory: result.json, NPZ data, all plots (PNG/SVG), HTML report, config copy — everything in one dir
- Copy config.json into output directory for full reproducibility

### Seeding & Reproducibility
- Call `set_seed(config.seed)` once at pipeline start before any stochastic operations
- Log seed to console at startup and record in result.json metadata
- Include git hash via existing `get_git_hash()` in result.json metadata
- Trust Phase 16 dual-key emission fix for NPZ key consistency — no extra runtime assertions

### Pipeline Scope
- Full pipeline only — always run all stages, no --skip flags or stage selection
- Individual stages remain callable programmatically for testing/development

### P2 Gap Closures (also in scope)
- Fix `src/visualization/__init__.py` to export public API via `__all__` (consistency with other packages)
- Wire Makefile `pdf` target to call `generate_math_pdf`; add `make report` to run full pipeline

### Claude's Discretion
- Exact stage ordering within evaluation → analysis → visualization chain
- How to structure the pipeline function internally (one big function vs stage functions)
- Logging format strings and level assignments
- How to handle the config copy (serialize from object vs copy original file)

</decisions>

<specifics>
## Specific Ideas

- Pipeline should feel like a standard ML experiment runner — `python run_experiment.py --config config.json` and walk away
- Stage banners should include elapsed time for each stage
- The `results/{experiment_id}/` directory should be a complete reproducible experiment record

</specifics>

<deferred>
## Deferred Ideas

- Stage-level skip flags (`--skip-viz`, `--stages train,eval`) — future enhancement if needed
- Versioned output directories with timestamps — not needed given experiment ID collision model

</deferred>

---

*Phase: 17-e2e-pipeline-wiring*
*Context gathered: 2026-02-27*
