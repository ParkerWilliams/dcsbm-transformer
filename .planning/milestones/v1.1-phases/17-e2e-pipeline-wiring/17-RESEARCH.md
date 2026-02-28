# Phase 17: E2E Pipeline Wiring - Research

**Completed:** 2026-02-27
**Status:** RESEARCH COMPLETE

## Codebase Inventory

### Existing Pipeline Stages (all implemented, none wired)

| Stage | Module | Entry Function | Returns |
|-------|--------|---------------|---------|
| 1. Graph Generation | `src/graph/cache.py` | `generate_or_load_graph(config)` | `(GraphData, list[JumperInfo])` |
| 2. Walk Generation | `src/walk/cache.py` | `generate_or_load_walks(config, graph_data, jumpers)` | `(train_walks, eval_walks, train_events, eval_events)` |
| 3. Model Creation | `src/model/transformer.py` | `create_model(config)` | `TransformerLM` |
| 4. Training | `src/training/pipeline.py` | `run_training_pipeline(model, train_walks, eval_walks, graph_data, jumpers, config, device)` | `TrainingPipelineResult` |
| 5. Evaluation | `src/evaluation/pipeline.py` | `fused_evaluate(model, eval_walks, graph_data, jumpers, config, device)` | `EvaluationResult` |
| 5b. Save Evaluation | `src/evaluation/pipeline.py` | `save_evaluation_results(result, output_dir, split_labels)` | summary dict |
| 5c. Split Assignment | `src/evaluation/split.py` | `assign_split(failure_index)` | split labels array |
| 6. AUROC Analysis | `src/analysis/auroc_horizon.py` | `run_auroc_analysis(eval_data, jumper_map, metric_keys)` | predictive_horizon dict |
| 6b. Statistical Controls | `src/analysis/statistical_controls.py` | `apply_statistical_controls(auroc_results, eval_data, metric_keys)` | statistical_controls dict |
| 7. Visualization | `src/visualization/render.py` | `render_all(result_dir)` | list of figure paths |
| 8. Reporting | `src/reporting/single.py` | `generate_single_report(result_dir)` | Path to HTML report |

### Current run_experiment.py Status

Lines 61-62: `"[stub] Experiment execution not yet implemented (Phase 2+)."`
Only loads config, prints summary, exits. Needs full pipeline.

### Integration Gap Analysis (from v1.0 Audit)

#### P0: run_experiment.py is a stub
Every phase module has well-defined functions. The pipeline orchestrator needs to call them in sequence.

#### P0: set_seed never called in production
`set_seed` defined in `src/reproducibility/seed.py`, exported in `__init__.py`. Never called from any production code path. Must be called before model creation and any stochastic operations.

#### P1: predictive_horizon never written to result.json
`run_training_pipeline` writes result.json with only `scalars` and `curves`. After evaluation, need to:
1. Run `fused_evaluate` to get `EvaluationResult`
2. Run `save_evaluation_results` to write NPZ
3. Run `run_auroc_analysis` to compute predictive horizon
4. Run `apply_statistical_controls` for stat rigor
5. Update result.json with predictive_horizon + statistical_controls blocks

#### P1: NPZ key format consistency
`save_evaluation_results` uses `target.layer_N.head_H.metric_name` keys (v1.1 per-head format).
Phase 16 added dual-key emission: single-head runs emit both `target.layer_N.metric_name` (legacy) and `target.layer_N.head_0.metric_name` (new).
`render_all` searches for keys with "." separator and SVD target prefixes — compatible with both formats.
**Conclusion:** Phase 16 dual-key emission resolves this. Trust and verify at pipeline integration time.

#### P2: visualization __init__.py exports nothing
`src/visualization/__init__.py` contains only `"""Publication-quality static figure generation."""`
Should export `render_all`, `load_result_data`, `apply_style`, `save_figure`.

#### P2: Makefile pdf target is a stub
`make pdf` prints "not yet implemented". Need to call `generate_math_pdf`.
Also add `make report` for full pipeline.

### Key Dependencies and Data Flow

```
config.json
  |
  v
ExperimentConfig
  |
  +---> set_seed(config.seed)
  |
  +---> generate_or_load_graph(config) --> GraphData, [JumperInfo]
  |                                            |
  +---> generate_or_load_walks(config, ...) -> (train, eval, events)
  |                                               |
  +---> create_model(config) ---> TransformerLM   |
  |                                  |            |
  +---> run_training_pipeline(...) --+--> TrainingPipelineResult
  |       (writes result.json)              |
  |                                         |
  +---> fused_evaluate(model, ...) -----> EvaluationResult
  |                                         |
  +---> assign_split(failure_index) ----> split_labels
  |                                         |
  +---> save_evaluation_results(...) ---> eval_summary (writes NPZ)
  |                                         |
  +---> run_auroc_analysis(...) --------> predictive_horizon dict
  |                                         |
  +---> apply_statistical_controls(...) -> statistical_controls dict
  |                                         |
  +---> UPDATE result.json with eval_summary, predictive_horizon, statistical_controls
  |                                         |
  +---> render_all(result_dir) ----------> figure files
  |                                         |
  +---> generate_single_report(result_dir) -> report.html
```

### Result Directory Structure

After pipeline completes, `results/{experiment_id}/` should contain:
- `result.json` — full experiment record with all analysis blocks
- `token_metrics.npz` — per-step SVD metrics and behavioral labels
- `spectrum_trajectories.npz` — full spectrum data (Phase 15)
- `figures/` — all PNG + SVG plots
- `report.html` — self-contained HTML report
- `config.json` — copy of input config for reproducibility
- `checkpoints/` — model checkpoints (from training)

### Testing Strategy

The E2E pipeline test should use tiny configs to keep tests fast:
- n=20 vertices, K=2 blocks, w=8, walk_length=16, corpus_size=2000
- d_model=16, n_layers=1, n_heads=1
- max_epochs=1 (just verify the wiring, not training quality)

Test should verify:
1. Pipeline runs without error
2. result.json exists and validates
3. predictive_horizon block exists in result.json
4. set_seed is called (verify via determinism check)
5. NPZ file exists with expected keys
6. Figures directory is created with files
7. HTML report exists

---

*Phase: 17-e2e-pipeline-wiring*
*Research completed: 2026-02-27*
