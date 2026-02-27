---
plan: 14-02
phase: 14-softmax-filtering-bound
status: complete
started: 2026-02-26
completed: 2026-02-26
duration: ~10min
---

# Plan 14-02: Empirical Bound Verification and Visualization

## What was built
- Perturbation experiment module (`src/analysis/perturbation_bound.py`) with 7 functions:
  - `compute_theoretical_bound`: Computes epsilon-bound from Theorem 6.1
  - `inject_perturbation`: Injects controlled perturbation into scaled QK^T and recomputes AVWo
  - `compute_spectral_change`: Measures L2 norm of singular value difference vector
  - `generate_adversarial_direction`: Top singular vector outer product of QK^T (unit Frobenius norm)
  - `generate_random_direction`: Random unit-norm direction with causal masking
  - `run_perturbation_at_step`: Single-step perturbation experiment across magnitudes
  - `run_perturbation_experiment`: Top-level orchestrator aggregating across steps
- Bound tightness visualization (`src/visualization/perturbation_bound.py`):
  - `plot_bound_tightness`: Scatter + envelope with theoretical bound line at ratio=1.0
  - `plot_bound_by_magnitude`: Per-magnitude bar charts showing adversarial vs random ratios
- Report integration:
  - `render.py`: Added perturbation bound figure generation
  - `single.py`: Added perturbation_bound_figures collection and template variables
  - `single_report.html`: Added collapsible "Softmax Filtering Bound" section with verification summary table
  - `schema.py`: Added backward-compatible perturbation_bound validation block
- Test suite (`tests/test_perturbation_bound.py`) with 12 tests across 5 test classes:
  - TestComputeTheoreticalBound: known values, zero epsilon, linear scaling
  - TestInjectPerturbation: zero perturbation identity, nonzero produces change
  - TestAdversarialDirection: unit Frobenius norm, upper triangle zero
  - TestRandomDirection: unit norm, upper triangle zero, seed reproducibility
  - TestBoundHolds: bound holds on small model, adversarial larger than random
  - TestRunPerturbationExperiment: output structure validation, bound_verified flag

## Key files
- `src/analysis/perturbation_bound.py` - Core perturbation experiment module
- `src/visualization/perturbation_bound.py` - Bound tightness plots
- `src/visualization/render.py` - Added perturbation bound render section
- `src/reporting/single.py` - Added figure collection and template variables
- `src/reporting/templates/single_report.html` - Added Softmax Filtering Bound section
- `src/results/schema.py` - Added perturbation_bound validation
- `tests/test_perturbation_bound.py` - 12 tests

## Key design decisions
- Works with SCALED QK^T (already divided by sqrt(d_k) in attention.py), so the bound simplifies to epsilon * ||qkt_scaled||_F * ||V||_2 * ||W_O||_2 / 2 (the sqrt(d_k) factors cancel)
- Random direction generator always runs on CPU then moves to device, avoiding CUDA generator compatibility issues
- Success criterion: fewer than 5% of perturbations exceed theoretical bound (bound_verified flag)
- Tightness ratio: median of all empirical/theoretical ratios across magnitudes and directions

## Deviations
- Schema validation and HTML template sections were already added in the previous session during Plan 14-01 execution (similar to Phase 13 pattern of editing once). No duplicate work needed.
- Fixed CUDA device compatibility for torch.Generator in random direction generation (always generate on CPU, then move to target device).

## Self-Check: PASSED
- [x] inject_perturbation correctly injects perturbation into QK^T and recomputes AVWo
- [x] compute_theoretical_bound returns correct value using scaled-QK^T simplified formula
- [x] run_perturbation_experiment generates both random and adversarial perturbations with per-magnitude summary
- [x] Bound tightness visualization shows theoretical envelope vs empirical with tightness ratio annotated
- [x] result.json perturbation_bound block validates through schema.py
- [x] Collapsible HTML report section renders verification summary table
- [x] Tests cover all core functions and integration points
