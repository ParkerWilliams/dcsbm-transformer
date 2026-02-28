# Phase 15: Advanced Analysis - Research

**Researched:** 2026-02-26
**Domain:** Full spectrum trajectory tracking, Frenet-Serret curvature/torsion, compliance curve sweep
**Confidence:** HIGH

## Summary

Phase 15 adds two advanced analysis capabilities: (1) full spectrum trajectory tracking that stores raw singular value vectors and computes discrete Frenet-Serret curvature/torsion as secondary AUROC metrics, and (2) a sharp compliance curve that aggregates results across r/w ratio experiments to characterize the phase transition from compliance to failure.

Both capabilities build on existing infrastructure: spectrum storage extends the evaluation pipeline's NPZ output, curvature/torsion feeds into the AUROC analysis framework, and the compliance curve aggregates existing result.json files. No changes to training or core SVD computation are required.

**Primary recommendation:** Implement as three plans -- (1) spectrum storage and curvature/torsion analysis module, (2) curvature/torsion AUROC integration, (3) compliance curve sweep and publication figure.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Full spectrum storage: top-k singular values in float16, separate NPZ file (spectrum_trajectories.npz), QK^T by default
- Curvature/torsion: Savitzky-Golay smoothing before differentiation, windowed statistics, exploratory (secondary) status
- Compliance curve: post-hoc aggregation, existing R_SCALES r values, 3 seeds, dual-axis publication figure
- NPZ keys: qkt.layer_N.spectrum with shape [n_sequences, n_steps, k]

### Claude's Discretion
- Top-k value (suggested 8)
- Savitzky-Golay window/polyorder tuning
- Torsion formula variant
- Compliance figure layout
- Ordering crossing detection threshold
</user_constraints>

## Technical Analysis

### 1. Full Spectrum Storage

**Current state:** The evaluation pipeline (`src/evaluation/pipeline.py`) computes full SVD via `torch.linalg.svd(matrix, full_matrices=False)` at every step, extracts U, S, Vh, then immediately reduces S to scalar metrics (stable_rank, spectral_entropy, etc.) and discards S. The singular values S are available in the inner loop but not stored.

**Required change:** After computing S at each step, store S[:k] (top-k singular values) in a separate array. This is purely additive -- existing metric computation is unchanged.

**Storage format:**
```
spectrum_trajectories.npz keys:
  qkt.layer_0.spectrum  -> [n_sequences, n_steps, k] float16
  qkt.layer_1.spectrum  -> [n_sequences, n_steps, k] float16
  qkt.layer_2.spectrum  -> [n_sequences, n_steps, k] float16
  qkt.layer_3.spectrum  -> [n_sequences, n_steps, k] float16
```

**Storage overhead (anchor config):**
- n_sequences=1000, n_steps=256, k=8, n_layers=4
- Per array: 1000 * 256 * 8 * 2 bytes = ~4 MB
- Total (4 layers): ~16 MB
- With k=8 and float16, this is very manageable (far less than the 125 MB estimated in ARCHITECTURE.md which assumed k=64)

**Key insight:** Using k=8 (top 8 singular values) instead of full k=64 dramatically reduces storage while preserving all information needed for curvature/torsion analysis. The top 8 SVs capture the dominant spectral structure; higher-order SVs are noise-dominated.

### 2. Curvature and Torsion Computation

**Mathematical foundation:**

Given a trajectory sigma(t) in R^k (the singular value vector at each step), the discrete Frenet-Serret curvature and torsion are:

**Curvature (kappa):**
```
v(t) = sigma(t+1) - sigma(t)             # velocity (first derivative)
a(t) = v(t+1) - v(t)                     # acceleration (second derivative)
kappa(t) = ||cross(v, a)|| / ||v||^3      # curvature
```

For k-dimensional curves (k > 3), the cross product is replaced by:
```
kappa(t) = ||a - (a . v_hat) * v_hat|| / ||v||^2
```
where v_hat = v / ||v|| is the unit velocity. This is the component of acceleration perpendicular to velocity, divided by speed squared.

**Torsion (tau):**
```
j(t) = a(t+1) - a(t)                     # jerk (third derivative)
b(t) = cross(v, a)                        # binormal direction
tau(t) = (j . b) / ||b||^2               # torsion (scalar)
```

For k > 3 dimensions, torsion requires projecting the third derivative onto the binormal direction in the osculating plane.

**Numerical stability concerns (from PITFALLS.md Pitfall 7):**
1. **Noise amplification:** Second derivatives amplify noise by 1/dt^2. With dt=1 and SVD noise ~1e-5, this is borderline.
2. **Savitzky-Golay solution:** Apply `scipy.signal.savgol_filter(spectra, window_length=7, polyorder=3, axis=0)` before differentiation. This removes high-frequency noise while preserving the curvature signal. The window of 7 steps is ~3% of the total 256 steps, small enough to preserve temporal localization.
3. **Ordering crossings:** When sigma_i and sigma_{i+1} swap, detect by checking for sign changes in (sigma_i - sigma_{i+1}) and mark those steps as invalid for curvature.
4. **Guard for zero velocity:** When ||v|| < epsilon, curvature is undefined. Set to NaN at those steps.

**Output:** Per-sequence, per-layer arrays of curvature and torsion values at each step (after smoothing), stored in the same spectrum_trajectories.npz.

### 3. Curvature/Torsion as AUROC Metrics

**Integration approach:** The existing AUROC pipeline (`src/analysis/auroc_horizon.py`) operates on metric arrays of shape [n_sequences, n_steps-1]. Curvature and torsion arrays have the same shape (minus boundary effects from derivatives). They can be fed into `compute_auroc_curve` as additional metrics.

**Key design:** Curvature/torsion are SECONDARY metrics. They are NOT added to `PRIMARY_METRICS` (which are locked by pre-registration). They appear in a separate section of the AUROC analysis output, clearly labeled as exploratory.

**Expected signal:** A curvature spike should precede rule violations -- the spectral trajectory "bends" as the model transitions from compliant to non-compliant behavior. If curvature AUROC > 0.6 at j=2..5 lookback, this supports the geometric interpretation.

### 4. Compliance Curve

**Data source:** The compliance curve requires results from multiple experiments, each trained with a different r value. Each r value corresponds to a different r_scale from R_SCALES = (0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0).

**Current infrastructure:**
- `config/experiment.py`: TrainingConfig has `r` field (jump length)
- `graph/jumpers.py`: `compute_r_values(w)` returns all discrete r values
- `evaluation/behavioral.py`: Computes compliance rates
- `analysis/auroc_horizon.py`: Computes predictive horizons
- `results/schema.py`: Stores compliance and horizon in result.json

**New module:** `src/analysis/compliance_curve.py` -- loads multiple result.json files, extracts (r/w ratio, edge compliance, rule compliance, predictive horizon) tuples, and computes aggregate statistics.

**Visualization:** `src/visualization/compliance.py` -- dual-axis figure with:
- Left y-axis: rule compliance rate (0-1), plotted as line with error bands
- Right y-axis: predictive horizon (steps), plotted as line with error bands
- X-axis: r/w ratio
- Expected shape: sigmoid-like drop in compliance near r/w=1, peak in horizon near the transition

**Note on GPU budget:** The compliance curve requires 8 r-values * 3 seeds = 24 training runs. This is the analysis and visualization infrastructure ONLY -- the actual training runs are outside this phase's scope. The compliance_curve module operates on pre-existing result.json files.

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Curvature too noisy to be useful | Low (secondary metric) | Medium | Savitzky-Golay smoothing, windowed statistics, fallback to Grassmannian curvature |
| Spectrum storage bloats NPZ | Low | Low | k=8 float16 is only ~16 MB total |
| Ordering crossings corrupt curvature | Medium | Medium | Detection and interpolation, alternative Grassmannian approach |
| Compliance curve requires many training runs | High | N/A | Module only aggregates existing results; GPU budget concern is operational |
| Torsion undefined in low dimensions | Low | Low | k=8 is sufficient for torsion; guard for degenerate cases |

## Codebase Patterns to Follow

1. **Analysis modules:** Follow `src/analysis/pr_curves.py` pattern -- standalone functions, reuse event_extraction, return nested dicts
2. **Visualization:** Follow `src/visualization/calibration.py` pattern -- import style.py, use save_figure, return plt.Figure
3. **NPZ integration:** Follow `save_evaluation_results` pattern in pipeline.py for spectrum storage
4. **Schema:** Follow backward-compatible validation pattern from Phase 13
5. **Tests:** Follow `tests/test_pr_curves.py` pattern -- synthetic data, known expected outputs
6. **Report:** Follow collapsible section pattern from Phase 13 HTML template

## Plan Decomposition

### Plan 15-01: Full Spectrum Storage and Curvature/Torsion Analysis
**Scope:** Modify evaluation pipeline to store top-k singular values. Create `src/analysis/spectrum.py` with curvature/torsion computation. Create tests.
**Requirements:** SPEC-01 (store full spectrum), SPEC-02 (curvature/torsion computation)
**Depends on:** Nothing (first plan)
**Files:** src/evaluation/pipeline.py (modify), src/analysis/spectrum.py (new), tests/test_spectrum.py (new)

### Plan 15-02: Curvature/Torsion as AUROC Predictive Metrics
**Scope:** Feed curvature/torsion arrays into AUROC pipeline as secondary metrics. Add visualization and report integration.
**Requirements:** SPEC-03 (AUROC integration)
**Depends on:** 15-01
**Files:** src/analysis/auroc_horizon.py (minor), src/visualization/spectrum.py (new), src/visualization/render.py (modify), src/reporting/single.py (modify), src/reporting/templates/single_report.html (modify), src/results/schema.py (modify), tests/test_spectrum.py (extend)

### Plan 15-03: Compliance Curve Sweep and Publication Figure
**Scope:** Create compliance curve aggregation module and dual-axis publication figure.
**Requirements:** COMP-01 (compliance sweep), COMP-02 (publication figure)
**Depends on:** Nothing (independent of 15-01/15-02)
**Files:** src/analysis/compliance_curve.py (new), src/visualization/compliance.py (new), src/visualization/render.py (modify), src/reporting/single.py (modify), src/reporting/templates/single_report.html (modify), src/results/schema.py (modify), tests/test_compliance_curve.py (new)

---

*Phase: 15-advanced-analysis*
*Researched: 2026-02-26*
