# Phase 15: Advanced Analysis - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Full spectrum trajectory tracking with curvature/torsion geometric analysis feeding into the AUROC pipeline as secondary predictive metrics, and sharp compliance curve characterization across fine-grained r/w ratio sweep. Deliverables: spectrum storage in NPZ, Frenet-Serret curvature/torsion computation with Savitzky-Golay smoothing, curvature/torsion AUROC integration, compliance curve sweep aggregation, and dual-axis publication figure. Multi-head extensions and softmax filtering bound are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Full Spectrum Storage
- Store full singular value vectors (top-k values, k configurable, default 8) per step in NPZ alongside existing scalar metrics
- QK^T only by default (primary target), configurable to include other targets
- float16 storage to manage overhead (~125 MB for anchor config with QK^T, all 4 layers)
- Separate NPZ file (spectrum_trajectories.npz) to preserve backward compatibility with token_metrics.npz
- NPZ keys follow existing convention: qkt.layer_N.spectrum with shape [n_sequences, n_steps, k]

### Curvature/Torsion Computation
- Savitzky-Golay filter (window=7, polyorder=3) applied before differentiation to suppress numerical noise
- Discrete Frenet-Serret curvature: kappa(t) = ||d2sigma/dt2|| / ||d1sigma/dt||^3
- Discrete torsion: tau(t) via the standard formula involving first, second, and third derivatives
- Handle singular value ordering crossings by detecting and interpolating through them
- Report windowed statistics (mean curvature over lookback window) rather than raw pointwise values
- Treat curvature/torsion as exploratory (secondary) metrics, not primary hypothesis

### Compliance Curve
- Post-hoc aggregation over independently trained experiments at different r values
- Uses existing r values from R_SCALES in jumpers.py: {0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w}
- Compliance data extracted from result.json files (edge_compliance, rule_compliance, predictive_horizon)
- 3 seeds per r value for error bars (mean +/- std)
- Publication figure with dual y-axes: compliance rate (left) and predictive horizon (right)

### Claude's Discretion
- Top-k value for spectrum storage (8 suggested, may adjust based on signal analysis)
- Savitzky-Golay window/polyorder tuning if needed
- Torsion formula variant (cross-product vs determinant form)
- Compliance figure layout choices (line styles, error band vs error bars)
- Whether to include edge compliance alongside rule compliance on the figure
- Ordering crossing detection threshold

</decisions>

<specifics>
## Specific Ideas

- Curvature spike before violation events is the key signal to validate -- "does the spectrum trajectory bend sharply before failures?"
- The compliance curve should show sigmoid-like transition near r/w ~ 1.0, analogous to phase transitions in statistical mechanics
- Predictive horizon should peak in the transition region (model "trying but failing" creates strongest spectral instability)
- Curvature/torsion AUROC should be computed alongside but reported separately from primary metrics

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope

</deferred>

---

*Phase: 15-advanced-analysis*
*Context gathered: 2026-02-26*
