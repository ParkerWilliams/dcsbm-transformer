# Phase 7: Predictive Horizon and Statistical Analysis - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

For each SVD metric, measure how far in advance it can predict transformer rule violations using AUROC at each lookback distance j (from 1 to r), with position-matched baselines from successful jumper completions and rigorous statistical controls. Produces per-metric AUROC curves, predictive horizon estimates, correlation matrices, and metric importance rankings. Visualization and reporting are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Event definition and extraction
- **Violation events**: First rule violation per walk only. Avoids double-counting from cascading failures after initial break
- **Non-violation comparison class**: Rule-followed steps at jumper+r (successful jumper completions). Same context type, opposite outcome — not arbitrary non-violation steps
- **Multiple jumper encounters**: Each encounter is an independent event, with contamination filter: exclude any encounter whose countdown window [-r, 0] overlaps with a preceding violation's countdown window in the same walk. Specifically, if encounter B starts at step j_B and a prior violation occurred at step v_A, exclude B if j_B < v_A + r
- **Contamination audit**: Record exclusion count per configuration in result.json. Flag configurations losing >30% of encounters to contamination filtering
- **Alignment**: Rule resolution step (jumper_step + r) is step 0. All curves count backward from 0 to -r
- **Stratification**: Separate AUROC curves and event-aligned analysis per r value. Never mix different r values in the same curve. This is the primary analysis axis

### Pre-registered primary metrics (Holm-Bonferroni corrected)
Five pre-registered metrics covering two targets, two instability aspects, both timescales:
1. **QK^T Grassmannian distance** — routing direction drift, longest expected horizon, primary early-warning as jumper exits context window
2. **QK^T spectral gap (sigma1 - sigma2)** — dominant routing mode losing separation from noise modes. Complementary to Grassmannian: gap can collapse before subspace visibly rotates
3. **QK^T spectral entropy** — detects diffuse/confused routing as qualitatively different failure mode from directional drift. Separable from Grassmannian (direction loss vs concentration loss)
4. **AVWo stable rank** — net residual update rank collapse, strongest absolute signal in final steps before violation
5. **AVWo Grassmannian distance** — residual deposit direction rotation, distinct from rank collapse. Captures head maintaining energy but writing in wrong direction

Why not AVWo condition number: too correlated with AVWo stable rank (both detect rank deficiency), stable rank is more interpretable.

### Secondary metrics
- All 21 metrics (7 metrics x 3 targets) computed through full AUROC pipeline
- 5 primary metrics: Holm-Bonferroni corrected, reported as primary results
- 16 remaining metrics: reported separately as exploratory (uncorrected)
- Clear labeling distinguishes primary from exploratory throughout output

### Headline prediction (descriptive)
- Core falsifiable prediction: QK^T predictive horizon > AVWo predictive horizon across all r values, with the gap widening as r increases past w
- Reported as descriptive comparison with bootstrap CIs — no formal test
- If results show opposite (AVWo leading QK^T), that indicates OV circuit degrading independently of routing — a distinct mechanistic finding worth highlighting

### Statistical parameters
- **Bootstrap**: 10,000 iterations, 95% BCa (bias-corrected and accelerated) confidence intervals
- **Shuffle controls**: 10,000 permutations. Flag any metric where shuffled AUROC > 0.6 (default threshold, configurable)
- **Predictive horizon threshold**: AUROC > 0.75 (default, configurable). Furthest j exceeding threshold
- **Thresholds configurable**: Both 0.75 (horizon) and 0.6 (shuffle flag) are config parameters with these defaults, enabling sensitivity analysis

### Correlation and redundancy
- **Two correlation matrices**: (1) raw metric values pooled across events — measurement redundancy, (2) AUROC values across lookback distances — predictive redundancy. They answer different questions
- **Redundancy threshold**: |r| > 0.9 flags a pair as redundant
- **Ranking annotation**: Metric importance ranking annotated to show which top-ranked metrics are measuring essentially the same thing
- **Per-layer ranking**: Multi-layer models produce per-layer metric importance rankings (not aggregated across layers)

### Claude's Discretion
- WvWo handling: per-checkpoint reference metric, not per-step predictor — compute but don't expect per-step predictive signal
- Exact AUROC implementation details (sklearn vs manual)
- Computational optimization (batching, parallelization of bootstrap/shuffle)
- result.json schema extensions for new metrics blocks
- Edge cases: what to do when event count is too small for reliable AUROC

</decisions>

<specifics>
## Specific Ideas

- The r sweep is already a core experimental axis — stratification by r is natural and produces one predictive horizon estimate per r value, which is exactly what's needed to show the step change as r crosses w
- Successful prior encounters do not contaminate subsequent ones — the model handled them correctly and the residual stream reflects normal operation. Only prior violations are grounds for exclusion
- The relative predictive horizon comparison between QK^T and AVWo is a core result, not a secondary analysis
- The contamination filter generalizes cleanly: first-encounter-only is the degenerate case when the first encounter is a violation

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-predictive-horizon-and-statistical-analysis*
*Context gathered: 2026-02-25*
