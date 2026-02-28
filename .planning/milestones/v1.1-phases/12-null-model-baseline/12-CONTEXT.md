# Phase 12: Null Model Baseline - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Demonstrate that the SVD Grassmannian drift signal is a real response to block jumper events, not an artifact of normal attention dynamics. Generate jumper-free null walks, compute statistical comparison against violation walks, provide a Marchenko-Pastur random matrix reference, store results in result.json, and render null overlays on event-aligned plots.

</domain>

<decisions>
## Implementation Decisions

### Null walk design
- Same DCSBM graph, same trained model, walks that stay within their planted community (no block jumps)
- Generate 5x as many null walks as violation walks for tighter null distribution estimate
- Position-matched: null walks have the same total length and measurement positions as violation walks, just no jumper event at those positions
- Standalone generator function (not integrated into evaluation pipeline) — takes a trained model + graph and produces null walks independently; can run null analysis on any existing experiment

### Statistical thresholds
- Holm-Bonferroni correction at alpha=0.05 (locked from Phase 7 and Phase 11 pre-registration — already implemented in `src/analysis/statistical_controls.py`)
- Null model Mann-Whitney U tests form a **separate Holm-Bonferroni family** — correct across lookback distances within the null comparison, independent of the primary metrics family
- Cohen's d threshold matches pre-registration Gate 2: d >= 0.5 is the bar for meaningful separation between null and violation drift

### Null overlay visualization
- Shaded 95% CI band of null distribution **plus** solid null median line on event-aligned plots
- Color scheme: light gray band with gray median line for null; existing color scheme for violation signal
- Gray communicates "baseline/noise floor" intuitively

### Result JSON structure
- Full statistical summary in `null_model` block of result.json
- Per-lookback distance: null mean/std, violation mean/std, Mann-Whitney U statistic, raw p-value, Holm-Bonferroni adjusted p-value, Cohen's d, reject flag
- Aggregate: number of null walks, number of violation walks, global summary

### Reporting
- Standalone "Null Model Baseline" section in the report with its own plots and statistical summary table
- Marchenko-Pastur analysis appears as a subsection within the null model report section (supporting element of the null model narrative)

### Marchenko-Pastur reference
- Overlay MP theoretical density curve on empirical histogram of QK^T singular values
- Compute KS (Kolmogorov-Smirnov) statistic and p-value as quantitative divergence metric between empirical SVs and MP distribution
- Compute at anchor points only: event position and a few reference positions (pre-event, post-event) — not every evaluation step

### Claude's Discretion
- Exact null walk generation algorithm (how to ensure walks stay within community)
- MP density parameterization details (aspect ratio from d_k, matrix dimensions from anchor config)
- Internal structure of the standalone generator function
- Exact layout of the null model report section

</decisions>

<specifics>
## Specific Ideas

- Holm-Bonferroni implementation already exists in `src/analysis/statistical_controls.py` — reuse it for the null model family
- Cohen's d >= 0.5 threshold aligns with pre-registration Gate 2 from Phase 11, maintaining consistency across the analysis pipeline
- Position-matching is critical: without it, any drift differences could be attributed to sequence position effects rather than jumper events

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-null-model-baseline*
*Context gathered: 2026-02-26*
