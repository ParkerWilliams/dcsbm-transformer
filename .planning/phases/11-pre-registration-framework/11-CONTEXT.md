# Phase 11: Pre-Registration Framework - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Lock the primary hypothesis, analysis plan, and held-out evaluation protocol in git history before any v1.1 confirmatory analysis runs. Deliverables: pre-registration document, exploratory/confirmatory walk split in the evaluation pipeline, and deviation log. No confirmatory analysis runs in this phase — only the framework.

</domain>

<decisions>
## Implementation Decisions

### Document format & location
- Markdown file at `docs/pre-registration.md`, versioned in git
- Custom structure tailored to this project (not OSF/AsPredicted template)
- Sections: hypothesis, primary metric, statistical plan, held-out protocol, secondary metrics, deviation policy
- Include a "Prior evidence" section referencing v1.0 exploratory AUROC/horizon results as hypothesis motivation

### Primary metric
- Single primary metric: Grassmannian distance of QK^T
- All lookback distances j from 1 to r are tested (Holm-Bonferroni corrects across all j)
- All other SVD metrics (stable rank, spectral entropy, condition number, spectral gap, rank-1 residual, read-write alignment) explicitly listed as secondary/exploratory — computed and reported but not used for the confirm/reject decision

### Held-out split mechanics
- 50/50 exploratory/confirmatory split at evaluation time (not walk generation time)
- Stratified by event type: equal proportions of violation/non-violation events in each set
- Deterministic assignment using fixed seed + walk index
- Per-walk tagging in result.json: each walk/sequence entry gets a `split: "exploratory"` or `split: "confirmatory"` field
- Soft separation: both sets accessible, but convention and documentation specify confirmatory walks are only used for pre-registered tests

### Decision criterion (three-outcome framework)
- **Confirm** (Gate 1 + Gate 2 pass): QK^T subspace departure is a statistically significant, practically meaningful, and superior-to-output-level-metrics predictor of rule violations
  - Gate 1: Holm-Bonferroni corrected p < 0.05 at any lookback j
  - Gate 2: Cohen's d >= 0.5 (medium effect) at the j that passes Gate 1, AND Grassmannian distance AUROC exceeds the best probability-level baseline metric's AUROC (entropy of softmax output, oracle KL divergence)
- **Inconclusive** (Gate 1 passes, Gate 2 fails): Signal is real but either too small (d < 0.5) or doesn't beat probability-level baselines — informative about attention geometry but scientifically redundant with output-level metrics
- **Reject** (Gate 1 fails): No statistically significant Grassmannian distance change precedes violations
- The confirmatory script outputs one of these three words alongside the numbers
- Pre-registration defines criteria only — does not commit to post-outcome next steps
- Baseline comparators for Gate 2 are probability-level metrics only (entropy of softmax output, oracle KL divergence) — not other SVD metrics

### Deviation log
- Claude's Discretion: format, location, and referencing mechanism from pre-registration document

### Claude's Discretion
- Exact sections and ordering within the pre-registration document
- Deviation log format and structure
- How the stratified split is implemented (algorithm choice)
- How the confirmatory script is structured

</decisions>

<specifics>
## Specific Ideas

- The AUROC > 0.75 threshold should NOT be a fixed pre-registered cutoff — the comparative criterion (beats baseline metrics) replaces it
- The comparative Gate 2 condition answers the key scientific question: does tracking QK^T subspace departure give you anything beyond what's visible at the output probability level?
- Three outcomes are essential because "real but small/redundant" is the most diagnostically useful outcome — it says "right track, refine the metric" without overstating or discarding the finding
- Cohen's d >= 0.5 (medium effect) is the practical significance floor

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 11-pre-registration-framework*
*Context gathered: 2026-02-26*
