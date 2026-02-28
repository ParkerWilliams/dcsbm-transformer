# Phase 13: Evaluation Enrichment - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Enrich the evaluation pipeline with precision-recall curves (AUPRC), calibration diagnostics (reliability diagrams + ECE), and SVD computational cost benchmarking. All new metrics and visualizations integrate into the existing HTML report and result.json schema. No changes to the core training loop or SVD computation itself.

</domain>

<decisions>
## Implementation Decisions

### PR Curve Presentation
- Mirror existing AUROC layout: same grid structure with one subplot per metric×lookback distance
- AUPRC values displayed both as summary table and annotated directly on each plot
- Include no-skill baseline as dashed horizontal line at the positive class prevalence rate
- Store AUPRC and PR curve data in result.json alongside existing AUROC fields (same nested structure by metric and lookback)

### Calibration Diagnostics
- 10 equal-width bins for reliability diagrams
- One reliability diagram per metric, with lookback distances as separate colored lines on the same plot (more compact than per-metric×lookback grid)
- ECE annotated on each reliability diagram plus a summary table with ECE per metric per lookback
- Include histogram of predicted probabilities below each reliability diagram showing bin counts (contextualizes sample density per bin)

### SVD Benchmark Reporting
- Cost summary table plus grouped bar chart (targets on x-axis, SVD methods as groups) for visual comparison
- Report both relative Frobenius error and singular value correlation as accuracy metrics for the cost-accuracy tradeoff
- Separate profiling mode (--benchmark flag or separate script), not run during every evaluation
- 5 warmup iterations + 20 timed iterations for CUDA event benchmarking

### Report Section Ordering
- Group by analysis type: AUROC section → PR curves section → Calibration section → SVD cost section
- Each new section is a collapsible block, collapsed by default (keeps report scannable, AUROC stays prominent)

### Claude's Discretion
- Exact color palettes and styling for new plots
- Collapsible section implementation details (HTML/CSS approach)
- PR curve interpolation method
- How to handle metrics with insufficient positive samples for meaningful PR curves

</decisions>

<specifics>
## Specific Ideas

- PR curves should feel consistent with existing AUROC plots — same styling conventions, same grid dimensions where applicable
- Reliability diagrams should include the perfect-calibration diagonal line for reference
- SVD benchmark should compare: full SVD, randomized SVD (torch.svd_lowrank), values-only SVD (torch.linalg.svdvals)
- SVD benchmark targets: QK^T, WvWo, AVWo — benchmarked per target and matrix dimension

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 13-evaluation-enrichment*
*Context gathered: 2026-02-26*
