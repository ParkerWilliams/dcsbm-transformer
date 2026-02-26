# Phase 9: Reporting and Math Verification - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate self-contained HTML reports (single-experiment and multi-experiment comparison) and a math verification PDF for peer review. Reports embed all figures as base64 and include reproduction instructions. The math PDF documents all implemented mathematics with LaTeX alongside source code for verification.

</domain>

<decisions>
## Implementation Decisions

### Report presentation
- Clean academic visual style — minimal styling, serif fonts, paper-like feel (journal supplementary page aesthetic)
- Full-width figures, one per row — no side-by-side grids
- Configuration displayed as structured HTML tables grouped by category (model, training, data) — not raw YAML/JSON
- Sections with no data shown as placeholders with "Not available" note — don't hide missing sections

### Comparison layout
- Per-metric rows with sparklines (mini bar charts) showing relative values across experiments
- Curve overlays shown as grid of individual subplots with aligned axes — not overlaid on single plot
- Config diff shown as full table with all parameters, rows where values differ highlighted
- Include auto-generated summary verdict section (e.g., "Experiment B outperforms A on 4/5 metrics")

### Math PDF scope
- All source files included — math-heavy files get full LaTeX treatment, others listed in appendix
- Sequential layout: code block first, then LaTeX math below with annotations linking math to code lines
- Thorough plain-language summaries — paragraph per file explaining what it computes, why it matters, how it connects to overall method
- Final implemented equations only — no derivation steps
- AI-generated disclaimer as standard footnote on title page (not a prominent warning box)
- pdflatex as the compilation toolchain
- Appendix lists non-math files with filename + one-liner description only
- Symbols defined in context where they first appear — no separate notation/symbol glossary table

### Reproduction block
- Minimal content: git checkout command + full CLI arguments + random seed
- Copy-pasteable code block with copy button in HTML reports
- Git commit hash captured at experiment runtime (not at report generation time)
- Dirty repo state flagged with warning if uncommitted changes existed at runtime

### Claude's Discretion
- Report generation CLI design (separate command vs automatic, flags)
- Jinja2 template structure and customization points
- Statistical test display formatting (p-values, confidence intervals)
- Sequence analysis section content and visualization approach
- Exact typography and spacing within academic style
- Loading/error state handling in reports

</decisions>

<specifics>
## Specific Ideas

- User wants to iterate on statistical test display, sequence analysis, template design, and CLI design once actual data is available to inform those decisions
- Academic style should feel like a journal supplementary page
- Comparison verdict should be auto-generated, not manual

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-reporting-and-math-verification*
*Context gathered: 2026-02-26*
