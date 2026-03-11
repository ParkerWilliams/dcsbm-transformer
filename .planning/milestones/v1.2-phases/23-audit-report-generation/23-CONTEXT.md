# Phase 23: Audit Report Generation - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate a self-contained HTML audit report documenting every formula-to-code mapping from the v1.2 Mathematical Audit (Phases 18-22) with correctness verdicts. The report covers 28 audited requirements across 6 categories (Graph, SVD, AUROC, Statistical, Softmax, Null Model). This is a reporting/documentation phase — no new audits or formula verification.

</domain>

<decisions>
## Implementation Decisions

### Formula data source
- Fresh curated Python registry (list of dicts) organized by audit category — do NOT reuse MATH_SECTIONS from math_pdf.py
- One entry per requirement ID (28 entries: GRAPH-01..05, SVD-01..06, AUROC-01..04, STAT-01..06, SFTX-01..03, NULL-01..04)
- Each entry contains: requirement ID, title, LaTeX formula(s), code file:line location, verdict, fix description (if applicable)
- REPT-01/02/03 entries excluded from the report (they describe the report itself, not audit findings)

### Report layout & navigation
- Sticky left sidebar TOC with category links (Graph, SVD, AUROC, Statistical, Softmax, Null Model) — stays visible while scrolling
- Each category expands in sidebar to show individual requirement IDs
- Summary dashboard at the top of the page with per-category verdict breakdown (correct/fixed/concern counts per category)
- Card-based layout per requirement: bordered card with requirement ID/title, LaTeX formula block, code location, verdict badge, fix description inline if applicable
- Code locations displayed as plain text file:line references (e.g., `src/graph/dcsbm.py:42`) — no GitHub links

### LaTeX rendering
- KaTeX for LaTeX rendering (not MathJax) — lighter, faster
- KaTeX CSS + JS vendored in `src/reporting/vendor/` — always available, no build-time network dependency (~300KB)
- KaTeX fonts embedded as base64 data URIs in the CSS — fully self-contained single HTML file (total ~2-3MB)
- Display-mode (centered block) equations for all formulas — no inline math

### Verdict presentation
- Three-tier verdicts: Correct / Fixed / Concern
- Color-coded pill-shaped badges in card headers: green (Correct), amber (Fixed), red (Concern)
- Fix descriptions displayed inline on the card below the verdict badge (not expandable/collapsible)
- Summary dashboard shows per-category breakdown table: each row = category name, correct count, fixed count, concern count

### Claude's Discretion
- Exact CSS styling within the academic aesthetic (Phase 9 established serif fonts, paper-like feel)
- Card spacing, typography, sidebar width
- Jinja2 template structure and helper functions
- How to organize the Python module (single file vs split)
- KaTeX version selection and font subsetting approach
- Order of categories in sidebar and report body

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/reporting/single.py` — Jinja2-based HTML report generator with base64 embedding pattern
- `src/reporting/templates/single_report.html` — Academic-style HTML template (serif fonts, tables, figures)
- `src/reporting/embed.py` — Base64 data URI encoding for self-contained files
- `src/reporting/math_pdf.py:MATH_SECTIONS` — LaTeX formulas for 15 math-heavy files (reference for formula content, not reused directly)
- `tests/audit/` — 26 test files with all audit findings (source of truth for what was verified)

### Established Patterns
- Jinja2 template rendering with `FileSystemLoader` and `select_autoescape` (src/reporting/single.py)
- Academic visual style: Georgia/Times serif, max-width 900px, subtle borders, alternating row colors (templates/single_report.html)
- Self-contained HTML with base64-embedded assets (src/reporting/embed.py)

### Integration Points
- New module in `src/reporting/` alongside existing report generators
- New Jinja2 template in `src/reporting/templates/`
- Vendored KaTeX in `src/reporting/vendor/`
- Registry data structure references code locations from `src/` (graph, analysis, evaluation modules)

</code_context>

<specifics>
## Specific Ideas

- The report should feel like a peer-review-ready audit document — a reviewer can open it offline and immediately see what was verified, what was fixed, and the mathematical evidence
- Per-category breakdown in dashboard gives domain experts a quick "is my area clean?" answer
- Card layout with prominent verdict badges makes scanning 28 entries fast — green/green/green/amber immediately draws attention to the fix

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 23-audit-report-generation*
*Context gathered: 2026-03-10*
