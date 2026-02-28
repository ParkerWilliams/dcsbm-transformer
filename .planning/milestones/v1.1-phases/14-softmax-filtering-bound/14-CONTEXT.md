# Phase 14: Softmax Filtering Bound - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Derive and empirically verify the epsilon-bound from QK^T perturbation through softmax to AVWo spectral change. Deliverables: a LaTeX derivation document and controlled perturbation experiments with bound tightness visualization. Advanced analysis (curvature/torsion, compliance sweeps) and multi-head extensions are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Derivation scope & presentation
- Full chain derivation with intermediate bounds at each stage (QK^T -> softmax, softmax -> AV, AV -> AVWo), then composed into end-to-end bound
- Standalone .tex file (e.g., docs/softmax_bound.tex) that compiles to PDF — academic-style document
- Standard academic notation (W_Q, W_K, etc.) as in transformer papers, self-contained without codebase-specific naming
- Include tightness analysis section discussing when the bound is tight (e.g., low-entropy attention distributions) vs loose, with intuition for why

### Perturbation experiment design
- Perturbation magnitudes sized as fractions of ||QK^T|| (e.g., 1%, 5%, 10%, 25%) — relative scaling, not absolute
- Adversarial direction defined as alignment with the top singular vector of QK^T (amplifies dominant attention pattern)
- Random directions as baseline comparison alongside adversarial
- Success criterion: fewer than 5% of perturbations exceed the theoretical bound

### Claude's Discretion
- Injection step strategy (event-aligned, uniform, or both)
- Number of random perturbation directions per (step, magnitude) pair — enough for statistical reliability
- Bound tightness visualization style (scatter + envelope, violin/box, or other)
- Chain decomposition strategy in experiments — whether to track intermediate stages (softmax output, AV) or only end-to-end QK^T -> AVWo
- Publication figure quality and layout choices

</decisions>

<specifics>
## Specific Ideas

- Derivation should show the Lipschitz constant of softmax (1/2) and 1/sqrt(d_k) scaling factor explicitly in the chain
- The tightness ratio (median empirical / theoretical bound) is a key reported metric
- User expects to iterate on experiment details once data is available — initial design should be modular enough to adjust parameters easily

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-softmax-filtering-bound*
*Context gathered: 2026-02-26*
