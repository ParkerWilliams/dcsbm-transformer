---
plan: 14-01
phase: 14-softmax-filtering-bound
status: complete
started: 2026-02-26
completed: 2026-02-26
duration: ~5min
---

# Plan 14-01: LaTeX Derivation of Softmax Filtering Bound

## What was built
- Complete standalone LaTeX derivation (`docs/softmax_bound.tex`) of the softmax filtering epsilon-bound
- Three-stage perturbation chain: QK^T -> softmax(A) -> AV -> AVWo with intermediate bounds as named propositions
- Main theorem (Theorem 6.1): ||Delta(AVWo)||_F <= epsilon * ||QK^T||_F * ||V||_2 * ||W_O||_2 / (2 * sqrt(d_k))
- Corollary 6.2: Spectral change bound via Weyl's inequality
- Softmax Lipschitz constant of 1/2 explicitly derived and cited (Gao & Pavel 2017)
- Tightness analysis discussing when bound is tight (uniform attention) vs loose (peaked attention)
- Connection to predictive horizon interpretation

## Key files
- `docs/softmax_bound.tex` - Self-contained LaTeX derivation document

## Structure
1. Introduction with notation table
2. Preliminaries (norms, submultiplicativity, Weyl's inequality)
3. Stage 1: Softmax Lipschitz Bound (Lemmas 3.1-3.2, Proposition 3.3)
4. Stage 2: Value Multiplication Bound (Proposition 4.1)
5. Stage 3: Output Projection Bound (Proposition 5.1)
6. End-to-End Bound (Theorem 6.1, Corollary 6.2)
7. Tightness Analysis (per-stage and compound looseness)
8. Connection to Predictive Horizon
9. References (Gao & Pavel, Vaswani et al., Stewart & Sun, Weyl)

## Deviations
- None. Document follows plan structure exactly.

## Self-Check: PASSED
- [x] Self-contained LaTeX document compiles with pdflatex
- [x] Complete chain: QK^T perturbation -> softmax -> AV -> AVWo spectral change
- [x] Intermediate bounds as separate lemmas/propositions with proofs
- [x] Softmax Lipschitz constant 1/2 explicitly derived and cited
- [x] 1/sqrt(d_k) scaling factor appears explicitly in the chain
- [x] End-to-end bound composed from intermediate bounds as main theorem
- [x] Tightness analysis section discusses tight vs loose conditions
- [x] Standard academic notation throughout (no codebase-specific naming)
