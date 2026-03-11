# Milestones

## v1.2 Mathematical Audit (Shipped: 2026-03-10)

**Phases:** 6 (Phases 18-23) | **Plans:** 13 | **Requirements:** 31/31
**Timeline:** 9 days (2026-03-02 → 2026-03-10)
**Codebase:** 33,515 LOC Python | 107 files changed, +19,172/-307 lines
**Audit tests created:** 308 across 22 test files
**Production bugs found and fixed:** 4
**Git range:** feat(18)..feat(23)

**Key accomplishments:**
1. Verified all graph-theoretic, SVD, AUROC, and statistical formulas against textbook definitions — 308 audit tests created
2. Fixed critical float16 spectrum storage bug causing 1130% curvature error; upgraded to float32 (SVD-05)
3. Fixed Pearson→Spearman correlation for redundancy analysis (STAT-05) and MP sigma^2 calibration divisor (NULL-02)
4. Expanded behavioral classification from 3-class to 4-class enum (UNCONSTRAINED/PENDING/FOLLOWED/VIOLATED)
5. Verified softmax bound LaTeX derivation step-by-step and proved sqrt(d_k) cancellation algebraically
6. Generated self-contained HTML audit report with 28 formula-to-code entries, KaTeX rendering, and per-category verdict dashboard

---

## v1.1 Journal Feedback (Shipped: 2026-02-28)

**Phases:** 7 (Phases 11-17) | **Plans:** 15 | **Commits:** 39
**Timeline:** 5 days (2026-02-23 → 2026-02-28)
**Codebase:** 23,652 LOC Python across 111 files
**Tests:** 536+ passing, 0 failures
**Git range:** feat(11)..feat(17)

**Key accomplishments:**
1. Pre-registration framework: locked Grassmannian distance hypothesis, held-out split, deviation log
2. Null model baseline: validated SVD signal is real via Mann-Whitney U comparison against jumper-free sequences
3. Evaluation enrichment: precision-recall curves, calibration diagnostics (ECE), SVD overhead benchmarks
4. Softmax filtering bound: LaTeX derivation of QK^T → AVWo epsilon-bound + empirical verification
5. Advanced analysis: full spectrum trajectory with curvature/torsion, compliance curve r/w sweep
6. Multi-head ablation: 1h/2h/4h support with per-head SVD extraction and signal concentration analysis
7. E2E pipeline wiring: run_experiment.py chains all stages, closing P0/P1/P2 integration gaps

**Known Gaps:**
- SFTX-01/02/03, MHAD-01/02/03/04, PREG-01/02/03, COMP-01/02, SPEC-01/02/03: Code shipped but traceability checkboxes were never updated during bulk execution of phases 14-16. All features are implemented and verified via UAT (25/25 passed).
- Sweep infrastructure (MGMT-02/03/04/06) deferred to v2.

**Deferred Ideas (from UAT):**
- Perturbation bound violation logging (investigate assumption mismatches)
- Spectrum trajectory float32 storage (avoid quantization noise in torsion)
- Post-hoc detection threshold table for exploratory metrics

---

