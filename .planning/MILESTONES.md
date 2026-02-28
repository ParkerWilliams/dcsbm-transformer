# Milestones

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

