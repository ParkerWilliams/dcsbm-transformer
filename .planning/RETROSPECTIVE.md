# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.1 — Journal Feedback

**Shipped:** 2026-02-28
**Phases:** 7 | **Plans:** 15

### What Was Built
- Pre-registration framework locking Grassmannian distance hypothesis before confirmatory analysis
- Null model baseline proving SVD signal is real via Mann-Whitney U against jumper-free sequences
- Evaluation enrichment: PR curves, calibration diagnostics (ECE), SVD overhead benchmarks
- Softmax filtering bound: complete LaTeX derivation + empirical verification of QK^T -> AVWo epsilon-bound
- Full spectrum trajectory storage with Frenet-Serret curvature/torsion as exploratory AUROC metrics
- Sharp compliance curve sweep showing r/w phase transition with dual-axis publication figure
- Multi-head ablation (1h/2h/4h) with per-head SVD extraction and signal concentration analysis
- E2E pipeline wiring closing all P0/P1/P2 integration gaps from v1.0 audit

### What Worked
- Null model first strategy: validating the core signal before building enrichment features prevented wasted work on a potentially false premise
- TDD approach with standalone analysis modules: each new feature (null model, PR curves, calibration, perturbation bound) was a clean standalone module with its own test suite
- Bulk phase execution (14-16): parallelizing independent phases in a single session was efficient, though it skipped summaries
- UAT at the end: comprehensive 25-test verification caught nuanced concerns (float16 precision, bound violation semantics)

### What Was Inefficient
- Bulk execution of phases 14-16 skipped SUMMARY.md creation, causing 13 requirements to remain "Pending" in traceability despite code being shipped
- Phase 17 (gap closure) was added late after discovering integration issues from v1.0 audit — earlier integration testing would have caught these sooner
- STATE.md fell behind after bulk execution — stale context on next session resume

### Patterns Established
- Standalone analysis module pattern: each analysis feature is a self-contained module with orchestrator function and dedicated test file
- Optional schema block pattern: validate sub-structure only when block exists, no errors on absence
- Overlay visualization pattern: render base plot then add overlays from separate modules
- Separate Holm-Bonferroni correction families per analysis type
- Dual-key NPZ emission for backward compatibility during architectural changes

### Key Lessons
1. Don't skip SUMMARY.md even during bulk execution — the traceability cost compounds at milestone completion
2. Run milestone audit before bulk execution, not after — v1.0 audit revealed integration gaps that required an extra phase
3. Float16 storage is risky for derivative computation (curvature/torsion) — always consider downstream numerical precision requirements

### Cost Observations
- Model mix: ~80% opus, ~20% sonnet
- Sessions: ~5 sessions across 5 days
- Notable: Bulk execution of 3 phases in a single session was the highest-throughput pattern

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v1.0 MVP | 9 | 20 | Established TDD + GSD workflow |
| v1.1 Journal Feedback | 7 | 15 | Added pre-registration methodology, bulk phase execution |

### Cumulative Quality

| Milestone | Tests | Key Metric |
|-----------|-------|------------|
| v1.0 | ~340 | Core pipeline functional |
| v1.1 | 536+ | Full analysis suite with null validation |

### Top Lessons (Verified Across Milestones)

1. Standalone module pattern with orchestrator functions scales well across phases
2. Integration testing at milestone boundary catches gaps that unit tests miss
3. Pre-registration methodology should be the first phase when doing confirmatory research
