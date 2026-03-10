---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: Mathematical Audit
status: executing
stopped_at: Completed 22-01-PLAN.md
last_updated: "2026-03-10T18:22:12.557Z"
last_activity: 2026-03-10 — Completed 22-01 softmax bound derivation audit (SFTX-01, SFTX-02, SFTX-03)
progress:
  total_phases: 6
  completed_phases: 4
  total_plans: 11
  completed_plans: 10
  percent: 97
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-05)

**Core value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.
**Current focus:** v1.2 Mathematical Audit — Phase 22: Softmax Filtering Bound

## Current Position

Phase: 22 of 23 (Softmax Bound & Null Model)
Plan: 1 of 2
Status: Executing Phase 22 -- Plan 01 complete
Last activity: 2026-03-10 — Completed 22-01 softmax bound derivation audit (SFTX-01, SFTX-02, SFTX-03)

Progress: [██████████] 97% (v1.2 Milestone)

## Performance Metrics

**Velocity (v1.0):**
- Total plans completed: 20
- Average duration: 3.7 min
- Total execution time: ~1.25 hours

**Velocity (v1.1):**
- Total plans completed: 15
- Commits: 39
- Timeline: 5 days (2026-02-23 -> 2026-02-28)
- Codebase: 23,652 LOC Python (111 files)

**Velocity (v1.2):**
- Total plans completed: 3
- Phases: 6 (Phases 18-23)
- Requirements: 31

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 18    | 01   | 5min     | 3     | 5     |
| 18    | 02   | 8min     | 2     | 13    |
| 19    | 01   | 22min    | 3     | 3     |
| 19    | 03   | 23min    | 1     | 1     |
| Phase 19 P02 | 26min | 2 tasks | 3 files |
| 20    | 01   | 8min     | 2     | 2     |
| 20    | 02   | 7min     | 2     | 2     |
| Phase 20 P02 | 7min | 2 tasks | 2 files |
| 21    | 01   | 11min    | 2     | 4     |
| 21    | 02   | 12min    | 2     | 3     |
| Phase 22 P01 | 8min | 2 tasks | 2 files |

## Accumulated Context

### Decisions

All decisions logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:
- [v1.2]: Mathematical audit milestone — verify every formula matches its implementation
- [18-01]: Batch walk floor(U*d) bias documented as negligible (<1/2^53)
- [18-01]: All DCSBM/walk/jumper/compliance formulas verified correct -- no production code changes needed
- [18-02]: PENDING labels steps where constraint active but deadline in future, distinct from UNCONSTRAINED
- [18-02]: Consumers filter to resolved-only outcomes (FOLLOWED/VIOLATED) preserving semantic parity
- [19-01]: All SVD matrix constructions and metric formulas verified correct -- no production code changes needed
- [19-03]: Discrete curvature achieves O(h^2) convergence on circle, better than expected O(h)
- [19-03]: Synthetic spectra use descending bases to isolate formula audit from crossing-mask logic
- [v1.1 deferred]: Spectrum trajectory float32 storage concern flagged for audit (SVD-05)
- [v1.1 deferred]: Curvature/torsion float16 quantization concern (SVD-06)
- [Phase 19]: Float16 spectrum storage produces 1130% curvature error -- upgraded to float32 in pipeline.py (SVD-05)
- [Phase 19]: Grassmannian distance formula verified correct against Edelman et al. (1998) geodesic definition (SVD-04)
- [20-01]: auroc_from_groups matches sklearn and Mann-Whitney U within 1e-10 -- no production code changes needed
- [20-01]: Lookback indexing has no fence-post error -- j=1 retrieves metric at resolution_step-1
- [20-01]: compute_predictive_horizon correctly uses strict inequality (val > threshold)
- [20-02]: AST-based import verification avoids brittle string matching for code-path audits
- [20-02]: null_model.py confirmed to NOT compute AUROC -- uses Mann-Whitney U for a different statistical question
- [20-02]: Living regression test catalogs all 0.75 occurrences in src/ to detect future drift
- [Phase 20]: AST-based import verification avoids brittle string matching for code-path audits
- [21-01]: All four statistical primitives (shuffle null, BCa bootstrap, Holm-Bonferroni, Cohen's d) verified correct -- no production code changes needed
- [21-01]: Shuffle H0 uniformity tested via 100 independent trials with KS test at alpha=0.01
- [21-01]: BCa delegation verified via unittest.mock.patch on scipy.stats.bootstrap
- [21-02]: Measurement mode correlation fixed from Pearson to Spearman per STAT-05 requirement
- [21-02]: Predictive mode retains Pearson on AUROC curves -- bounded scale makes linear correlation appropriate
- [21-02]: Exploratory/confirmatory split verified: balanced, stratified, deterministic with SPLIT_SEED=2026
- [Phase 22]: LaTeX derivation verified correct step-by-step -- no formula errors found, no production code changes needed
- [Phase 22]: sqrt(d_k) cancellation algebraically proven: unscaled and scaled formulations produce identical bounds within 1e-12
- [Phase 22]: Masking consistency verified: zero-fill in direction + -inf in softmax = no inconsistency

### Pending Todos

- Sweep infrastructure (MGMT-02/03/04/06) deferred to v2
- Perturbation bound violation logging (UAT deferred idea)
- Spectrum trajectory float32 storage (UAT deferred idea)

### Blockers/Concerns

None.

## Session Continuity

Last session: 2026-03-10T18:22:12.546Z
Stopped at: Completed 22-01-PLAN.md
Resume file: None
Next action: Plan and execute Phase 22
