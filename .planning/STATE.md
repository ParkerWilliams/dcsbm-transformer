---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: milestone
status: executing
last_updated: "2026-02-26T17:41:31Z"
progress:
  total_phases: 10
  completed_phases: 10
  total_plans: 22
  completed_plans: 22
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-26)

**Core value:** Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations before they happen, and measure the predictive horizon.
**Current focus:** Phase 12 -- Null Model Baseline (v1.1 Journal Feedback)

## Current Position

Phase: 12 of 16 (Null Model Baseline)
Plan: 1 of 2 in current phase
Status: Executing
Last activity: 2026-02-26 -- Completed 12-01-PLAN.md (null walk generator, MP reference)

Progress: [##░░░░░░░░░░░░] 1/14 plans (7%)

## Performance Metrics

**Velocity (v1.0):**
- Total plans completed: 20
- Average duration: 3.7 min
- Total execution time: ~1.25 hours

**Velocity (v1.1):**
- Total plans completed: 1
- Average duration: 5 min
- Total execution time: 5 min

| Phase | Plan | Duration | Tasks | Files |
|-------|------|----------|-------|-------|
| 12    | 01   | 5min     | 2     | 2     |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1 roadmap]: Multi-head ablation last -- most invasive change (10+ files), build/validate all features on single-head first
- [v1.1 roadmap]: Pre-registration before any analysis -- methodological requirement, cannot be backdated
- [v1.1 roadmap]: Null model before enrichment -- if null model fails, project pivots; validate core signal first
- [v1.1 roadmap]: d_k constant across ablation (128) -- d_model scales with n_heads (1h=128, 2h=256, 4h=512)
- [v1.1 roadmap]: Phases 13 and 14 can parallelize after Phase 12 (independent dependencies)
- [12-01]: Column-filtered adjacency (zero jumper columns) instead of discard approach for null walk generation
- [12-01]: Data-calibrated sigma2 for MP reference via mean formula sigma2 = mean(sv^2) / (1 + gamma)
- [12-01]: Vectorized CDF for scipy.stats.kstest compatibility

### Pending Todos

- v1.0 integration gaps: run_experiment.py stub, set_seed not called in production, predictive_horizon not written to result.json
- v1.0 Phase 10 (Sweep Infrastructure) not implemented: MGMT-02, MGMT-03, MGMT-04, MGMT-06

### Blockers/Concerns

- Multi-head ablation requires relaxing the single-head constraint in TransformerLM (Phase 16)
- Softmax bound tightness unknown until derivation is complete -- may be vacuous (Phase 14)
- Curvature/torsion on noisy SVD output is numerically delicate -- treat as exploratory (Phase 15)
- Compliance curve requires ~45 training runs (15 r values x 3 seeds) -- GPU budget sensitive (Phase 15)
- NPZ storage for full spectrum: estimated 125-250 MB per experiment (Phase 15)

## Session Continuity

Last session: 2026-02-26
Stopped at: Completed 12-01-PLAN.md
Resume file: None
Next action: Execute 12-02-PLAN.md (statistical comparison, result storage, visualization)
