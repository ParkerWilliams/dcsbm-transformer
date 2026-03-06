# Phase 20: AUROC & Predictive Horizon - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify the AUROC predictive horizon pipeline is mathematically correct from event extraction through lookback indexing to horizon determination, fixing all issues found. Covers AUROC rank-based computation, lookback indexing, horizon definition consistency across all code paths, and event extraction logic. This is an audit-and-fix phase — no new features.

</domain>

<decisions>
## Implementation Decisions

### AUROC reference validation (AUROC-01)
- Verify `auroc_from_groups` against `sklearn.metrics.roc_auc_score` as the oracle reference
- Use analytic distributions with known theoretical AUROC (e.g., two Gaussians with known separation) for test fixtures
- Edge cases: tied values (midrank handling), single-element groups (NaN return), identical distributions (AUROC ~ 0.5), perfect separation (AUROC = 1.0)
- Also verify Mann-Whitney U equivalence: `scipy.stats.mannwhitneyu` U/(n1*n0) must match our AUROC output — strengthens the mathematical chain since the docstring claims this equivalence

### Lookback fence-post testing (AUROC-02)
- Planted-signal fixtures: create metric arrays with distinctive values (e.g., 999.0) at exactly one position, verify lookback j retrieves from the correct step — a fence-post error of +/-1 would retrieve wrong value
- Trace full indexing chain: build a test that constructs a walk with a known jumper encounter, traces through behavioral.py classification, event extraction, and lookback indexing end-to-end
- Verify j=1 is the correct start of the lookback range — confirm j=1 means "one step before resolution" (not the resolution step itself), document why j=0 is excluded (post-hoc, not predictive)
- Explicitly verify metric_array shape offset: metric array has max_steps-1 columns while generated has max_steps columns — this off-by-one between token indices and metric indices needs a dedicated assertion

### Cross-path horizon consistency (AUROC-03)
- Code-path audit: read null_model.py to determine if it has its own AUROC/horizon computation or delegates through run_experiment.py which calls auroc_horizon.py — document the actual code path
- Include multi-head ablation path: check signal_concentration.py and any per-head analysis for AUROC usage, verify it uses the same computation as primary
- Grep and enumerate all occurrences of 0.75 and horizon_threshold across the entire codebase — verify they all trace back to the same configurable parameter or document where they diverge
- Fix policy: if duplicated AUROC implementations are found, refactor to import from auroc_horizon.py (eliminates future drift risk)

### Event extraction boundaries (AUROC-04)
- Construct a test walk producing all 4 outcome types (UNCONSTRAINED, PENDING, FOLLOWED, VIOLATED) at known positions — verify extract_events only yields FOLLOWED/VIOLATED events
- Scenario-based contamination filter tests: (1) violation then nearby encounter -> excluded, (2) FOLLOWED then nearby encounter -> NOT excluded, (3) violation with encounter just outside window -> NOT excluded
- Cross-module seam test: build a small walk with known jumper at step t with r=3, run behavioral.py to get rule_outcome, then run extract_events, verify resolution_step matches the behavioral.py contract
- Test is_first_violation with multi-violation walks: build a walk with 2+ violations, verify only the first gets is_first_violation=True

### Claude's Discretion
- Specific numeric values for analytic distribution parameters (means, variances)
- Exact metric array dimensions and planted signal positions
- Internal organization of test classes within tests/audit/
- Order of requirement verification within plans
- Tolerance thresholds for distribution-based assertions

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/analysis/auroc_horizon.py:auroc_from_groups` — Rank-based AUROC via scipy.stats.rankdata, audit target for AUROC-01
- `src/analysis/auroc_horizon.py:compute_auroc_curve` — Lookback indexing with resolution_step - j, audit target for AUROC-02
- `src/analysis/auroc_horizon.py:compute_predictive_horizon` — Scans from largest j to smallest, returns max j > threshold, audit target for AUROC-03
- `src/analysis/event_extraction.py:extract_events` — Event extraction from generated sequences, audit target for AUROC-04
- `src/analysis/event_extraction.py:filter_contaminated_events` — Contamination filter with asymmetric logic (only violations contaminate)
- `src/analysis/event_extraction.py:stratify_by_r` — Groups events by r value for independent AUROC analysis
- `tests/audit/` — Established audit test directory from Phases 18-19 (existing tests)

### Established Patterns
- Audit test pattern: descriptive class per formula aspect, 1-2 line mathematical reasoning comment per assertion (Phase 18)
- Independent algebra check pattern: recompute with separate code, compare (Phase 19)
- Fix-on-discovery policy (Phase 18)
- `statistical_controls.py` imports `auroc_from_groups`, `compute_auroc_curve`, `compute_predictive_horizon` directly from `auroc_horizon.py` — shared code by import
- `null_model.py` does NOT import from `auroc_horizon.py` — potential consistency gap to investigate

### Integration Points
- `auroc_horizon.py` imports from `event_extraction.py` (AnalysisEvent, extract_events, filter_contaminated_events, stratify_by_r)
- `auroc_horizon.py` imports RuleOutcome from `behavioral.py` — 4-class enum verified in Phase 18
- `run_experiment.py` calls `run_auroc_analysis` and stores results in `metrics.predictive_horizon`
- `statistical_controls.py` calls `compute_auroc_curve` and `auroc_from_groups` for bootstrap CIs and correlation analysis
- `signal_concentration.py` may have per-head AUROC — needs code-path investigation

</code_context>

<specifics>
## Specific Ideas

- Phase 18 verified the 4-class behavioral labels, Phase 19 verified SVD metrics — this phase audits the pipeline that consumes both
- The null_model.py non-import of auroc functions is a concrete lead for AUROC-03 consistency investigation
- The metric_array shape (max_steps-1 vs max_steps) is a subtle potential source of off-by-one bugs that may not be caught by naive testing

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 20-auroc-predictive-horizon*
*Context gathered: 2026-03-06*
