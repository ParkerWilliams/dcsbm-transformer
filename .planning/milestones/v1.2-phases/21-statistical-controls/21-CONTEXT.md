# Phase 21: Statistical Controls - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify all statistical testing machinery produces mathematically correct results — permutation tests, confidence intervals, multiple comparison corrections, effect sizes, and study design splits. Covers shuffle permutation null, BCa bootstrap CIs, Holm-Bonferroni correction, Cohen's d, Spearman redundancy analysis, and exploratory/confirmatory split. This is an audit-and-fix phase — no new features.

</domain>

<decisions>
## Implementation Decisions

### Shuffle permutation null verification (STAT-01)
- Synthetic H0 test: generate data where violated/followed come from the same distribution, run 100 independent trials, verify p-values are uniformly distributed via KS test against U[0,1]
- Assert metric array immutability: hash or copy the metric array before shuffle, verify unchanged after — catches accidental mutation bugs
- Assert group sizes per-permutation: inside the test, verify each permutation produces exactly n_viol 'violation' labels and n_ctrl 'control' labels (not just final output counts)
- 100 trials for the H0 uniformity test (fast, ~2 seconds, sufficient for KS at alpha=0.05)

### Bootstrap BCa confidence intervals (STAT-02)
- Verify delegation to scipy.stats.bootstrap: confirm correct arguments (method='BCa', data structure, confidence_level) and correct handling of return value
- Known-answer end-to-end test: use a dataset with analytically known CI bounds to confirm BCa produces correct intervals
- Skip BCa fallback path testing (BCa→percentile fallback is defensive code, not a mathematical claim)

### Holm-Bonferroni correction (STAT-03)
- Textbook worked example: use a published example (5 p-values) with known adjusted p-values, verify exact match
- Dedicated monotonicity enforcement test: construct p-values where raw adjustment would break monotonicity, verify the code enforces step-down monotonicity correctly

### Cohen's d effect size (STAT-04)
- Known-values hand-calculation: construct two groups with hand-calculable means and variances, verify Cohen's d matches manual computation exactly
- Test NaN guard edge cases: single-element groups return NaN, identical-value groups (pooled_std < 1e-12) return NaN, normal groups return finite values

### Spearman correlation and redundancy (STAT-05)
- Check both correlation paths: measurement redundancy uses np.corrcoef (Pearson) and predictive redundancy may differ — audit both to determine if Spearman is required per the requirement and fix if needed
- Boundary tests around |r| = 0.9 threshold: create metric pairs with correlations at 0.89, 0.90, 0.91 — verify the 0.9 cutoff is strict inequality (> not >=) as coded

### Exploratory/confirmatory split (STAT-06)
- Verify both proportions and stratification independently: (1) each stratum gets ~50/50 split, (2) violation walks evenly distributed, (3) non-violation walks evenly distributed, (4) overall ratio matches input ratio
- Assert exact determinism: run assign_split twice with same input and seed, verify arrays are identical element-by-element
- All edge cases: empty array, single walk, all-violation input, all-non-violation input, odd-count groups (verify floor division behavior)

### Claude's Discretion
- Specific numeric values for test fixtures (means, variances, sample sizes)
- Which published textbook example to use for Holm-Bonferroni
- Internal organization of test classes within tests/audit/
- Order of requirement verification within plans
- Tolerance thresholds for statistical assertions (e.g., KS p-value cutoff)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/analysis/statistical_controls.py:holm_bonferroni` — Custom step-down implementation, audit target for STAT-03
- `src/analysis/statistical_controls.py:auroc_with_bootstrap_ci` — BCa via scipy.stats.bootstrap with percentile fallback, audit target for STAT-02
- `src/analysis/statistical_controls.py:cohens_d` — Pooled std formula, audit target for STAT-04
- `src/analysis/statistical_controls.py:compute_correlation_matrix` — Pearson correlation with |r|>0.9 redundancy flagging, audit target for STAT-05
- `src/analysis/auroc_horizon.py:run_shuffle_control` — Label permutation null, audit target for STAT-01
- `src/evaluation/split.py:assign_split` — Stratified 50/50 split with fixed seed, audit target for STAT-06
- `tests/audit/` — Established audit test directory from Phases 18-20 (57 existing audit tests)

### Established Patterns
- Audit test pattern: descriptive class per formula aspect, 1-2 line mathematical reasoning comment per assertion (Phase 18)
- Independent algebra check pattern: recompute with separate code, compare (Phase 19)
- Fix-on-discovery policy (Phase 18)
- AST-based import verification for code-path audits (Phase 20)

### Integration Points
- `statistical_controls.py` imports `auroc_from_groups`, `compute_auroc_curve`, `compute_predictive_horizon` from `auroc_horizon.py` — verified in Phase 20
- `statistical_controls.py` imports event extraction functions — verified in Phase 20
- `split.py` is consumed by `run_experiment.py` at evaluation time
- `apply_statistical_controls` orchestrates all controls in a single top-level function

</code_context>

<specifics>
## Specific Ideas

- The np.corrcoef vs Spearman discrepancy (STAT-05) is a concrete lead — the requirement says Spearman but the code uses Pearson. This may be a real bug to fix.
- Holm-Bonferroni uses multiplier (m - i) with 0-based i, which is mathematically equivalent to the standard (m - k + 1) with 1-based k — but the textbook example should confirm this.
- The shuffle code pre-extracts metric values into values_by_j before permuting, which is correct (permutes labels not values) — the immutability assertion strengthens this guarantee.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 21-statistical-controls*
*Context gathered: 2026-03-07*
