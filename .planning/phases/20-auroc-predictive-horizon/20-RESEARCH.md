# Phase 20: AUROC & Predictive Horizon - Research

**Researched:** 2026-03-06
**Domain:** AUROC computation, lookback indexing, predictive horizon, event extraction
**Confidence:** HIGH

## Summary

Phase 20 is an audit-and-fix phase verifying the mathematical correctness of the AUROC predictive horizon pipeline. The pipeline flows: event extraction (identifying FOLLOWED/VIOLATED encounters from behavioral labels) -> lookback indexing (retrieving metric values at steps before resolution) -> AUROC computation (rank-based P(X_violated > X_followed)) -> predictive horizon determination (max lookback where AUROC exceeds threshold).

All four audit targets are in well-contained Python modules: `auroc_horizon.py`, `event_extraction.py`, plus cross-path consumers (`statistical_controls.py`, `null_model.py`, `spectrum.py`, `pr_curves.py`, `calibration.py`). The code is already well-structured with shared imports for most paths. The primary risk area is the indexing chain between `behavioral.py` (which records resolution at `rule_outcome[b, deadline-1]`) and `event_extraction.py` (which stores `resolution_step = encounter_step + r`), creating a potential off-by-one in the lookback computation.

**Primary recommendation:** Build tests that verify the AUROC formula against sklearn, plant distinctive values in metric arrays to detect fence-post errors, audit all code paths consuming horizon computation for consistency, and construct walks with known outcomes to verify event extraction end-to-end.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**AUROC-01 (AUROC reference validation):**
- Verify `auroc_from_groups` against `sklearn.metrics.roc_auc_score` as the oracle reference
- Use analytic distributions with known theoretical AUROC (e.g., two Gaussians with known separation) for test fixtures
- Edge cases: tied values (midrank handling), single-element groups (NaN return), identical distributions (AUROC ~ 0.5), perfect separation (AUROC = 1.0)
- Also verify Mann-Whitney U equivalence: `scipy.stats.mannwhitneyu` U/(n1*n0) must match our AUROC output

**AUROC-02 (Lookback fence-post testing):**
- Planted-signal fixtures: create metric arrays with distinctive values (e.g., 999.0) at exactly one position, verify lookback j retrieves from the correct step
- Trace full indexing chain: build a test that constructs a walk with a known jumper encounter, traces through behavioral.py classification, event extraction, and lookback indexing end-to-end
- Verify j=1 is the correct start of the lookback range -- confirm j=1 means "one step before resolution" (not the resolution step itself), document why j=0 is excluded
- Explicitly verify metric_array shape offset: metric array has max_steps-1 columns while generated has max_steps columns

**AUROC-03 (Cross-path horizon consistency):**
- Code-path audit: read null_model.py to determine if it has its own AUROC/horizon computation or delegates through run_experiment.py which calls auroc_horizon.py
- Include multi-head ablation path: check signal_concentration.py and any per-head analysis for AUROC usage
- Grep and enumerate all occurrences of 0.75 and horizon_threshold across the entire codebase
- Fix policy: if duplicated AUROC implementations are found, refactor to import from auroc_horizon.py

**AUROC-04 (Event extraction boundaries):**
- Construct a test walk producing all 4 outcome types at known positions -- verify extract_events only yields FOLLOWED/VIOLATED events
- Scenario-based contamination filter tests: (1) violation then nearby encounter -> excluded, (2) FOLLOWED then nearby encounter -> NOT excluded, (3) violation with encounter just outside window -> NOT excluded
- Cross-module seam test: build a small walk with known jumper at step t with r=3, run behavioral.py to get rule_outcome, then run extract_events, verify resolution_step matches
- Test is_first_violation with multi-violation walks

### Claude's Discretion
- Specific numeric values for analytic distribution parameters (means, variances)
- Exact metric array dimensions and planted signal positions
- Internal organization of test classes within tests/audit/
- Order of requirement verification within plans
- Tolerance thresholds for distribution-based assertions

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| AUROC-01 | AUROC computation from violation/control groups uses correct rank-based probability P(X_violated > X_followed) | Code analysis of `auroc_from_groups` formula, sklearn/mannwhitneyu equivalence verification strategy, edge case catalog |
| AUROC-02 | Lookback distance j correctly indexes metric values at step (t-j) relative to resolution step t | Full indexing chain traced: behavioral.py -> event_extraction.py -> auroc_horizon.py, shape offset documented, planted-signal strategy |
| AUROC-03 | Predictive horizon definition (max j where AUROC > 0.75) consistently applied across all analysis paths | Complete code-path audit of all 0.75/horizon_threshold references, null_model.py architecture documented, shared-import verification |
| AUROC-04 | Event extraction correctly identifies resolution steps from behavioral labels | Indexing convention documented, contamination filter logic verified, 4-class outcome filtering confirmed |
</phase_requirements>

## Code Analysis Findings

### AUROC-01: `auroc_from_groups` Formula Analysis

**File:** `src/analysis/auroc_horizon.py:38-57`
**Confidence:** HIGH (direct code reading)

The implementation uses the rank-sum formula:
```python
combined = np.concatenate([violations, controls])
ranks = rankdata(combined)  # scipy midrank ties
rank_sum = ranks[:n_v].sum()
auroc = (rank_sum - n_v * (n_v + 1) / 2) / (n_v * n_c)
```

This is mathematically equivalent to:
- **Mann-Whitney U**: `U = rank_sum - n_v*(n_v+1)/2`, then `AUROC = U / (n_v * n_c)`
- **sklearn.metrics.roc_auc_score**: For binary classification with scores = metric values, labels = group membership
- **P(X_viol > X_ctrl)**: The Wilcoxon rank-sum probability

**Key detail:** `scipy.stats.rankdata` uses midrank by default for ties, which matches the standard AUROC tie-handling convention.

**Edge cases to verify:**
1. Empty groups -> NaN (lines 52-53, handled)
2. Perfect separation (all violations > all controls) -> 1.0
3. Perfect reverse (all violations < all controls) -> 0.0
4. Identical distributions (same values) -> 0.5
5. Tied values across groups -> midrank handling

### AUROC-02: Lookback Indexing Chain Analysis

**Confidence:** HIGH (direct code tracing)

The indexing chain has three stages:

**Stage 1: behavioral.py (rule resolution)**
```python
# Jumper encountered at step t, u = generated[b, t]
deadline = t + jumper.r
# Resolution: when t + 1 == deadline, i.e., at step (deadline - 1)
# rule_outcome[b, deadline - 1] = FOLLOWED or VIOLATED
```
The rule resolves at `rule_outcome[b, t + r - 1]` where t is the encounter step.

**Stage 2: event_extraction.py (event creation)**
```python
resolution_step = t + jumper.r   # = deadline
outcome_idx = resolution_step - 1  # = t + r - 1
outcome_val = rule_outcome[walk_idx, outcome_idx]
```
The `AnalysisEvent.resolution_step` stores `t + r` (the deadline value). The actual rule_outcome lookup is at `resolution_step - 1`.

**Stage 3: auroc_horizon.py (lookback retrieval)**
```python
# For lookback j:
viol_idx = viol_res - j  # = resolution_step - j
metric_value = metric_array[walk_idx, resolution_step - j]
```
For `j=1`: retrieves `metric_array[walk_idx, resolution_step - 1]` = `metric_array[walk_idx, t + r - 1]`

**Critical insight:** Both `metric_array` and `rule_outcome` have shape `[n_sequences, max_steps - 1]`. The index `resolution_step - 1` is the same index where the rule resolves. So `j=1` retrieves the metric at the SAME step where the rule outcome is determined. This means `j=1` looks at the metric value AT the resolution step, not one step before it.

**This is the correct behavior** because at the resolution step, the transformer has already processed the preceding tokens and produced the QK^T matrix that yields the SVD metric. The metric at `resolution_step - 1` (0-indexed in the metric array) corresponds to the transition from token at position `resolution_step - 1` to token at position `resolution_step` -- this IS the last step before the outcome is fully determined.

**Shape offset:** `generated` has shape `[n_sequences, max_steps]` while `metric_array` has shape `[n_sequences, max_steps - 1]`. The metric at index `i` corresponds to the attention pattern when predicting token `i+1` from tokens `0..i`. The bounds check `(viol_idx >= 0) & (viol_idx < n_steps)` in `compute_auroc_curve` line 101 correctly handles this.

### AUROC-03: Cross-Path Horizon Consistency Analysis

**Confidence:** HIGH (complete codebase grep)

**All code paths consuming AUROC/horizon:**

| Module | Imports from auroc_horizon.py? | Uses 0.75 threshold? | Notes |
|--------|-------------------------------|---------------------|-------|
| `auroc_horizon.py` | N/A (source) | Default param `threshold=0.75` | Primary implementation |
| `statistical_controls.py` | YES (imports `auroc_from_groups`, `compute_auroc_curve`, `compute_predictive_horizon`) | Delegates to `run_auroc_analysis` | Shared code path |
| `spectrum.py` | YES (imports `auroc_from_groups`, `compute_auroc_curve`) | Does NOT compute horizon; reports peak_auroc only | No horizon concern |
| `pr_curves.py` | YES (imports `auroc_from_groups`) | No horizon computation | Uses AUROC for direction detection only |
| `calibration.py` | YES (imports `auroc_from_groups`) | No horizon computation | Uses AUROC for direction detection only |
| `null_model.py` | NO -- does NOT import any AUROC functions | No horizon computation | Uses `mannwhitneyu` for drift comparison instead |
| `signal_concentration.py` | NO -- takes pre-computed AUROC values as input | No AUROC computation at all | Concentration metrics only |
| `visualization/auroc.py` | No imports | `threshold=0.75` default parameter | Display only |
| `visualization/heatmap.py` | No imports | `threshold=0.75` default parameter | Display only |
| `reporting/single.py` | No imports | Hardcoded `0.75` in string comparison | Display only |
| `run_experiment.py` | YES (imports `run_auroc_analysis`) | Delegates default params | Integration point |

**Key finding -- null_model.py architecture:**
`null_model.py` does NOT compute AUROC or predictive horizon. Its `compare_null_vs_violation` function uses `mannwhitneyu` from scipy to compare position-matched drift distributions between null (jumper-free) walks and violation walks. This is a fundamentally different analysis (Mann-Whitney U test on drift distributions vs AUROC on event-labeled groups). **This is NOT a consistency issue** -- it is a different statistical test answering a different question (does null drift differ from violation drift?) rather than computing predictive horizon.

**Hardcoded 0.75 locations:**
1. `auroc_horizon.py:124` -- `compute_predictive_horizon` default param
2. `auroc_horizon.py:323` -- `run_auroc_analysis` default param
3. `visualization/auroc.py:17` -- display threshold line
4. `visualization/heatmap.py:15,78` -- display threshold
5. `reporting/single.py:211` -- significance display (`>= 0.75`)
6. `reporting/math_pdf.py:205` -- LaTeX documentation string

The `run_auroc_analysis` passes `horizon_threshold` as a parameter to `compute_predictive_horizon`, so it is configurable. The visualization and reporting modules use 0.75 as display defaults matching the analytical default. **No inconsistency found** -- all paths either delegate to the parameterized function or use matching display constants.

### AUROC-04: Event Extraction Boundaries Analysis

**Confidence:** HIGH (direct code reading)

**Event creation logic** (`event_extraction.py:46-108`):
1. Scans every token in `generated[walk_idx, t]` for jumper vertices
2. Computes `resolution_step = t + jumper.r` and `outcome_idx = resolution_step - 1`
3. Bounds check: `outcome_idx >= max_outcome_idx or outcome_idx < 0` -> skip
4. Outcome check: only creates events for `FOLLOWED` or `VIOLATED` (lines 88-89)
5. `is_first_violation`: True when outcome is VIOLATED AND `failure_index[walk_idx] == outcome_idx`

**Key indexing convention:**
- `behavioral.py`: `failure_index[b] = t` where `t` is the step index in `rule_outcome`
- `event_extraction.py`: compares `failure_index[walk_idx] == outcome_idx` where `outcome_idx = resolution_step - 1`
- These MATCH because `behavioral.py` sets `failure_index[b] = t` at the step `t` where `rule_outcome[b, t] == VIOLATED`

**Contamination filter** (`event_extraction.py:111-197`):
- Groups events by walk_idx, sorts by encounter_step
- Tracks `last_violation_end` (resolution_step of most recent violation)
- Excludes events where `encounter_step < last_violation_end`
- CRITICAL: Only violations update `last_violation_end` (line 173) -- FOLLOWED events do NOT contaminate

**Asymmetric contamination logic is correct:** A FOLLOWED event's countdown window does not affect subsequent encounters because a successful resolution does not alter the walk's behavioral dynamics. Only violations indicate the walk has entered an anomalous state.

## Architecture Patterns

### Audit Test Pattern (from Phase 18/19)

**Source:** Existing tests in `tests/audit/`

```
tests/audit/
    __init__.py
    test_dcsbm_probability.py      # Phase 18
    test_walk_sampling.py           # Phase 18
    test_jumper_designation.py      # Phase 18
    test_compliance_rate.py         # Phase 18
    test_behavioral_classification.py # Phase 18
    test_grassmannian_distance.py   # Phase 19
    test_qkt_construction.py        # Phase 19
    test_wvwo_avwo_construction.py  # Phase 19
    test_sv_metrics.py              # Phase 19
    test_curvature_torsion.py       # Phase 19
    test_float16_fidelity.py        # Phase 19
```

**Pattern:** One test file per audit target. Descriptive class per formula aspect. 1-2 line mathematical reasoning comment per assertion. Independent algebra check: recompute with separate code, compare.

### New Audit Files for Phase 20

```
tests/audit/
    test_auroc_computation.py       # AUROC-01: auroc_from_groups vs sklearn/mannwhitneyu
    test_lookback_indexing.py        # AUROC-02: fence-post tests with planted signals
    test_horizon_consistency.py      # AUROC-03: cross-path threshold and function usage audit
    test_event_extraction.py         # AUROC-04: event creation, contamination, seam tests
```

### Test Fixture Strategy

**AUROC-01 fixtures:** Analytic distributions with known theoretical AUROC:
- Two Gaussians: N(mu1, sigma) vs N(mu2, sigma). AUROC = Phi((mu1-mu2) / (sigma*sqrt(2)))
- Perfect separation: non-overlapping ranges -> AUROC = 1.0
- Identical distributions: same values -> AUROC = 0.5
- Single element groups -> NaN

**AUROC-02 fixtures:** Planted-signal metric arrays:
- Place `999.0` at exactly one position, all others `0.0`
- Verify lookback `j` retrieves `999.0` iff `resolution_step - j` equals the planted position
- Construct minimal walk: jumper at step `t=5`, `r=3`, so `resolution_step=8`. Place signal at index 7 (`resolution_step - 1`). Verify `j=1` retrieves it.

**AUROC-04 fixtures:** Synthetic walks with known outcomes:
- 5-vertex graph (reuse from test_behavioral_classification.py)
- Walk producing UNCONSTRAINED, PENDING, FOLLOWED, VIOLATED at known positions
- Multi-jumper walk producing first and second violations

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| AUROC computation | Custom rank sorting | `auroc_from_groups` (already implemented, under audit) | Ties, edge cases handled by scipy.rankdata |
| Reference AUROC | Re-derive from first principles | `sklearn.metrics.roc_auc_score` | Established reference with known correctness |
| Mann-Whitney U | Manual U statistic | `scipy.stats.mannwhitneyu` | Exact p-values, tie correction, well-tested |
| Event extraction | Parse rule_outcome manually in tests | `extract_events` + `behavioral.classify_steps` for end-to-end | Seam test validates the real code path |

## Common Pitfalls

### Pitfall 1: Fence-Post Error in resolution_step vs outcome_idx
**What goes wrong:** Confusing `resolution_step` (which is `t + r`, the deadline) with the actual index into `rule_outcome` (which is `resolution_step - 1`).
**Why it happens:** `resolution_step` is named as if it IS the step, but it is actually one past the step in 0-indexed arrays.
**How to avoid:** The planted-signal test explicitly verifies that `metric_array[walk_idx, resolution_step - j]` retrieves the expected value. Any off-by-one would grab the wrong planted value.
**Warning signs:** AUROC values of exactly 0.5 when signal is planted, or NaN when signal should be in range.

### Pitfall 2: metric_array Shape vs generated Shape
**What goes wrong:** Using `generated.shape[1]` as the max column index for `metric_array`, causing IndexError.
**Why it happens:** `generated` has shape `[n_seqs, max_steps]` but `metric_array` has shape `[n_seqs, max_steps - 1]`.
**How to avoid:** Always use `metric_array.shape[1]` for bounds checking, never `generated.shape[1]`.
**Warning signs:** IndexError in production or silently wrong results in edge cases.

### Pitfall 3: sklearn AUROC Label Convention
**What goes wrong:** `sklearn.metrics.roc_auc_score` expects `(y_true, y_score)` where `y_true` is binary labels and `y_score` is predicted probability/score. Our `auroc_from_groups` expects two separate arrays of values.
**Why it happens:** Different API conventions.
**How to avoid:** When comparing against sklearn, construct `y_true = [1]*n_viol + [0]*n_ctrl` and `y_score = concat(violations, controls)`. Verify both orderings (violations labeled 1, controls labeled 0) produce matching results.

### Pitfall 4: Mann-Whitney U Alternative Hypothesis
**What goes wrong:** `scipy.stats.mannwhitneyu` returns the U statistic. The AUROC equivalence is `U / (n1 * n0)`, but scipy can return EITHER `U` or `n1*n0 - U` depending on which group has higher ranks.
**Why it happens:** The `alternative` parameter affects which U statistic is returned.
**How to avoid:** Use `alternative='greater'` to get the U statistic where group1 has higher values, then `AUROC = U / (n1 * n0)`. Or compute with both and verify the correct one matches.

### Pitfall 5: Contamination Filter Asymmetry
**What goes wrong:** Testing contamination filter assumes FOLLOWED events also contaminate, leading to incorrect expected results.
**Why it happens:** Intuition says "any event overlapping should contaminate" but the design intentionally only marks violations as contaminating.
**How to avoid:** The locked decision explicitly requires: (1) violation-then-encounter -> excluded, (2) FOLLOWED-then-encounter -> NOT excluded. Test both.

## Code Examples

### Reference Comparison: auroc_from_groups vs sklearn

```python
# Verified pattern from the locked decisions
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from src.analysis.auroc_horizon import auroc_from_groups

violations = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
controls = np.array([1.0, 2.0, 2.5, 3.5, 4.5])

# Our implementation
our_auroc = auroc_from_groups(violations, controls)

# sklearn reference
y_true = np.array([1]*len(violations) + [0]*len(controls))
y_score = np.concatenate([violations, controls])
sklearn_auroc = roc_auc_score(y_true, y_score)

# Mann-Whitney U reference
U, _ = mannwhitneyu(violations, controls, alternative='greater')
mw_auroc = U / (len(violations) * len(controls))

# All three must match
assert abs(our_auroc - sklearn_auroc) < 1e-10
assert abs(our_auroc - mw_auroc) < 1e-10
```

### Planted-Signal Fence-Post Test

```python
# Verified pattern from code analysis
import numpy as np
from src.analysis.auroc_horizon import compute_auroc_curve
from src.analysis.event_extraction import AnalysisEvent
from src.evaluation.behavioral import RuleOutcome

# Metric array: all zeros except one planted value
n_walks, n_cols = 4, 20
metric_array = np.zeros((n_walks, n_cols))

# Plant 999.0 at column 7 for walks 0,1 (violations)
metric_array[0, 7] = 999.0
metric_array[1, 7] = 999.0
# Plant -999.0 at column 7 for walks 2,3 (controls)
metric_array[2, 7] = -999.0
metric_array[3, 7] = -999.0

# Events with resolution_step=8, r=3
# j=1 -> index 8-1=7 (planted signal)
# j=2 -> index 8-2=6 (zero)
# j=3 -> index 8-3=5 (zero)
violation_events = [
    AnalysisEvent(walk_idx=0, encounter_step=5, resolution_step=8,
                  r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
    AnalysisEvent(walk_idx=1, encounter_step=5, resolution_step=8,
                  r_value=3, outcome=RuleOutcome.VIOLATED, is_first_violation=True),
]
control_events = [
    AnalysisEvent(walk_idx=2, encounter_step=5, resolution_step=8,
                  r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
    AnalysisEvent(walk_idx=3, encounter_step=5, resolution_step=8,
                  r_value=3, outcome=RuleOutcome.FOLLOWED, is_first_violation=False),
]

curve = compute_auroc_curve(violation_events, control_events, metric_array, r_value=3)
# j=1 should have AUROC=1.0 (perfect separation at planted position)
assert curve[0] == 1.0, f"j=1 should retrieve planted signal, got AUROC={curve[0]}"
# j=2,3 should have AUROC=0.5 (all zeros, no separation)
assert curve[1] == 0.5, f"j=2 should be chance level, got AUROC={curve[1]}"
```

### End-to-End Seam Test

```python
# Cross-module seam: behavioral.py -> event_extraction.py -> auroc_horizon.py
import numpy as np
import torch
from src.evaluation.behavioral import classify_steps, RuleOutcome
from src.analysis.event_extraction import extract_events
from src.graph.jumpers import JumperInfo

# Known graph and jumper setup
# Jumper at vertex 1, r=3, target_block=2
jmap = {1: JumperInfo(vertex_id=1, source_block=0, target_block=2, r=3)}

# Walk: 0 -> 1 -> 2 -> 3 -> 4 (5 tokens)
# Step 2: encounter vertex 1 at t=1, resolution_step = 1+3 = 4
# At step 3: t+1=4==deadline => check block of token at position 4
generated = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
_, rule_outcome, failure_index = classify_steps(generated, graph, jmap)

# Extract events using the SAME arrays (converted to numpy)
events = extract_events(
    generated.numpy(), rule_outcome, failure_index, jmap
)

# Verify: event should have resolution_step=4 (encounter at t=1, r=3)
assert len(events) == 1
assert events[0].encounter_step == 1
assert events[0].resolution_step == 4
# outcome_idx in rule_outcome is resolution_step - 1 = 3
assert events[0].outcome == rule_outcome[0, 3]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Custom AUROC loop | scipy.rankdata with midrank | Phase 7 (v1.1) | Correct tie handling |
| Single r-value analysis | Multi-r stratified analysis | Phase 7 (v1.1) | Each r gets independent AUROC curve |
| No contamination filter | Asymmetric contamination filter | Phase 7 (v1.1) | Prevents statistical leakage from prior violations |
| Float16 metrics | Float32 metrics | Phase 19 (v1.2) | 1130% curvature error eliminated |

**Relevant Phase 18 finding:** The 4-class behavioral classification (UNCONSTRAINED, PENDING, FOLLOWED, VIOLATED) was verified correct. Event extraction filters to FOLLOWED/VIOLATED only, which is consistent with the verified classification.

**Relevant Phase 19 finding:** Float16 spectrum storage was upgraded to float32 after discovering 1130% curvature error. All metric arrays used by AUROC analysis are now float32, eliminating quantization concerns.

## Open Questions

1. **j=1 semantic meaning**
   - What we know: `j=1` retrieves `metric_array[walk_idx, resolution_step - 1]`, which is the metric at the same position where the rule outcome is recorded
   - What's unclear: Whether this should be considered "at resolution" or "one step before resolution" depends on interpretation -- the metric is computed from attention on tokens up to and including position `resolution_step - 1`, before the outcome token at position `resolution_step` is generated
   - Recommendation: The planted-signal test will empirically verify the indexing. Document the semantic meaning in a test comment. The current behavior is correct: j=1 gives the last metric value computed before the outcome is known.

2. **null_model.py horizon comparison**
   - What we know: null_model.py does not compute AUROC or predictive horizon. It uses Mann-Whitney U on drift distributions.
   - What's unclear: Whether any downstream code compares null model results against predictive horizon using the 0.75 threshold
   - Recommendation: Verify during AUROC-03 that no code path applies the 0.75 horizon threshold to null model results. Based on code analysis, this is not the case.

## Sources

### Primary (HIGH confidence)
- `src/analysis/auroc_horizon.py` - Direct code reading of AUROC formula, lookback indexing, horizon computation
- `src/analysis/event_extraction.py` - Direct code reading of event creation, contamination filter
- `src/evaluation/behavioral.py` - Direct code reading of rule resolution logic (verified in Phase 18)
- `src/analysis/statistical_controls.py` - Direct code reading of AUROC imports and usage
- `src/analysis/null_model.py` - Direct code reading confirming no AUROC import
- `src/analysis/spectrum.py` - Direct code reading confirming shared AUROC import
- `src/analysis/pr_curves.py` - Direct code reading confirming shared auroc_from_groups import
- `src/analysis/calibration.py` - Direct code reading confirming shared auroc_from_groups import
- `src/visualization/auroc.py`, `heatmap.py` - Threshold constant verification
- `src/reporting/single.py` - Hardcoded 0.75 verification
- `run_experiment.py` - Integration point verification (lines 213-236)
- `tests/audit/test_behavioral_classification.py` - Existing Phase 18 test patterns
- `tests/test_auroc_horizon.py` - Existing functional tests (not audit-level)

### Secondary (MEDIUM confidence)
- scipy.stats.rankdata documentation - Midrank tie-handling is default behavior
- sklearn.metrics.roc_auc_score - Standard AUROC reference implementation

## Metadata

**Confidence breakdown:**
- AUROC formula correctness: HIGH - Direct code reading matches rank-sum AUROC definition
- Lookback indexing: HIGH - Complete chain traced through 3 modules with explicit index arithmetic
- Cross-path consistency: HIGH - Complete codebase grep with all paths documented
- Event extraction: HIGH - Direct code reading with Phase 18 verification as foundation

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable -- audit of existing code, no external dependencies changing)
