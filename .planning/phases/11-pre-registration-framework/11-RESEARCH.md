# Phase 11: Pre-Registration Framework - Research

**Researched:** 2026-02-26
**Domain:** Scientific pre-registration methodology, held-out evaluation splitting, Python evaluation pipeline modification
**Confidence:** HIGH

## Summary

Phase 11 establishes the methodological foundation for all v1.1 confirmatory analysis. The deliverables are: (1) a pre-registration document committed to git specifying the primary hypothesis, metric, alpha level, correction method, and three-outcome decision criterion; (2) modifications to the evaluation pipeline to split walks into exploratory/confirmatory sets with per-walk tagging in result.json; (3) a deviation log document referenced from the pre-registration.

The codebase already has the statistical infrastructure needed (Holm-Bonferroni correction in `src/analysis/statistical_controls.py`, Cohen's d computation, AUROC pipeline). The primary work is: writing the pre-registration markdown document, adding a deterministic walk-splitting function to the evaluation pipeline, modifying `save_evaluation_results()` and the result schema to include split tags, and creating the deviation log template.

**Primary recommendation:** Implement the held-out split as a thin layer between walk generation and evaluation — a function that assigns `split: "exploratory"` or `split: "confirmatory"` to each walk index using a fixed seed + stratified sampling by event type, then tags the result.json output accordingly.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Markdown file at `docs/pre-registration.md`, versioned in git
- Custom structure tailored to this project (not OSF/AsPredicted template)
- Sections: hypothesis, primary metric, statistical plan, held-out protocol, secondary metrics, deviation policy
- Include a "Prior evidence" section referencing v1.0 exploratory AUROC/horizon results as hypothesis motivation
- Single primary metric: Grassmannian distance of QK^T
- All lookback distances j from 1 to r are tested (Holm-Bonferroni corrects across all j)
- All other SVD metrics explicitly listed as secondary/exploratory
- 50/50 exploratory/confirmatory split at evaluation time (not walk generation time)
- Stratified by event type: equal proportions of violation/non-violation events in each set
- Deterministic assignment using fixed seed + walk index
- Per-walk tagging in result.json: each walk/sequence entry gets a `split: "exploratory"` or `split: "confirmatory"` field
- Soft separation: both sets accessible, but convention and documentation specify confirmatory walks are only used for pre-registered tests
- Three-outcome decision criterion (Confirm/Inconclusive/Reject) with Gate 1 (Holm-Bonferroni p < 0.05) and Gate 2 (Cohen's d >= 0.5 AND Grassmannian AUROC > best probability-level baseline)
- Baseline comparators for Gate 2 are probability-level metrics only (entropy of softmax output, oracle KL divergence) — not other SVD metrics

### Claude's Discretion
- Exact sections and ordering within the pre-registration document
- Deviation log format and structure
- How the stratified split is implemented (algorithm choice)
- How the confirmatory script is structured

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PREG-01 | Pre-registration document specifying primary hypothesis, primary metric, alpha level, correction method, and decision criterion committed to git | Pre-registration document template design (Section: Architecture Patterns) |
| PREG-02 | System implements held-out evaluation split (exploratory/confirmatory walks) and tags results with split membership | Walk splitting function design, result.json schema extension, pipeline integration points (Sections: Architecture Patterns, Code Examples) |
| PREG-03 | System maintains a deviation log tracking changes to analysis plan with rationale | Deviation log template design (Section: Architecture Patterns) |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | existing | Walk index array operations for split assignment | Already in project |
| scipy.stats | existing | Statistical functions (already used in statistical_controls.py) | Already in project |
| hashlib | stdlib | Deterministic seed derivation for split assignment | Standard library, reproducible |

### Supporting
No new libraries needed. All functionality can be built with existing project dependencies.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom split function | sklearn.model_selection.StratifiedShuffleSplit | Adds dependency for a 20-line function; not worth it |
| Markdown deviation log | JSON deviation log | Markdown is more human-readable for timestamped entries |

## Architecture Patterns

### Integration Points in Existing Codebase

The held-out split needs to integrate at the evaluation layer. Key files:

```
src/
├── evaluation/
│   ├── pipeline.py          # fused_evaluate() — add split tags to result
│   └── split.py             # NEW: walk split assignment function
├── analysis/
│   ├── event_extraction.py  # extract_events() — may need split-aware filtering
│   ├── auroc_horizon.py     # run_auroc_analysis() — may need split parameter
│   └── statistical_controls.py  # apply_statistical_controls() — no change needed
├── results/
│   └── schema.py            # validate_result() — add split field validation
docs/
├── pre-registration.md      # NEW: pre-registration document
└── deviation-log.md         # NEW: deviation log
```

### Pattern 1: Deterministic Stratified Split

**What:** Assign each walk to exploratory or confirmatory set using deterministic hashing.

**When to use:** At evaluation time, after walks are generated but before AUROC analysis.

**Algorithm:**
1. After `fused_evaluate()` completes, we have `failure_index` per walk (identifies which walks have violations).
2. Separate walk indices into two pools: violation walks (failure_index >= 0) and non-violation walks (failure_index < 0).
3. Within each pool, use a fixed-seed RNG to shuffle and split 50/50.
4. Assign `split` labels to each walk index.

**Why post-evaluation split:** The split is at evaluation time (CONTEXT.md: "not walk generation time"). We need behavioral labels to stratify, and behavioral labels come from evaluation. The split function takes the evaluation result and assigns labels.

**Example:**
```python
def assign_split(
    failure_index: np.ndarray,
    split_seed: int = 2026,
) -> np.ndarray:
    """Assign each walk to 'exploratory' or 'confirmatory' split.

    Stratified by event type (violation vs non-violation) to ensure
    equal proportions in each set. Deterministic via fixed seed.

    Args:
        failure_index: First violation step per walk, -1 if none. Shape [n_walks].
        split_seed: Fixed seed for deterministic assignment.

    Returns:
        Array of strings, shape [n_walks], each 'exploratory' or 'confirmatory'.
    """
    n_walks = len(failure_index)
    splits = np.empty(n_walks, dtype='U13')  # 'confirmatory' = 12 chars + 1

    rng = np.random.default_rng(split_seed)

    # Separate by violation status
    violation_mask = failure_index >= 0
    viol_indices = np.where(violation_mask)[0]
    nonviol_indices = np.where(~violation_mask)[0]

    # Shuffle and split each pool 50/50
    for indices in [viol_indices, nonviol_indices]:
        shuffled = rng.permutation(indices)
        half = len(shuffled) // 2
        splits[shuffled[:half]] = 'exploratory'
        splits[shuffled[half:]] = 'confirmatory'

    return splits
```

### Pattern 2: Result.json Split Tagging

**What:** Add split membership to result.json output.

**Where to modify:** `save_evaluation_results()` in `src/evaluation/pipeline.py` and `validate_result()` in `src/results/schema.py`.

**Schema extension:**
```json
{
  "metrics": {
    "scalars": { ... },
    "split_assignment": {
      "split_seed": 2026,
      "n_exploratory": 250,
      "n_confirmatory": 250,
      "n_exploratory_violations": 15,
      "n_confirmatory_violations": 15
    }
  },
  "sequences": [
    {
      "sequence_id": 0,
      "split": "exploratory",
      ...
    }
  ]
}
```

The split assignment metadata goes in `metrics.scalars` (or a new top-level key). Per-sequence split tags go in the sequences array.

### Pattern 3: Pre-Registration Document Structure

**What:** Markdown document at `docs/pre-registration.md` following scientific pre-registration conventions adapted to this project.

**Structure:**
```markdown
# Pre-Registration: QK^T Subspace Departure as Violation Predictor

## 1. Prior Evidence
## 2. Primary Hypothesis
## 3. Primary Metric
## 4. Statistical Analysis Plan
## 5. Held-Out Protocol
## 6. Decision Criterion
## 7. Secondary Metrics (Exploratory)
## 8. Deviation Policy
```

### Pattern 4: Deviation Log

**What:** A timestamped log documenting any departures from the pre-registered plan.

**Structure:**
```markdown
# Deviation Log

Referenced from: [docs/pre-registration.md](pre-registration.md)

## Log

_No deviations recorded._

<!-- Template for new entries:
### YYYY-MM-DD: [Brief description]
**Section affected:** [which pre-registration section]
**Original plan:** [what was specified]
**Change:** [what changed]
**Rationale:** [why the change was necessary]
**Impact on interpretation:** [how this affects conclusions]
-->
```

### Anti-Patterns to Avoid
- **Post-hoc pre-registration:** The pre-registration document MUST be committed before any confirmatory analysis code runs. Phase ordering enforces this.
- **Leaking confirmatory walks into exploratory analysis:** The split is soft (both accessible), but the AUROC pipeline should accept a `split` parameter to filter walks.
- **Hardcoding the AUROC > 0.75 threshold as a pre-registered cutoff:** CONTEXT.md explicitly says the comparative criterion (beats baseline metrics) replaces it.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Multiple comparison correction | Custom Bonferroni | Existing `holm_bonferroni()` in `statistical_controls.py` | Already implemented and tested |
| Bootstrap CIs | Custom bootstrap | Existing `auroc_with_bootstrap_ci()` | Already implemented |
| Cohen's d | Custom effect size | Existing `cohens_d()` | Already implemented |
| Stratified sampling | Complex stratification | Simple numpy split with fixed seed | Only 2 strata (violation/non-violation), 50/50 split |

**Key insight:** Most statistical machinery is already built in v1.0. This phase is primarily about documentation and a thin split assignment layer, not new statistical code.

## Common Pitfalls

### Pitfall 1: Non-Deterministic Split Assignment
**What goes wrong:** Using `random.random()` or system-dependent seeding produces different splits across runs.
**Why it happens:** Forgetting to fix the seed, or using a seed that depends on system state.
**How to avoid:** Use `np.random.default_rng(fixed_seed)` with a hard-coded seed constant. Document the seed in the pre-registration.
**Warning signs:** Running the same experiment twice produces different exploratory/confirmatory counts.

### Pitfall 2: Stratification Imbalance with Odd Numbers
**What goes wrong:** When violation count is odd, one split gets one more violation than the other.
**Why it happens:** Integer division rounds down.
**How to avoid:** Accept the off-by-one (it's negligible for reasonable sample sizes). Document the rounding behavior. The function should handle edge cases where all walks are violations or all are non-violations.
**Warning signs:** Assertion errors on exact 50/50 when counts are odd.

### Pitfall 3: Forgetting to Update Result Schema
**What goes wrong:** Split tags are computed but not stored in result.json because `validate_result()` rejects the new field or `save_evaluation_results()` doesn't include it.
**Why it happens:** The schema validator in `schema.py` doesn't know about the new `split` field.
**How to avoid:** Update `validate_result()` to accept (but not require) split-related fields. Make the split tagging backward-compatible (old results without splits still validate).
**Warning signs:** result.json files without split tags after running evaluation.

### Pitfall 4: Pre-Registration After Analysis
**What goes wrong:** Running confirmatory analysis code before the pre-registration is committed to git.
**Why it happens:** Phases executed out of order or confirmatory pipeline code tested on real data.
**How to avoid:** Phase ordering in the roadmap prevents this. The pre-registration document must be committed in this phase (11), before Phase 12+ runs any confirmatory analysis.
**Warning signs:** Git log shows pre-registration commit dated after analysis commits.

## Code Examples

### Walk Split Assignment
```python
# src/evaluation/split.py
import numpy as np

SPLIT_SEED = 2026  # Fixed seed for reproducibility

def assign_split(
    failure_index: np.ndarray,
    split_seed: int = SPLIT_SEED,
) -> np.ndarray:
    """Assign each walk to exploratory or confirmatory split.

    Stratified by event type to ensure equal proportions of
    violation/non-violation walks in each set.
    """
    n_walks = len(failure_index)
    splits = np.empty(n_walks, dtype='U13')
    rng = np.random.default_rng(split_seed)

    violation_mask = failure_index >= 0
    for mask_val in [True, False]:
        indices = np.where(violation_mask == mask_val)[0]
        shuffled = rng.permutation(indices)
        half = len(shuffled) // 2
        splits[shuffled[:half]] = 'exploratory'
        splits[shuffled[half:]] = 'confirmatory'

    return splits
```

### Integration with save_evaluation_results
```python
# Modification to src/evaluation/pipeline.py
def save_evaluation_results(
    result: EvaluationResult,
    output_dir: str | Path,
    split_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    # ... existing code ...

    if split_labels is not None:
        npz_data["split"] = split_labels
        summary["scalars"]["split_assignment"] = {
            "split_seed": SPLIT_SEED,
            "n_exploratory": int((split_labels == 'exploratory').sum()),
            "n_confirmatory": int((split_labels == 'confirmatory').sum()),
        }

    return summary
```

### Test for Deterministic Split
```python
def test_split_deterministic():
    """Split assignment is deterministic across runs."""
    failure_index = np.array([5, -1, -1, 3, -1, -1, 2, -1])
    split_a = assign_split(failure_index, split_seed=2026)
    split_b = assign_split(failure_index, split_seed=2026)
    assert np.array_equal(split_a, split_b)

def test_split_stratified():
    """Equal proportions of violations in each set."""
    failure_index = np.array([5, -1, -1, 3, -1, -1, 2, -1, 7, -1])
    splits = assign_split(failure_index, split_seed=2026)
    violation_mask = failure_index >= 0
    for split_val in ['exploratory', 'confirmatory']:
        split_mask = splits == split_val
        viol_in_split = (violation_mask & split_mask).sum()
        # Each split should have roughly half the violations
        total_viol = violation_mask.sum()
        assert abs(viol_in_split - total_viol // 2) <= 1
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Post-hoc corrections only | Pre-registration + held-out split | Standard practice since 2013 (OSF) | Prevents p-hacking, strengthens claims |
| Single alpha threshold | Three-outcome framework | Emerging best practice | More informative than binary accept/reject |
| Fixed AUROC cutoff | Comparative criterion (vs baseline) | Project-specific innovation | Answers "does this metric add value beyond baselines?" |

## Open Questions

1. **Baseline metrics for Gate 2 (entropy of softmax output, oracle KL divergence)**
   - What we know: These are probability-level metrics that the Grassmannian distance must outperform.
   - What's unclear: Whether these metrics are already computed in the evaluation pipeline.
   - Recommendation: Check if they exist; if not, they'll need to be added in a later phase or as part of the confirmatory analysis script. The pre-registration should specify them by name without requiring they're implemented yet.

2. **NPZ storage of split labels**
   - What we know: NPZ stores numpy arrays. String arrays work but are less efficient.
   - What's unclear: Whether to store splits as strings ('exploratory'/'confirmatory') or integers (0/1) in NPZ.
   - Recommendation: Use integers (0=exploratory, 1=confirmatory) in NPZ for efficiency, with a mapping documented in the pre-registration and in code constants. The result.json summary uses string labels for human readability.

## Sources

### Primary (HIGH confidence)
- Codebase analysis: `src/evaluation/pipeline.py` — fused_evaluate() and save_evaluation_results()
- Codebase analysis: `src/analysis/event_extraction.py` — event extraction with walk_idx tracking
- Codebase analysis: `src/analysis/auroc_horizon.py` — AUROC pipeline with PRIMARY_METRICS frozenset
- Codebase analysis: `src/analysis/statistical_controls.py` — Holm-Bonferroni, Cohen's d, bootstrap CIs
- Codebase analysis: `src/results/schema.py` — result.json validation and writing
- Codebase analysis: `src/config/experiment.py` — ExperimentConfig with frozen dataclasses

### Secondary (MEDIUM confidence)
- Scientific pre-registration best practices (OSF, AsPredicted conventions)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new libraries needed; all functionality uses existing project dependencies
- Architecture: HIGH - Clear integration points identified in existing pipeline code
- Pitfalls: HIGH - Well-understood domain (deterministic seeding, schema validation)

**Research date:** 2026-02-26
**Valid until:** Indefinite (methodology, not library-version-dependent)
