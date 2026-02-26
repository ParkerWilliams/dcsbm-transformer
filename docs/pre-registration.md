# Pre-Registration: QK^T Subspace Departure as Transformer Rule Violation Predictor

**Version:** 1.0
**Date:** 2026-02-26
**Status:** Locked (committed to git before confirmatory analysis)
**Deviation log:** [deviation-log.md](deviation-log.md)

## 1. Prior Evidence

v1.0 exploratory analysis observed that SVD metrics computed on the QK^T attention matrix
exhibit systematic changes in the steps preceding block jumper rule violations in a
DCSBM-trained transformer. Specifically, Grassmannian distance of the top-2 left singular
subspace of QK^T showed elevated values at lookback distances j=1 through j=r before
violation events, with AUROC exceeding chance levels in exploratory runs. These findings
motivate the confirmatory hypothesis below.

All v1.0 results are exploratory. This pre-registration locks the confirmatory analysis
plan before any v1.1 confirmatory runs.

## 2. Primary Hypothesis

**H1:** The Grassmannian distance of the QK^T attention matrix's top-2 left singular
subspace increases significantly in the steps preceding a block jumper rule violation,
compared to steps preceding successful rule compliance.

**Directionality:** One-sided (violation events show *higher* Grassmannian distance than
control events at matched lookback distances).

## 3. Primary Metric

**Metric:** Grassmannian distance of QK^T (key: `qkt.layer_{N}.grassmannian_distance`)

**Definition:** The Grassmannian distance between consecutive top-k left singular subspaces
of the QK^T matrix, computed as:

    d_G(U_t, U_{t-1}) = ||U_t U_t^T - U_{t-1} U_{t-1}^T||_F / sqrt(2k)

where U_t is the matrix of the top-k=2 left singular vectors at step t.

**All lookback distances j from 1 to r** are tested. Holm-Bonferroni correction is applied
across all j values within the primary metric.

## 4. Statistical Analysis Plan

### 4.1 Event Extraction
- Events: block jumper encounters extracted from generated sequences
- Contamination filter: encounters whose countdown window overlaps a preceding
  violation's window are excluded (only violations contaminate, not successful
  compliance events)
- Stratification: events grouped by r value; each r-group analyzed independently

### 4.2 Test Statistic
- AUROC (Area Under the Receiver Operating Characteristic curve) computed via the
  rank-based method (equivalent to Mann-Whitney U / (n1 * n0))
- Computed at each lookback distance j = 1, 2, ..., r
- Violation events vs. control events (rule followed) at matched lookback positions

### 4.3 Multiple Comparison Correction
- **Method:** Holm-Bonferroni step-down procedure
- **Family:** All lookback distances j = 1..r for the primary metric
- **Alpha level:** 0.05 (family-wise error rate)
- **Implementation:** `holm_bonferroni()` in `src/analysis/statistical_controls.py`

### 4.4 Shuffle Controls
- 10,000 label permutations per metric per r-value
- Flag threshold: shuffled p95 AUROC > 0.6 indicates positional artifact
- Any flagged metric's results are annotated but not excluded

### 4.5 Confidence Intervals
- BCa bootstrap confidence intervals (10,000 resamples) on AUROC at the
  max-signal lookback distance
- Reported for strata with 10+ events per class

### 4.6 Effect Size
- Cohen's d (pooled standard deviation) at each lookback distance
- Computed for violation vs. control metric value distributions

## 5. Held-Out Protocol

### 5.1 Split Design
- **Ratio:** 50% exploratory / 50% confirmatory
- **Unit:** Individual walk (sequence)
- **Timing:** Split applied at evaluation time, after behavioral labels are computed
- **Stratification:** Equal proportions of violation and non-violation walks in each set
- **Seed:** Fixed seed (2026) with `np.random.default_rng`
- **Assignment:** Deterministic by walk index -- same walks always map to same split

### 5.2 Usage Convention
- **Exploratory set:** Used for hypothesis generation, visualization, debugging,
  parameter tuning. No restrictions on analysis.
- **Confirmatory set:** Used ONLY for the pre-registered statistical tests defined
  in this document. Results from the confirmatory set determine the final
  Confirm/Inconclusive/Reject outcome.
- **Separation:** Soft (both sets accessible in the same result.json/NPZ).
  Discipline enforced by convention and documentation, not code barriers.

### 5.3 Tagging
- Each walk tagged with `split: "exploratory"` or `split: "confirmatory"` in result.json
- Split assignment metadata (seed, counts) stored in metrics.scalars.split_assignment
- NPZ stores integer encoding: 0=exploratory, 1=confirmatory

## 6. Decision Criterion

### Three-Outcome Framework

**Confirm** (Gate 1 AND Gate 2 pass):
  QK^T subspace departure is a statistically significant, practically meaningful,
  and superior-to-output-level-metrics predictor of rule violations.

  - **Gate 1:** Holm-Bonferroni corrected p < 0.05 at any lookback j
  - **Gate 2:** Cohen's d >= 0.5 (medium effect) at the j that passes Gate 1,
    AND Grassmannian distance AUROC exceeds the best probability-level baseline
    metric's AUROC (entropy of softmax output, oracle KL divergence)

**Inconclusive** (Gate 1 passes, Gate 2 fails):
  Signal is real but either too small (d < 0.5) or does not beat probability-level
  baselines. Informative about attention geometry but scientifically redundant with
  output-level metrics.

**Reject** (Gate 1 fails):
  No statistically significant Grassmannian distance change precedes violations
  on the confirmatory set.

### Notes
- The confirmatory script outputs one of these three words alongside the numbers
- Baseline comparators for Gate 2 are probability-level metrics only (entropy of
  softmax output, oracle KL divergence) -- NOT other SVD metrics
- This pre-registration defines criteria only -- does not commit to post-outcome
  next steps

## 7. Secondary Metrics (Exploratory)

The following SVD metrics are computed and reported but are NOT used for the
Confirm/Inconclusive/Reject decision. They are exploratory:

| Metric | Target | Key Pattern |
|--------|--------|-------------|
| Stable rank | QK^T, AVWo | `{target}.layer_{N}.stable_rank` |
| Spectral entropy | QK^T, AVWo | `{target}.layer_{N}.spectral_entropy` |
| Spectral gap (1-2) | QK^T, AVWo | `{target}.layer_{N}.spectral_gap_1_2` |
| Spectral gap (2-3) | QK^T, AVWo | `{target}.layer_{N}.spectral_gap_2_3` |
| Spectral gap (4-5) | QK^T, AVWo | `{target}.layer_{N}.spectral_gap_4_5` |
| Condition number | QK^T, AVWo | `{target}.layer_{N}.condition_number` |
| Rank-1 residual norm | QK^T, AVWo | `{target}.layer_{N}.rank1_residual_norm` |
| Read-write alignment | WvWo | `wvwo.layer_{N}.read_write_alignment` |
| Grassmannian distance | AVWo | `avwo.layer_{N}.grassmannian_distance` |

These metrics have their own AUROC curves, horizons, and statistical summaries
computed and stored in result.json, but their p-values are not part of the
Holm-Bonferroni family and do not affect the decision criterion.

## 8. Deviation Policy

Any change to sections 2-6 of this document after the initial git commit
constitutes a deviation and MUST be recorded in the [deviation log](deviation-log.md)
with:
- Date
- Section affected
- Original specification
- New specification
- Rationale for the change
- Impact on interpretation of results

Changes to section 7 (secondary metrics) do not require deviation logging
since they do not affect the confirmatory decision.

---

*Pre-registration committed: 2026-02-26*
*Project: DCSBM Transformer SVD Hallucination Prediction*
