---
phase: 11
status: passed
verified: 2026-02-26
---

# Phase 11: Pre-Registration Framework - Verification

## Phase Goal
The primary hypothesis, analysis plan, and held-out evaluation protocol are locked in git history before any v1.1 confirmatory analysis runs.

## Success Criteria Verification

### 1. Pre-registration document exists with all required elements
**Status:** PASSED

| Element | Required | Found | Evidence |
|---------|----------|-------|----------|
| File exists | Yes | Yes | `docs/pre-registration.md` |
| Grassmannian distance of QK^T as primary hypothesis | Yes | Yes | Section 2 |
| Primary metric specified | Yes | Yes | Section 3: `qkt.layer_{N}.grassmannian_distance` |
| Alpha level (0.05) | Yes | Yes | Section 4.3 |
| Holm-Bonferroni correction method | Yes | Yes | Section 4.3 |
| Decision criterion for confirm/reject | Yes | Yes | Section 6: Three-outcome (Confirm/Inconclusive/Reject) with Gate 1 + Gate 2 |

### 2. Evaluation pipeline splits walks into exploratory/confirmatory sets
**Status:** PASSED

| Element | Required | Found | Evidence |
|---------|----------|-------|----------|
| 50/50 split | Yes | Yes | `assign_split()` in `src/evaluation/split.py` |
| Stratified by event type | Yes | Yes | Violation/non-violation pools split independently |
| Deterministic assignment | Yes | Yes | Fixed seed 2026 with `np.random.default_rng` |
| result.json tags with split membership | Yes | Yes | `save_evaluation_results()` accepts `split_labels` parameter, stores in NPZ and summary |
| Tests pass | Yes | Yes | 11/11 test cases pass in `tests/test_split.py` |

### 3. Deviation log file exists and is referenced
**Status:** PASSED

| Element | Required | Found | Evidence |
|---------|----------|-------|----------|
| File exists | Yes | Yes | `docs/deviation-log.md` |
| Referenced from pre-registration | Yes | Yes | Link in pre-registration.md header and Section 8 |
| Template for recording changes | Yes | Yes | Commented template with Date, Section, Original, Change, Rationale, Impact fields |

## Requirement Coverage

| Requirement | Status | Verification |
|-------------|--------|--------------|
| PREG-01 | Covered | Pre-registration document committed to git with all specified elements |
| PREG-02 | Covered | Held-out split implemented with pipeline integration and result tagging |
| PREG-03 | Covered | Deviation log exists and is referenced from pre-registration |

## Must-Haves Verification

| Truth | Verified |
|-------|----------|
| Pre-registration document exists with Grassmannian distance as primary hypothesis | Yes |
| assign_split() deterministically assigns walks to exploratory/confirmatory | Yes (11 tests pass) |
| save_evaluation_results() accepts optional split_labels | Yes (signature verified) |
| Schema validates split_assignment fields when present | Yes (backward compatible) |
| Deviation log exists and is cross-referenced | Yes |

## Test Results

- `tests/test_split.py`: 11/11 passed
- Full suite: 342/342 passed, 0 failed

## Overall Status

**PASSED** -- All success criteria met. Phase 11 is complete.
