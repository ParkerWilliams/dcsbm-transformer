# Phase 18: Graph & Walk Foundations - Context

**Gathered:** 2026-03-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify every graph-theoretic formula and walk-generation algorithm against its mathematical definition, fixing all issues found. Covers DCSBM edge probabilities, walk sampling uniformity, block jumper designation, behavioral classification, and walk compliance rates. This is an audit-and-fix phase — no new features.

</domain>

<decisions>
## Implementation Decisions

### Audit evidence strategy
- Verification test scripts in `tests/audit/` directory, separate from unit tests
- Descriptive test names (not requirement-ID-coupled), e.g., `test_dcsbm_probability_symmetry`
- Each key assertion includes a 1-2 line comment explaining the mathematical reasoning
- All findings (correct formulas and fixed discrepancies) documented in a structured audit log

### Bug handling policy
- Fix discrepancies immediately upon discovery — no separate document-first pass
- Tests assert correct behavior only (no regression tests for old bugs)
- Standard fix commit messages — no special BREAKING/BEHAVIORAL tags
- When a formula is ambiguous, align to the mathematical literature definition, not existing code intent

### 4-class behavioral labels (GRAPH-04)
- Expand `RuleOutcome` from 3 values to 4: UNCONSTRAINED=0, PENDING=1, FOLLOWED=2, VIOLATED=3
- Logical enum ordering (no constraint → waiting → resolved-correct → resolved-wrong)
- Fix classification code (`behavioral.py`) AND immediate consumers (event extraction, confusion matrix)
- Leave downstream AUROC/visualization updates to their respective audit phases (20, 21)

### Sampling verification (GRAPH-02)
- Code-path argument (rng.integers is uniform by construction) plus a small-graph empirical test
- Batch walk generator float-to-int bias: flag in documentation but accept as negligible (< 1/2^53)
- Uniformity test uses fixed seed with large sample count (100k walks on ~5-node graph), deterministic assertions

### Compliance rate verification (GRAPH-05)
- Programmatic algebra check: independently count violations and constrained steps, compare ratios
- Verify formula matches mathematical literature definition (violations / constrained steps)

### Claude's Discretion
- Specific small graph topologies for test fixtures
- Tolerance thresholds for empirical frequency assertions
- Internal structure of the audit log format
- Order of requirement verification within plans

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/graph/dcsbm.py:build_probability_matrix` — Core DCSBM formula, audit target for GRAPH-01
- `src/graph/degree_correction.py:sample_theta` — Zipf theta sampling with per-block normalization
- `src/graph/jumpers.py:designate_jumpers` — Jumper designation with variable r values, audit target for GRAPH-03
- `src/graph/validation.py:check_non_trivial` — Non-triviality validation via sparse reachability
- `src/walk/generator.py` — Walk generation with path splicing (guided) and batch (unguided), audit target for GRAPH-02
- `src/walk/compliance.py:precompute_viable_paths` — Viable path pools, relevant to GRAPH-05
- `src/evaluation/behavioral.py:classify_steps` — 3-class classification, must be expanded to 4-class for GRAPH-04

### Established Patterns
- CSR sparse matrix format throughout (indptr/indices arrays for neighbor lookup)
- NumPy random Generator for reproducibility (per-walk seeds)
- Frozen dataclasses for immutable data containers (GraphData, JumperInfo, WalkResult)
- Existing test infrastructure uses pytest (tests/ directory)

### Integration Points
- `behavioral.py` RuleOutcome enum is consumed by `event_extraction.py`, `confusion.py`, `split.py`
- 4-class refactor in behavioral.py must propagate to these immediate consumers
- Block assignments computed as `np.arange(n) // block_size` — used in both graph and walk code

</code_context>

<specifics>
## Specific Ideas

- Batch walk bias (`rng.random * degree` truncation) should be documented in test comments noting the theoretical bound, not silently accepted
- Audit log should track: requirement area, finding (correct/discrepancy), file:line, description, fix applied (if any)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 18-graph-walk-foundations*
*Context gathered: 2026-03-05*
