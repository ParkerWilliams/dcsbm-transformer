---
phase: 01-config-schema-and-reproducibility-foundation
status: passed
verified: 2026-02-24
requirements: [MGMT-01, MGMT-05, TRNG-02, TRNG-07]
---

# Phase 1 Verification: Config, Schema, and Reproducibility Foundation

## Phase Goal

Every experiment is fully specified by a frozen, hashable configuration object; results conform to a validated schema; and all sources of randomness are controlled from a single master seed.

## Success Criteria Verification

### SC1: ExperimentConfig with anchor parameters, serialization round-trip, hash identity
**Status: PASSED**

- ExperimentConfig instantiates with anchor defaults: n=500, w=64, t=200k, d_model=128, n_layers=4, n_heads=1
- Serialized to JSON via `config_to_json()`, deserialized back via `config_from_json()`
- Hash identity preserved: `config_hash(original) == config_hash(restored)` (hash=1cc09c7454869e73)
- Cross-parameter validation rejects invalid configs (walk_length < 2*w, corpus_size < 100*n, n_heads != 1, r > walk_length)

### SC2: result.json creation and schema validation
**Status: PASSED**

- `write_result()` creates `results/{experiment_id}/result.json` with all required fields
- Schema validates: schema_version, experiment_id, timestamp, description, tags, config, metrics
- metrics.scalars required and enforced
- Sequence array length consistency checked (token_logprobs/token_entropy vs tokens)
- Optional token_metrics.npz written alongside result.json for large arrays
- `load_result()` validates on load, rejecting invalid files

### SC3: Master seed produces identical sequences across all RNG sources
**Status: PASSED**

- `set_seed(42)` produces identical sequences from:
  - Python `random.random()` (50 values)
  - NumPy `np.random.rand()` (50 values)
  - PyTorch `torch.rand()` (50 values)
- `verify_seed_determinism()` self-test passes for seeds: 42, 123, 7, 0, 999999
- CUDA determinism configured: cudnn.deterministic=True, cudnn.benchmark=False, use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=:4096:8

### SC4: Git short SHA captured in result.json metadata
**Status: PASSED**

- `get_git_hash()` returns short SHA (7+ hex chars)
- Dirty detection works: appends `-dirty` when uncommitted changes exist (both staged and unstaged checked)
- Returns `"unknown"` when not in a git repository
- `write_result()` includes `metadata.code_hash` from live `get_git_hash()` call

## Requirements Traceability

| Requirement | Description | Plan | Status |
|-------------|-------------|------|--------|
| MGMT-01 | Frozen, serializable, hashable experiment config | 01-01 | COMPLETE |
| MGMT-05 | result.json conforming to project schema | 01-01 | COMPLETE |
| TRNG-02 | All random seeds controlled from single master seed | 01-02 | COMPLETE |
| TRNG-07 | Git code hash tracked and stored with results | 01-02 | COMPLETE |

## Test Coverage

- **tests/test_config.py**: 18 tests (defaults, immutability, round-trip, hashing, validation, strict mode)
- **tests/test_results.py**: 15 tests (validation, experiment ID, write_result, load_result, token_metrics)
- **tests/test_reproducibility.py**: 13 tests (seed determinism, worker seeding, git hash, integration)
- **Total: 46 tests, all passing**

## Verification Score

**4/4 must-haves verified. Phase 1 PASSED.**

---
*Verified: 2026-02-24*
