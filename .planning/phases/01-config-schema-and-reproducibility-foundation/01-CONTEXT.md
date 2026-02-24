# Phase 1: Config, Schema, and Reproducibility Foundation - Context

**Gathered:** 2026-02-24
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the experiment configuration system, result.json schema with validation, seed management, and git hash tracking. This is pure infrastructure — no graph generation, no model code, no training. Everything downstream depends on these foundations being frozen and correct.

</domain>

<decisions>
## Implementation Decisions

### Config Structure
- Nested dataclasses: GraphConfig, ModelConfig, TrainingConfig, SweepConfig (and a top-level ExperimentConfig composing them)
- Typed, validated, IDE-friendly with frozen=True for immutability
- Strict validation with cross-parameter constraints (walk_length >= 2*w, corpus_size >= 100*n, n_heads == 1)
- Config hashing: all params except seed — graph cache shared across seeds, model/training cache includes seed
- Serialization: JSON format (matches result.json, easy to diff)

### Result Storage Format
- Hybrid JSON + npz: result.json holds summary metrics, scalars, curves, statistical tests, and sequence metadata; token_metrics.npz holds full per-step SVD metric arrays referenced by sequence_id
- Python validation function checks required fields, types, and array length consistency before writing — not a formal JSON Schema file
- All evaluation walks containing block jumper events get full sequence data in result.json (not sampled)
- Experiment ID format: key params in slug — e.g., n500_w64_r32_d128_L4_s42_{timestamp} — scannable at a glance in file listings

### Anchor Config Defaults (LOCKED)
- n = 500 (vertices)
- w = 64 (context window)
- t = 200,000 (training corpus walks)
- d_model = 128
- n_layers = 4
- n_heads = 1 (non-negotiable)
- r = 0.9w = 57 (jump length, rounded)
- walk_length = 4w = 256
- DCSBM: K, p_in, p_out per spec defaults
- 3 random seeds per config

### Makefile and CLI
- Makefile as primary entry point: `make run` for anchor config, `make sweep` for full grid, `make test`, `make pdf`
- Python script (`run_experiment.py`) with argparse for custom invocations: `python run_experiment.py --config config.json`
- All Python commands run in venv

### Claude's Discretion
- Exact dataclass field names and type annotations
- Validation error message formatting
- pyproject.toml structure and dependency pinning
- Makefile target organization beyond the required ones
- Internal helper functions for serialization/deserialization

</decisions>

<specifics>
## Specific Ideas

- Experiment IDs should be scannable in `ls` output — key params encoded in the slug so you can tell what a run was without opening it
- Config must round-trip perfectly: instantiate -> serialize to JSON -> deserialize -> hash matches original
- The schema validation function is a Python function (not jsonschema), kept simple and maintainable

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-config-schema-and-reproducibility-foundation*
*Context gathered: 2026-02-24*
