---
phase: 04-transformer-model
plan: 01
subsystem: model
tags: [pytorch, transformer, nanogpt, attention, svd-extraction, strenum]

# Dependency graph
requires:
  - phase: 01-project-setup
    provides: ExperimentConfig with GraphConfig, ModelConfig, TrainingConfig
provides:
  - TransformerLM single-head causal transformer with 4 extraction modes
  - ExtractionMode StrEnum (NONE, SVD_TARGETS, RESIDUAL, FULL)
  - ForwardOutput dataclass with logits and optional extraction tensors
  - CausalSelfAttention with manual QK^T computation and zero-fill masking
  - get_wvwo() method returning stacked WvWo weight product per layer
  - create_model factory from ExperimentConfig
affects: [05-training-loop, 06-svd-metrics, 07-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns: [pre-norm transformer, manual attention with dual masking, StrEnum extraction modes, GPT-2 weight initialization]

key-files:
  created:
    - src/model/__init__.py
    - src/model/types.py
    - src/model/attention.py
    - src/model/block.py
    - src/model/transformer.py
    - tests/test_model.py
  modified: []

key-decisions:
  - "Dual masking convention: zero-fill for SVD QK^T target, -inf for softmax attention"
  - "Separate W_q/W_k/W_v/W_o linear layers (not fused) for extraction clarity"
  - "Residual stream includes pre-block embedding state as index 0 (n_layers+1 total states)"
  - "WvWo computed as W_v.weight.T @ W_o.weight per layer (nn.Linear convention)"

patterns-established:
  - "ExtractionMode enum controls forward pass output via mode parameter"
  - "All extracted tensors are .detach()ed -- no gradient flow through extraction"
  - "Pre-norm transformer blocks: LN before attention, LN before MLP"
  - "GPT-2 init: Normal(0, 0.02) with 1/sqrt(2*n_layers) residual projection scaling"

requirements-completed: [MODL-01, MODL-02, MODL-03]

# Metrics
duration: 4min
completed: 2026-02-25
---

# Phase 4 Plan 1: Transformer Model Summary

**NanoGPT-scale single-head causal transformer with 4-mode QK^T/attention/residual extraction for SVD stability analysis**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-25T05:32:04Z
- **Completed:** 2026-02-25T05:36:14Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Complete `src/model/` package with types, attention, block, and transformer modules
- Manual single-head attention with dual masking: zero-fill for SVD targets, -inf for softmax
- 4 extraction modes (NONE, SVD_TARGETS, RESIDUAL, FULL) controlled by StrEnum parameter
- get_wvwo() convenience method returning stacked [n_layers, d_model, d_model] weight products
- 27 comprehensive tests covering all 3 requirements (MODL-01, MODL-02, MODL-03)
- Full suite 140 tests pass with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Types, Attention, and Block modules** - `1de3e25` (feat)
2. **Task 2: TransformerLM, factory function, and comprehensive tests** - `b1c2788` (feat)

## Files Created/Modified
- `src/model/types.py` - ExtractionMode StrEnum, AttentionInternals dataclass, ForwardOutput dataclass
- `src/model/attention.py` - CausalSelfAttention with manual QK^T and optional extraction
- `src/model/block.py` - TransformerBlock with pre-norm, GELU MLP, residual connections
- `src/model/transformer.py` - TransformerLM full model with embeddings, blocks, extraction, factory
- `src/model/__init__.py` - Public API exports
- `tests/test_model.py` - 27 tests across MODL-01, MODL-02, MODL-03, and integration

## Decisions Made
- Dual masking convention: QK^T SVD target uses zero-fill for clean SVD input; softmax uses -inf for proper probability distribution. Same raw QK^T matrix, two different mask applications.
- Separate W_q/W_k/W_v/W_o nn.Linear layers instead of fused 3*d_model for extraction transparency.
- Residual stream collects n_layers+1 states (pre-block embedding output plus each block's output) to capture the full residual evolution.
- WvWo computed as W_v.weight.T @ W_o.weight following nn.Linear [out, in] storage convention.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- TransformerLM is ready for Phase 5 (Training Loop) integration via `create_model(config)`
- All extraction modes verified for Phase 6 (SVD Metrics) consumption
- Wv and Wo are accessible as named parameters per layer at `block.attention.W_v` and `block.attention.W_o`
- No blockers identified

## Self-Check: PASSED

All 6 created files verified on disk. Both task commits (1de3e25, b1c2788) found in git log.

---
*Phase: 04-transformer-model*
*Completed: 2026-02-25*
