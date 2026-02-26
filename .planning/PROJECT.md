# DCSBM Transformer: SVD Predictive Signals for LLM Hallucination

## What This Is

A research framework that uses a degree-corrected stochastic block model (DCSBM) to generate synthetic token sequences with known ground-truth rules, trains a NanoGPT-scale transformer on these sequences, and analyzes whether SVD instability in the QK^T attention matrix predicts rule violations (hallucinations) before they occur. The framework includes a full experiment pipeline from graph generation through training, evaluation, SVD metric collection, and automated reporting.

## Core Value

Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations *before* they happen, and measure how far in advance (predictive horizon) this signal is detectable.

## Current Milestone: v1.1 Journal Feedback

**Goal:** Address convergent reviewer concerns — establish the SVD signal is real (not artifact) through null model baselines, theoretical formalization, multi-head ablation, and rigorous statistical reporting.

**Target features:**
- Null model / baseline Grassmannian drift on clean (no-jumper) sequences
- Softmax filtering bound: LaTeX derivation + empirical verification of QKᵀ → AVWₒ perturbation propagation
- Multi-head ablation (1h, 2h, 4h) with per-head SVD signal concentration analysis
- Precision-recall curves and calibration alongside AUROC
- Pre-registration framework: Grassmannian distance as primary hypothesis, held-out protocol
- Sharp compliance curve: demonstrate r/w transition from near-perfect to failure
- Full spectrum trajectory tracking with curvature/torsion analysis
- SVD computational overhead benchmarks and cheaper approximation candidates

## Requirements

### Validated

<!-- Shipped and confirmed in v1.0 (phases 1-9). -->

- DCSBM graph generation with configurable block structure, in-group/out-group probabilities, and degree correction
- Block jumper vertex designation with configurable jump length r and target block rules
- Walk generation on DCSBM graphs as training corpus with configurable walk length
- NanoGPT-scale transformer with single attention head, configurable d_model, n_layers, and context window
- Training pipeline with cross-entropy next-token prediction
- Training sufficiency gate (edge compliance >95%, rule compliance >80%)
- Behavioral evaluation: edge validity and rule compliance classification (4-class outcome per step)
- Full SVD metric collection from QK^T matrix at every token step (~20 metrics)
- Predictive horizon analysis: AUROC at each lookback distance j for each SVD metric
- Results storage per the project JSON schema (result.json per configuration)
- Automated plotting from result.json (aligned metrics, distributions, confusion matrices, curves)
- HTML report generation (single-experiment and comparison reports)
- Reproducibility (seed control, code hash tracking, config storage)
- Math verification PDF generation for peer review

### Active

<!-- v1.1 scope — addressing journal reviewer feedback. -->

- [ ] Null model baseline: Grassmannian drift distribution on clean sequences (no block jumpers)
- [ ] Softmax filtering bound: theoretical ε-bound on AVWₒ spectral change given QKᵀ perturbation ε, with LaTeX derivation and empirical verification
- [ ] Multi-head transformer support (2h, 4h) with per-head SVD extraction and signal concentration analysis
- [ ] Precision-recall curves and calibration (reliability diagrams) for violation prediction
- [ ] Pre-registration framework: Grassmannian distance of QKᵀ as primary hypothesis, held-out evaluation protocol
- [ ] Sharp compliance curve: r/w sweep showing near-perfect compliance at r ≪ w, degradation at r → w, failure at r ≫ w
- [ ] Full spectrum trajectory: store σ₁...σₖ vectors, compute spectral curve curvature and torsion
- [ ] SVD computational overhead: timing benchmarks, cost analysis, cheaper approximation candidates

### Out of Scope

- Multi-head beyond 4 heads — 1h/2h/4h ablation addresses reviewer concern; full multi-head is future work
- Clinical or real-world LLM data — this is a synthetic controlled environment study
- Real-time inference or deployment — this is a research framework only
- GPU cluster orchestration — runs on single RunPod instances within $100 budget

## Context

- **Research initiative:** AI Health Research, Hospital Mathematics Division
- **Core hypothesis:** SVD instability in QK^T precedes rule violations, and the predictive horizon depends on the ratio r/w (jump length to context window)
- **Expected key finding:** Hallucination rate increases monotonically with r, with a step change as r crosses w; predictive horizon (AUROC > 0.75) should be measurable at j > 1 steps before failure
- **Anchor configuration:** n=500, w=64, t=200k, d_model=128, n_layers=4, 1 attention head — all other sweeps vary around this
- **Compute budget:** $100 on RunPod (RTX 3090 or RTX 4090)
- **Framework:** GSD (Get Shit Done) for project scaffolding
- **Stack:** Python 3.11+, PyTorch, NanoGPT-scale architecture

## Constraints

- **Single attention head (v1.0):** Essential for unambiguous QK^T analysis — v1.1 adds 2h/4h ablation to test generalization
- **Compute budget:** $100 on RunPod — must prioritize scientifically critical configs first
- **Training sufficiency gate:** Configs that don't reach edge >95% / rule >80% compliance are excluded from SVD analysis
- **SVD performance:** O(d^3) per step — must use torch.linalg.svd with full_matrices=False and batching
- **Reproducibility:** Every run must be reproducible from stored config + code hash + seed
- **Venv requirement:** All Python commands run in a virtual environment
- **Walk corpus size:** Must be at least 2 orders of magnitude larger than n

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Single attention head only | Keeps QK^T analysis unambiguous and interpretable | — Pending |
| DCSBM for synthetic data | Provides known ground truth with tunable difficulty via block structure | — Pending |
| Block jumper rules as hallucination proxy | Creates predictable failure events at known positions for alignment analysis | — Pending |
| Anchor config first, then sweep | Calibrates wall time and validates pipeline before spending budget | — Pending |
| NanoGPT-scale architecture | Small enough for budget, large enough for meaningful attention patterns | — Pending |
| Job queue with priority ordering | Budget can be cut at any point without losing core results | — Pending |

| Multi-head ablation (1h/2h/4h) | Reviewer concern: single-head multiplexing may be artifact | — Pending |
| Null model baseline | All 3 reviewers: need baseline drift to validate signal | — Pending |
| Softmax filtering bound | Mathematician: derive theoretical lag prediction | — Pending |

---
*Last updated: 2026-02-26 after milestone v1.1 Journal Feedback started*
