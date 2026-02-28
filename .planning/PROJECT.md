# DCSBM Transformer: SVD Predictive Signals for LLM Hallucination

## What This Is

A research framework that uses a degree-corrected stochastic block model (DCSBM) to generate synthetic token sequences with known ground-truth rules, trains a NanoGPT-scale transformer on these sequences, and analyzes whether SVD instability in the QK^T attention matrix predicts rule violations (hallucinations) before they occur. The framework includes a full experiment pipeline from graph generation through training, evaluation, SVD metric collection, null model validation, multi-head ablation, and automated HTML reporting.

## Core Value

Determine whether SVD instability metrics from the QK^T attention matrix can predict transformer rule violations *before* they happen, and measure how far in advance (predictive horizon) this signal is detectable.

## Requirements

### Validated

- ✓ DCSBM graph generation with configurable block structure, degree correction — v1.0
- ✓ Block jumper vertex designation with configurable jump length r and target block rules — v1.0
- ✓ Walk generation on DCSBM graphs as training corpus — v1.0
- ✓ NanoGPT-scale transformer with configurable d_model, n_layers, context window — v1.0
- ✓ Training pipeline with cross-entropy next-token prediction and sufficiency gate — v1.0
- ✓ Behavioral evaluation: 4-class outcome classification per step — v1.0
- ✓ Full SVD metric collection from QK^T at every token step (~20 metrics) — v1.0
- ✓ Predictive horizon analysis: AUROC at each lookback distance — v1.0
- ✓ Results storage per JSON schema, automated plotting, HTML reporting — v1.0
- ✓ Reproducibility (seed control, code hash tracking) — v1.0
- ✓ Math verification PDF generation — v1.0
- ✓ Pre-registration framework: locked hypothesis, held-out split, deviation log — v1.1
- ✓ Null model baseline: Grassmannian drift on jumper-free sequences with MW-U comparison — v1.1
- ✓ Precision-recall curves and calibration diagnostics (ECE, reliability diagrams) — v1.1
- ✓ SVD computational overhead benchmarks (full vs randomized vs values-only) — v1.1
- ✓ Softmax filtering bound: LaTeX derivation + empirical verification — v1.1
- ✓ Full spectrum trajectory with Frenet-Serret curvature/torsion — v1.1
- ✓ Sharp compliance curve: r/w sweep with dual-axis publication figure — v1.1
- ✓ Multi-head ablation (1h/2h/4h) with per-head SVD and signal concentration — v1.1
- ✓ E2E pipeline: run_experiment.py chains all stages from config to report — v1.1

### Active

- [ ] Parameter sweep infrastructure with declarative ranges and priority queue
- [ ] Multi-seed runs (3 seeds per configuration)
- [ ] Sweep state persistence for RunPod preemption resume
- [ ] Grassmannian trajectory visualization (low-dimensional embedding)

### Out of Scope

- Multi-head beyond 4 heads — 1h/2h/4h ablation demonstrates the principle
- Clinical or real-world LLM data — synthetic controlled environment study
- Real-time inference or deployment — research framework only
- GPU cluster orchestration — single RunPod instances within $100 budget
- Full Bayesian calibration (Platt scaling) — undermines pre-registration claims
- Symbolic math verification (SymPy) — manual LaTeX sufficient for one derivation

## Context

Shipped v1.1 with 23,652 LOC Python across 111 files.
Tech stack: Python 3.11+, PyTorch, NanoGPT-scale architecture, scipy, sklearn, matplotlib/seaborn.
536+ tests passing, 0 failures.

**Research status:** All reviewer concerns addressed:
- Null model validates signal is real (not artifact)
- Softmax bound formalizes theoretical lag prediction
- Multi-head ablation tests single-head multiplexing concern
- Pre-registration locks hypothesis before confirmatory analysis
- Calibration and PR curves complement AUROC

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Single attention head (v1.0) | Keeps QK^T analysis unambiguous | ✓ Good — extended to multi-head in v1.1 |
| DCSBM for synthetic data | Known ground truth with tunable difficulty | ✓ Good |
| Block jumper rules as hallucination proxy | Predictable failure events at known positions | ✓ Good |
| Anchor config first, then sweep | Calibrates wall time before spending budget | ✓ Good |
| NanoGPT-scale architecture | Small enough for budget, meaningful attention patterns | ✓ Good |
| Pre-registration before analysis | Methodological requirement, cannot be backdated | ✓ Good |
| Null model before enrichment | If null fails, project pivots; validate signal first | ✓ Good — signal validated |
| d_k constant across ablation (128) | d_model scales with n_heads for fair comparison | ✓ Good |
| Multi-head ablation last | Most invasive change; validate on single-head first | ✓ Good |
| Column-filtered adjacency for null walks | 100% jumper-free without overgeneration | ✓ Good |
| Separate Holm-Bonferroni family for null model | Don't mix with primary metrics | ✓ Good |
| Curvature/torsion as exploratory only | Numerically delicate on noisy SVD output | ⚠️ Revisit — float16 quantization concern |

## Constraints

- **Compute budget:** $100 on RunPod — must prioritize scientifically critical configs first
- **Training sufficiency gate:** Configs that don't reach edge >95% / rule >80% compliance are excluded
- **SVD performance:** O(d^3) per step — must use batching and consider randomized alternatives
- **Reproducibility:** Every run must be reproducible from stored config + code hash + seed
- **Venv requirement:** All Python commands run in a virtual environment
- **Walk corpus size:** Must be at least 2 orders of magnitude larger than n

---
*Last updated: 2026-02-28 after v1.1 milestone*
