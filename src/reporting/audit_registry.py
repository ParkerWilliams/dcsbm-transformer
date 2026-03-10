"""Audit data registry for the v1.2 Mathematical Audit report.

Contains exactly 28 formula-to-code mapping entries -- one per audited
requirement from Phases 18-22.  Each entry records the requirement ID,
category, human-readable title, KaTeX-compatible LaTeX formula(s), the
primary source-code location, the audit verdict, and (when applicable)
a description of the fix that was applied.

This module is the single source of truth consumed by the HTML report
generator (Phase 23, Plan 02).  It does NOT reuse MATH_SECTIONS from
math_pdf.py -- every formula string is freshly curated for KaTeX display
mode rendering.

Verdict key:
  - "correct"  -- implementation matches the mathematical specification
  - "fixed"    -- implementation was corrected during the audit
  - "concern"  -- issue identified but not yet resolved
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Category display order
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = [
    "Graph",
    "SVD",
    "AUROC",
    "Statistical",
    "Softmax",
    "Null Model",
]

# ---------------------------------------------------------------------------
# 28 audit entries
# ---------------------------------------------------------------------------

AUDIT_ENTRIES: list[dict] = [
    # =======================================================================
    # GRAPH (5 entries: GRAPH-01 .. GRAPH-05)
    # =======================================================================
    {
        "req_id": "GRAPH-01",
        "category": "Graph",
        "title": "DCSBM Edge Probability",
        "latex": (
            r"\[P_{ij} = \theta_i \cdot \theta_j "
            r"\cdot \omega[b_i, b_j]\]"
            "\n"
            r"\[\omega[a,b] = \begin{cases} p_{\text{in}} & a = b \\"
            r" p_{\text{out}} & a \neq b \end{cases}\]"
        ),
        "code_location": "src/graph/dcsbm.py:25",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "GRAPH-02",
        "category": "Graph",
        "title": "Walk Sampling Uniformity",
        "latex": (
            r"\[p(v \mid u) = \frac{1}{\deg^+(u)}, "
            r"\quad v \in \mathcal{N}^+(u)\]"
        ),
        "code_location": "src/walk/generator.py:150",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "GRAPH-03",
        "category": "Graph",
        "title": "Block Jumper Designation",
        "latex": (
            r"\[r = \text{round}(\text{scale} \cdot w), "
            r"\quad \text{scale} \in \{0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0\}\]"
            "\n"
            r"\[r_j = r\text{\_values}[j \bmod |r\text{\_values}|]\]"
        ),
        "code_location": "src/graph/jumpers.py:37",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "GRAPH-04",
        "category": "Graph",
        "title": "Behavioral Classification",
        "latex": (
            r"\[\text{RuleOutcome} \in "
            r"\{\text{UNCONSTRAINED}, \text{PENDING}, "
            r"\text{FOLLOWED}, \text{VIOLATED}\}\]"
            "\n"
            r"\[\text{outcome}[t'] = \begin{cases} "
            r"\text{FOLLOWED} & \text{block}(v) = b^* \\"
            r" \text{VIOLATED} & \text{block}(v) \neq b^* "
            r"\end{cases}\]"
        ),
        "code_location": "src/evaluation/behavioral.py:24",
        "verdict": "fixed",
        "fix_description": (
            "Expanded 3-class RuleOutcome (NOT_APPLICABLE/FOLLOWED/VIOLATED) "
            "to 4-class enum (UNCONSTRAINED/PENDING/FOLLOWED/VIOLATED) with "
            "PENDING countdown labeling"
        ),
    },
    {
        "req_id": "GRAPH-05",
        "category": "Graph",
        "title": "Walk Compliance Rate",
        "latex": (
            r"\[\text{compliance} = "
            r"\frac{n_{\text{followed}}}{n_{\text{constrained}}} "
            r"= 1 - \frac{n_{\text{violated}}}{n_{\text{constrained}}}\]"
        ),
        "code_location": "src/evaluation/behavioral.py:39",
        "verdict": "correct",
        "fix_description": None,
    },
    # =======================================================================
    # SVD (6 entries: SVD-01 .. SVD-06)
    # =======================================================================
    {
        "req_id": "SVD-01",
        "category": "SVD",
        "title": "QK^T Attention Matrix Construction",
        "latex": (
            r"\[\text{QK}^T = \frac{(x W_q)(x W_k)^T}{\sqrt{d_k}}\]"
            "\n"
            r"\[\text{QK}^T_{\text{SVD}} = "
            r"\text{QK}^T \odot M_{\text{causal}}\]"
        ),
        "code_location": "src/model/attention.py:94",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "SVD-02",
        "category": "SVD",
        "title": "WvWo and AVWo Matrix Constructions",
        "latex": (
            r"\[W_v W_o = W_v^{T} \cdot W_o\]"
            "\n"
            r"\[\text{AVWo} = (A \cdot V) \cdot W_o^{T}\]"
        ),
        "code_location": "src/evaluation/pipeline.py:86",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "SVD-03",
        "category": "SVD",
        "title": "Singular-Value-Derived Metrics",
        "latex": (
            r"\[r_{\text{stable}}(M) = "
            r"\frac{\sum_i \sigma_i^2}{\sigma_1^2}, \quad "
            r"H(\sigma) = -\sum_i p_i \log p_i\]"
            "\n"
            r"\[\kappa(M) = \frac{\sigma_1}{\sigma_n + \varepsilon}, \quad "
            r"\rho_1 = \frac{\sqrt{\sum_{i \ge 2} \sigma_i^2}}"
            r"{\sqrt{\sum_i \sigma_i^2}}, \quad "
            r"\text{align} = |u_1 \cdot v_1|\]"
        ),
        "code_location": "src/evaluation/svd_metrics.py:31",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "SVD-04",
        "category": "SVD",
        "title": "Grassmannian Distance (Geodesic)",
        "latex": (
            r"\[d_G(U_{\text{prev}}, U_{\text{curr}}) = "
            r"\sqrt{\sum_{i=1}^{k} \theta_i^2}\]"
            "\n"
            r"\[\cos \theta_i = "
            r"\sigma_i(U_{\text{prev}}^T U_{\text{curr}})\]"
        ),
        "code_location": "src/evaluation/svd_metrics.py:146",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "SVD-05",
        "category": "SVD",
        "title": "Float16 Spectrum Storage",
        "latex": (
            r"\[\text{spectra}[t] = \sigma(M_t) "
            r"\quad \text{stored as float32}\]"
            "\n"
            r"\[\text{float16 error: } "
            r"\text{curvature } 1130\%, "
            r"\text{ torsion } 702 \times 10^6\%\]"
        ),
        "code_location": "src/evaluation/pipeline.py:126",
        "verdict": "fixed",
        "fix_description": (
            "Upgraded spectrum trajectory storage from float16 to float32 "
            "in pipeline.py; float16 produced 1130% curvature error and "
            "702M% torsion error"
        ),
    },
    {
        "req_id": "SVD-06",
        "category": "SVD",
        "title": "Frenet-Serret Curvature and Torsion",
        "latex": (
            r"\[\kappa[t] = \frac{\|T'(t)\|}{\|r'(t)\|}, \quad "
            r"T(t) = \frac{r'(t)}{\|r'(t)\|}\]"
            "\n"
            r"\[\tau[t] = \frac{(r' \times r'') \cdot r'''}"
            r"{\|r' \times r''\|^2}\]"
        ),
        "code_location": "src/analysis/spectrum.py:96",
        "verdict": "correct",
        "fix_description": None,
    },
    # =======================================================================
    # AUROC (4 entries: AUROC-01 .. AUROC-04)
    # =======================================================================
    {
        "req_id": "AUROC-01",
        "category": "AUROC",
        "title": "AUROC Rank-Based Computation",
        "latex": (
            r"\[\text{AUROC} = \frac{U}{n_v \cdot n_c}, \quad "
            r"U = R_v - \frac{n_v(n_v + 1)}{2}\]"
        ),
        "code_location": "src/analysis/auroc_horizon.py:38",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "AUROC-02",
        "category": "AUROC",
        "title": "Lookback Indexing (Fence-Post)",
        "latex": (
            r"\[\text{metric}[j] = "
            r"\text{metric\_array}[:, \text{resolution\_step} - j]\]"
        ),
        "code_location": "src/analysis/auroc_horizon.py:60",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "AUROC-03",
        "category": "AUROC",
        "title": "Predictive Horizon Consistency",
        "latex": (
            r"\[h = \max\{j : \text{AUROC}(j) > \tau\}, "
            r"\quad \tau = 0.75\]"
        ),
        "code_location": "src/analysis/auroc_horizon.py:122",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "AUROC-04",
        "category": "AUROC",
        "title": "Event Extraction Boundaries",
        "latex": (
            r"\[\text{resolution\_step} = t + r\]"
            "\n"
            r"\[\text{exclude if } "
            r"t < \text{last\_violation\_end}\]"
        ),
        "code_location": "src/analysis/event_extraction.py:46",
        "verdict": "correct",
        "fix_description": None,
    },
    # =======================================================================
    # Statistical (6 entries: STAT-01 .. STAT-06)
    # =======================================================================
    {
        "req_id": "STAT-01",
        "category": "Statistical",
        "title": "Shuffle Permutation Null",
        "latex": (
            r"\[p = \frac{1}{N}\sum_{k=1}^{N} "
            r"\mathbf{1}\!\left[\max_j "
            r"\text{AUROC}_k^{\text{shuffle}}(j) \geq "
            r"\max_j \text{AUROC}^{\text{obs}}(j)\right]\]"
        ),
        "code_location": "src/analysis/auroc_horizon.py:145",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "STAT-02",
        "category": "Statistical",
        "title": "Bootstrap BCa Intervals",
        "latex": (
            r"\[\widehat{\text{AUROC}}^{*(b)} = "
            r"\frac{R_v^{*(b)} - n_v(n_v+1)/2}{n_v \cdot n_c}\]"
        ),
        "code_location": "src/analysis/statistical_controls.py:84",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "STAT-03",
        "category": "Statistical",
        "title": "Holm-Bonferroni Correction",
        "latex": (
            r"\[p_{(i)}^{\text{adj}} = \max_{k \leq i}"
            r"\!\left\{\min\!\left("
            r"(m - k + 1) \cdot p_{(k)},\; 1\right)\right\}\]"
        ),
        "code_location": "src/analysis/statistical_controls.py:42",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "STAT-04",
        "category": "Statistical",
        "title": "Cohen's d Effect Size",
        "latex": (
            r"\[d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}, "
            r"\quad s_{\text{pooled}} = \sqrt{\frac{"
            r"(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}\]"
        ),
        "code_location": "src/analysis/statistical_controls.py:149",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "STAT-05",
        "category": "Statistical",
        "title": "Spearman Correlation",
        "latex": (
            r"\[r_s = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}\]"
            "\n"
            r"\[\text{redundant if } |r_s| > 0.9\]"
        ),
        "code_location": "src/analysis/statistical_controls.py:222",
        "verdict": "fixed",
        "fix_description": (
            "Changed measurement mode from Pearson (np.corrcoef) to "
            "Spearman (scipy.stats.spearmanr) per requirement specification"
        ),
    },
    {
        "req_id": "STAT-06",
        "category": "Statistical",
        "title": "Exploratory/Confirmatory Split",
        "latex": (
            r"\[\text{assign\_split}(\text{events}, "
            r"\text{seed}=2026) \to "
            r"\{E, C\}, \; |E| \approx |C|\]"
        ),
        "code_location": "src/evaluation/split.py:21",
        "verdict": "correct",
        "fix_description": None,
    },
    # =======================================================================
    # Softmax (3 entries: SFTX-01 .. SFTX-03)
    # =======================================================================
    {
        "req_id": "SFTX-01",
        "category": "Softmax",
        "title": "LaTeX Derivation Correctness",
        "latex": (
            r"\[\|\Delta(\text{AVWo})\|_F \leq "
            r"\frac{\varepsilon}{2} \|QK^T\|_F "
            r"\|V\|_2 \|W_O\|_2\]"
        ),
        "code_location": "src/analysis/perturbation_bound.py:25",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "SFTX-02",
        "category": "Softmax",
        "title": "Empirical Bound Verification",
        "latex": (
            r"\[\text{ratio} = "
            r"\frac{\|\Delta(\text{AVWo})\|_F}{\text{bound}} < 1.0\]"
            "\n"
            r"\[\sigma_{\text{SV-L2}} \leq "
            r"\|\Delta\|_F \leq \text{bound}\]"
        ),
        "code_location": "src/analysis/perturbation_bound.py:56",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "SFTX-03",
        "category": "Softmax",
        "title": "Bound Assumptions",
        "latex": (
            r"\[\text{Assumptions: causal mask, } "
            r"V \text{ and } W_O \text{ held fixed, single-head}\]"
        ),
        "code_location": "src/analysis/perturbation_bound.py:25",
        "verdict": "correct",
        "fix_description": None,
    },
    # =======================================================================
    # Null Model (4 entries: NULL-01 .. NULL-04)
    # =======================================================================
    {
        "req_id": "NULL-01",
        "category": "Null Model",
        "title": "Grassmannian Drift Code-Path Parity",
        "latex": (
            r"\[\text{null\_model} \xrightarrow{\text{imports}} "
            r"\text{fused\_evaluate}, \; "
            r"\text{extract\_events}, \; "
            r"\text{holm\_bonferroni}\]"
        ),
        "code_location": "src/analysis/null_model.py:453",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "NULL-02",
        "category": "Null Model",
        "title": r"Marchenko-Pastur \sigma^2 Calibration",
        "latex": (
            r"\[f_{MP}(x; \gamma, \sigma^2) = "
            r"\frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}"
            r"{2\pi \gamma \sigma^2 x}\]"
            "\n"
            r"\[E[\lambda_{MP}(\gamma, \sigma^2)] = \sigma^2\]"
        ),
        "code_location": "src/analysis/null_model.py:287",
        "verdict": "fixed",
        "fix_description": (
            "Corrected sigma^2 calibration formula: "
            "E[lambda_MP(gamma, sigma^2)] = sigma^2, "
            "not sigma^2*(1+gamma)"
        ),
    },
    {
        "req_id": "NULL-03",
        "category": "Null Model",
        "title": "Column-Filtered Adjacency",
        "latex": (
            r"\[A_{\text{null}}[:, j] = 0 "
            r"\quad \forall j \in \text{jumper\_vertices}\]"
        ),
        "code_location": "src/analysis/null_model.py:30",
        "verdict": "correct",
        "fix_description": None,
    },
    {
        "req_id": "NULL-04",
        "category": "Null Model",
        "title": "Holm-Bonferroni Family Separation",
        "latex": (
            r"\[\text{families: } \{p_{\text{primary}}\} "
            r"\text{ and } \{p_{\text{null}}\} "
            r"\text{ corrected independently}\]"
        ),
        "code_location": "src/analysis/null_model.py:453",
        "verdict": "correct",
        "fix_description": None,
    },
]


def entries_by_category() -> dict[str, list[dict]]:
    """Group audit entries by category, preserving CATEGORIES order.

    Returns:
        dict mapping category name to list of entry dicts for that
        category.  Keys appear in the same order as CATEGORIES.
    """
    grouped: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
    for entry in AUDIT_ENTRIES:
        grouped[entry["category"]].append(entry)
    return grouped
