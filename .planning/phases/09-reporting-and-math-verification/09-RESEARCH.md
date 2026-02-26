# Phase 9: Reporting and Math Verification - Research

**Researched:** 2026-02-26
**Domain:** HTML report generation (Jinja2), PDF generation (LaTeX/pdflatex), base64 image embedding
**Confidence:** HIGH (HTML reports), MEDIUM (Math PDF -- pdflatex not currently installed)

## Summary

Phase 9 requires generating two types of self-contained HTML reports (single-experiment and multi-experiment comparison) and a math verification PDF. The HTML reports embed all figures as base64 data URIs and include reproduction blocks. The math PDF documents all implemented mathematics with code alongside LaTeX formulas.

The HTML report stack is straightforward: Jinja2 3.1.2 is already installed, base64 encoding of PNG figures works correctly, and the existing `render.py` orchestrator plus `load_result_data()` provide a clean data loading pattern. The comparison report requires loading multiple result.json files and computing diffs. Jinja2 templates with `autoescape` and `lstrip_blocks` provide clean, maintainable template code.

The math PDF is more constrained. The user locked "pdflatex as the compilation toolchain" in CONTEXT.md. However, pdflatex is **not currently installed** on the system, and disk space is tight (303MB free on /). texlive-latex-base alone requires ~160MB+, and with fonts it needs more. The implementation must generate .tex source files via Jinja2 with LaTeX-safe delimiters, then compile with pdflatex. A prerequisite step to install texlive packages is required.

**Primary recommendation:** Implement HTML reports first (Plans 09-01, 09-02) using Jinja2 + base64 embedding, then tackle the math PDF (Plan 09-03) using Jinja2 with LaTeX-safe delimiters + subprocess pdflatex, with texlive installation as a prerequisite.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Clean academic visual style -- minimal styling, serif fonts, paper-like feel (journal supplementary page aesthetic)
- Full-width figures, one per row -- no side-by-side grids
- Configuration displayed as structured HTML tables grouped by category (model, training, data) -- not raw YAML/JSON
- Sections with no data shown as placeholders with "Not available" note -- don't hide missing sections
- Per-metric rows with sparklines (mini bar charts) showing relative values across experiments
- Curve overlays shown as grid of individual subplots with aligned axes -- not overlaid on single plot
- Config diff shown as full table with all parameters, rows where values differ highlighted
- Include auto-generated summary verdict section (e.g., "Experiment B outperforms A on 4/5 metrics")
- All source files included in math PDF -- math-heavy files get full LaTeX treatment, others listed in appendix
- Sequential layout: code block first, then LaTeX math below with annotations linking math to code lines
- Thorough plain-language summaries -- paragraph per file explaining what it computes, why it matters, how it connects to overall method
- Final implemented equations only -- no derivation steps
- AI-generated disclaimer as standard footnote on title page (not a prominent warning box)
- pdflatex as the compilation toolchain
- Appendix lists non-math files with filename + one-liner description only
- Symbols defined in context where they first appear -- no separate notation/symbol glossary table
- Minimal reproduction block content: git checkout command + full CLI arguments + random seed
- Copy-pasteable code block with copy button in HTML reports
- Git commit hash captured at experiment runtime (not at report generation time)
- Dirty repo state flagged with warning if uncommitted changes existed at runtime

### Claude's Discretion
- Report generation CLI design (separate command vs automatic, flags)
- Jinja2 template structure and customization points
- Statistical test display formatting (p-values, confidence intervals)
- Sequence analysis section content and visualization approach
- Exact typography and spacing within academic style
- Loading/error state handling in reports

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REPT-01 | Self-contained single-experiment HTML report with base64-embedded figures, covering: header, configuration, scalar metrics, curves, confusion matrix, statistical tests, sequence analysis, and reproduction command | Jinja2 + base64 embedding pattern verified; existing `load_result_data()` provides all data; `render.py` generates figures to embed |
| REPT-02 | Comparison HTML report across multiple experiments with: scalar metrics comparison table, curve overlays, config diff table, and aligned sequence plot overlays | Multiple `load_result_data()` calls + dataclass-based diff computation; Jinja2 template with sparkline generation via inline SVG or matplotlib mini-charts |
| REPT-03 | Every report includes a reproduction block with git checkout command and full CLI arguments | `result.json` already stores `metadata.code_hash` from `get_git_hash()`; dirty detection already implemented |
| MATH-01 | Peer-review PDF containing: title page, table of contents, one section per math-heavy source file with plain-language summary, full code block, LaTeX representation of implemented mathematics, and appendix listing all other source files | Jinja2 with LaTeX-safe delimiters generates .tex; pdflatex compiles to PDF; 15 math-heavy source files identified |
| MATH-02 | PDF title page notes clearly that LaTeX was AI-generated and requires researcher sign-off | Standard footnote on title page per CONTEXT.md decision |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Jinja2 | 3.1.2 | Template engine for both HTML and LaTeX | Already installed; standard Python templating; supports custom delimiters for LaTeX |
| base64 (stdlib) | -- | Encode PNG/SVG figures as data URIs | Self-contained HTML with no external file dependencies |
| pathlib (stdlib) | -- | File path handling | Already used throughout project |
| json (stdlib) | -- | Load result.json files | Already used throughout project |
| subprocess (stdlib) | -- | Run pdflatex for PDF compilation | Standard approach for calling external compilers |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| matplotlib | 3.10.8 | Generate sparklines/mini-charts for comparison report | Already installed; inline SVG or base64 PNG for sparklines |
| numpy | 2.3.5 | Load token_metrics.npz, compute comparison stats | Already installed; used by existing data loading |
| texlive-latex-base | system | pdflatex compiler for math PDF | Must be installed via apt; user locked pdflatex as toolchain |
| texlive-fonts-recommended | system | Standard LaTeX fonts | Needed for pdflatex compilation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pdflatex | fpdf2/reportlab | User locked pdflatex; alternatives can't render LaTeX math natively |
| Jinja2 for LaTeX | PyLaTeX | PyLaTeX adds API complexity for simple template rendering; Jinja2 is already installed and more flexible |
| Inline SVG sparklines | Matplotlib mini-charts as base64 | SVG sparklines are lighter weight but matplotlib provides consistent styling with existing figures |

**Installation:**
```bash
# HTML reports need nothing new (Jinja2 already installed)
# Math PDF requires:
apt-get update && apt-get install -y --no-install-recommends texlive-latex-base texlive-fonts-recommended texlive-latex-extra
# Note: ~300MB disk needed; current free space is ~303MB -- tight but feasible
```

## Architecture Patterns

### Recommended Project Structure
```
src/
  reporting/
    __init__.py           # Module exports
    templates/            # Jinja2 template directory
      single_report.html  # Single-experiment HTML template
      comparison_report.html  # Multi-experiment HTML template
      math_verification.tex   # LaTeX template for math PDF
    single.py             # Single-experiment report generator
    comparison.py         # Multi-experiment comparison generator
    math_pdf.py           # Math verification PDF generator
    embed.py              # Base64 encoding utilities
    reproduction.py       # Reproduction block builder
```

### Pattern 1: Jinja2 Environment Setup for HTML Reports
**What:** Configure Jinja2 with PackageLoader, autoescape, and strip settings
**When to use:** All HTML report generation
**Example:**
```python
# Source: Verified with Jinja2 3.1.2 in project venv
import jinja2

def create_html_env() -> jinja2.Environment:
    """Create Jinja2 environment for HTML templates."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("src/reporting/templates"),
        autoescape=jinja2.select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
```

### Pattern 2: Jinja2 Environment for LaTeX Templates
**What:** Configure Jinja2 with LaTeX-safe delimiters that don't conflict with TeX commands
**When to use:** Math verification PDF generation
**Example:**
```python
# Source: Verified working with Jinja2 3.1.2 in project venv
import jinja2

def create_latex_env() -> jinja2.Environment:
    """Create Jinja2 environment with LaTeX-safe delimiters."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader("src/reporting/templates"),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        line_statement_prefix="%%",
        line_comment_prefix="%#",
        trim_blocks=True,
        autoescape=False,
    )
```

### Pattern 3: Base64 Figure Embedding
**What:** Read PNG files, encode as base64, embed as data URI in HTML
**When to use:** All HTML figure embedding
**Example:**
```python
# Source: Verified working in project venv
import base64
from pathlib import Path

def embed_figure(fig_path: Path) -> str:
    """Encode a figure file as a base64 data URI string."""
    suffix = fig_path.suffix.lstrip(".")
    mime = {"png": "image/png", "svg": "image/svg+xml"}.get(suffix, "image/png")
    data = fig_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"
```

### Pattern 4: Config Diff Computation
**What:** Compare two experiment configs and identify differing parameters
**When to use:** Comparison report config diff table
**Example:**
```python
from dataclasses import asdict

def compute_config_diff(configs: list[dict]) -> list[dict]:
    """Compare configs, return rows with param, values, and differs flag."""
    if not configs:
        return []
    # Flatten nested config dicts
    all_keys = set()
    flat_configs = []
    for cfg in configs:
        flat = _flatten_dict(cfg)
        flat_configs.append(flat)
        all_keys.update(flat.keys())

    rows = []
    for key in sorted(all_keys):
        values = [fc.get(key, "N/A") for fc in flat_configs]
        differs = len(set(str(v) for v in values)) > 1
        rows.append({"param": key, "values": values, "differs": differs})
    return rows
```

### Pattern 5: Reproduction Block from result.json
**What:** Extract git hash, config, and seed to build a copy-pasteable reproduction command
**When to use:** Every report (REPT-03)
**Example:**
```python
def build_reproduction_block(result: dict) -> dict:
    """Build reproduction command from result.json metadata."""
    code_hash = result.get("metadata", {}).get("code_hash", "unknown")
    config = result.get("config", {})
    seed = config.get("seed", 42)

    is_dirty = code_hash.endswith("-dirty")
    clean_hash = code_hash.replace("-dirty", "")

    checkout_cmd = f"git checkout {clean_hash}"
    # Build CLI args from config (adjust to actual CLI structure)
    cli_args = f"python run_experiment.py --seed {seed}"

    return {
        "checkout_cmd": checkout_cmd,
        "cli_args": cli_args,
        "seed": seed,
        "is_dirty": is_dirty,
        "dirty_warning": "WARNING: Uncommitted changes existed at experiment runtime" if is_dirty else None,
    }
```

### Anti-Patterns to Avoid
- **External image references in HTML:** Never use `<img src="path/to/file.png">`. Always base64-embed. The report must be self-contained (single HTML file).
- **Raw JSON/YAML config display:** User explicitly decided against this. Display as structured HTML tables grouped by category.
- **Hiding missing sections:** Show placeholder with "Not available" note. Do not conditionally hide entire sections.
- **Overlaying curves on single plot for comparison:** User explicitly decided on grid of individual subplots with aligned axes.
- **LaTeX derivation steps:** User decided "final implemented equations only." No intermediate derivation.
- **Prominent AI-generated warning box:** User decided "standard footnote on title page." Keep it subtle.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Template rendering | Custom string interpolation | Jinja2 Environment | Handles escaping, conditionals, loops, includes, filters |
| Base64 encoding | Manual byte manipulation | `base64.b64encode()` | Standard library, handles padding correctly |
| Config comparison | Manual nested dict walking | `dataclasses.asdict()` + flatten utility | Existing serialization handles nested frozen dataclasses |
| LaTeX compilation | Custom TeX parsing | `subprocess.run(["pdflatex", ...])` | pdflatex is the standard; user locked this choice |
| Copy button in HTML | Complex JS clipboard API | Simple `navigator.clipboard.writeText()` with fallback | Works in all modern browsers; 5 lines of JS |
| Sparklines | D3.js or chart library | Matplotlib mini-figures or inline SVG paths | Consistent with project style; no new JS dependencies |

**Key insight:** The entire reporting layer is a data-formatting exercise. All data already exists in result.json and token_metrics.npz. The visualization module already generates all needed figures. The reporting layer reads existing data and renders it through templates. No new computation logic is needed.

## Common Pitfalls

### Pitfall 1: Jinja2 Delimiter Conflicts with LaTeX
**What goes wrong:** Default `{{ }}` and `{% %}` delimiters clash with LaTeX braces everywhere
**Why it happens:** LaTeX uses `{` and `}` pervasively for grouping arguments
**How to avoid:** Use LaTeX-safe delimiters as shown in Pattern 2: `\BLOCK{}`, `\VAR{}`, `\#{}`. This is a well-established pattern.
**Warning signs:** Template rendering errors mentioning unexpected `{` tokens

### Pitfall 2: Large Base64 Strings Slowing Browser
**What goes wrong:** Each 300 DPI PNG figure encodes to ~50-200KB base64 (70-280KB in HTML). With 10+ figures, the HTML can reach several MB.
**Why it happens:** base64 encoding adds ~33% overhead; 300 DPI PNGs are large
**How to avoid:** Use moderate DPI for report-embedded figures (150 DPI is fine for screen viewing). Keep the 300 DPI originals in the figures/ directory. Alternatively, embed SVG where possible (smaller, scalable).
**Warning signs:** HTML file > 5MB, slow browser rendering

### Pitfall 3: pdflatex Not Installed / Compilation Failure
**What goes wrong:** `subprocess.run(["pdflatex", ...])` fails with FileNotFoundError
**Why it happens:** pdflatex is not currently installed on the system. Disk space is tight (~303MB free).
**How to avoid:** Math PDF generation must check for pdflatex availability first and provide a clear error message. The installation step should be documented. Consider generating the .tex file even when pdflatex is unavailable (the .tex source is still useful for review).
**Warning signs:** FileNotFoundError from subprocess; "pdflatex not found" in PATH

### Pitfall 4: Missing Data Sections in Report
**What goes wrong:** Template crashes on `result["metrics"]["predictive_horizon"]["by_r_value"]` when key doesn't exist
**Why it happens:** Not all experiments will have all data (e.g., gate-failed experiments have no SVD/AUROC data)
**How to avoid:** Use Jinja2 `{% if %}` conditionals and `|default()` filters extensively. Every section should gracefully degrade to "Not available" placeholder. Access nested dicts with `.get()` in Python before passing to template.
**Warning signs:** KeyError or UndefinedError in template rendering

### Pitfall 5: Unicode/Special Characters in LaTeX
**What goes wrong:** Underscores, percent signs, ampersands in source code break LaTeX compilation
**Why it happens:** These are LaTeX special characters: `_ % & # $ { } ~ ^ \`
**How to avoid:** Code blocks must be inside `\begin{verbatim}...\end{verbatim}` or `lstlisting` environment. Variable names in prose must use `\_` escaping or `\texttt{}`. Use a Jinja2 filter for LaTeX-escaping text.
**Warning signs:** pdflatex compilation errors about "Missing $ inserted" or "Undefined control sequence"

### Pitfall 6: Comparison Report with Incompatible Experiments
**What goes wrong:** Comparing experiments with different graph sizes, different walk lengths, etc. produces meaningless metric comparisons
**Why it happens:** The comparison report assumes experiments are comparable
**How to avoid:** Include config diff prominently. Highlight any structural differences (different n, K, w). Auto-generate warnings when non-trivial config differences exist.
**Warning signs:** Misleading metric comparisons where one experiment has fundamentally different parameters

## Code Examples

### Existing Data Loading (from src/visualization/render.py)
```python
# Source: src/visualization/render.py lines 19-52
def load_result_data(result_dir: str | Path) -> dict[str, Any]:
    """Load result.json and token_metrics.npz from an experiment directory."""
    result_dir = Path(result_dir)
    result_path = result_dir / "result.json"
    with open(result_path) as f:
        result = json.load(f)
    npz_path = result_dir / "token_metrics.npz"
    metric_arrays = {}
    if npz_path.exists():
        npz = np.load(str(npz_path), allow_pickle=False)
        metric_arrays = dict(npz)
    curves = result.get("metrics", {}).get("curves", {})
    return {"result": result, "metric_arrays": metric_arrays, "curves": curves}
```

### Existing Git Hash (from src/reproducibility/git_hash.py)
```python
# Source: src/reproducibility/git_hash.py
# Already tracks: short SHA, dirty detection (staged + unstaged)
# Returns: "a3f9c1d" (clean) or "a3f9c1d-dirty" (uncommitted changes)
# Result.json stores this in: result["metadata"]["code_hash"]
```

### HTML Template Structure (verified Jinja2 pattern)
```html
{# src/reporting/templates/single_report.html #}
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{{ title }}</title>
    <style>
        /* Academic style: serif fonts, paper-like feel */
        body { font-family: "Georgia", "Times New Roman", serif;
               max-width: 900px; margin: 0 auto; padding: 2rem;
               color: #333; line-height: 1.6; }
        table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
        th, td { border: 1px solid #ccc; padding: 0.5rem; text-align: left; }
        th { background: #f5f5f5; }
        .figure { width: 100%; margin: 1.5rem 0; }
        .figure img { width: 100%; }
        .not-available { color: #999; font-style: italic; }
        .repro-block { background: #f8f8f8; padding: 1rem;
                       border-left: 3px solid #4a90d9; position: relative; }
        .copy-btn { position: absolute; top: 0.5rem; right: 0.5rem;
                    cursor: pointer; background: #e0e0e0; border: none;
                    padding: 0.25rem 0.5rem; border-radius: 3px; }
        .dirty-warning { color: #cc6600; font-weight: bold; }
        .diff-highlight { background: #fff3cd; }
    </style>
</head>
<body>
    <h1>{{ header.experiment_id }}</h1>
    <p class="timestamp">Generated: {{ header.timestamp }}</p>

    {# Configuration Section #}
    <h2>Configuration</h2>
    {% for category, params in config_tables.items() %}
    <h3>{{ category }}</h3>
    <table>
        <tr><th>Parameter</th><th>Value</th></tr>
        {% for param in params %}
        <tr><td>{{ param.name }}</td><td>{{ param.value }}</td></tr>
        {% endfor %}
    </table>
    {% endfor %}

    {# Figures: full-width, one per row #}
    {% for fig in figures %}
    <div class="figure">
        <h3>{{ fig.title }}</h3>
        {% if fig.data_uri %}
        <img src="{{ fig.data_uri }}" alt="{{ fig.title }}">
        {% else %}
        <p class="not-available">Not available</p>
        {% endif %}
    </div>
    {% endfor %}

    {# Reproduction Block #}
    <h2>Reproduction</h2>
    <div class="repro-block">
        {% if repro.is_dirty %}
        <p class="dirty-warning">WARNING: Uncommitted changes existed at experiment runtime</p>
        {% endif %}
        <pre id="repro-cmd">{{ repro.checkout_cmd }}
{{ repro.cli_args }}</pre>
        <button class="copy-btn" onclick="copyRepro()">Copy</button>
    </div>

    <script>
    function copyRepro() {
        const text = document.getElementById('repro-cmd').textContent;
        navigator.clipboard.writeText(text).catch(() => {
            // Fallback for older browsers
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        });
    }
    </script>
</body>
</html>
```

### LaTeX Template Structure (verified Jinja2 LaTeX-safe pattern)
```latex
%% src/reporting/templates/math_verification.tex
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{listings}
\usepackage{hyperref}
\usepackage{geometry}
\geometry{margin=1in}

\lstset{
  language=Python,
  basicstyle=\ttfamily\small,
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\tiny\color{gray},
}

\title{Mathematical Implementation Verification\\
\large DCSBM Transformer SVD Hallucination Prediction}
\author{Auto-generated for peer review}
\date{\VAR{generation_date}}

\begin{document}
\maketitle
\renewcommand{\thefootnote}{\fnsymbol{footnote}}
\footnotetext[1]{This document was generated by AI (Claude).
All LaTeX representations of mathematics require researcher sign-off
before citation or publication.}
\renewcommand{\thefootnote}{\arabic{footnote}}

\tableofcontents
\newpage

\BLOCK{for section in math_sections}
\section{\VAR{section.title}}
\VAR{section.summary}

\subsection{Source Code}
\begin{lstlisting}
\VAR{section.code}
\end{lstlisting}

\subsection{Mathematical Formulation}
\VAR{section.latex_math}

\newpage
\BLOCK{endfor}

\appendix
\section{Non-Mathematical Source Files}
\begin{itemize}
\BLOCK{for entry in appendix_files}
\item \texttt{\VAR{entry.filename}} --- \VAR{entry.description}
\BLOCK{endfor}
\end{itemize}

\end{document}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| External image files in HTML | Base64 data URI embedding | Standard practice | Single self-contained file; no broken image links |
| Jinja2 default delimiters for LaTeX | Custom `\BLOCK{}`, `\VAR{}` delimiters | Established pattern | Eliminates all brace conflicts with LaTeX |
| PyLaTeX for Python-LaTeX | Jinja2 templates + pdflatex | Jinja2 is more flexible | Direct template control; no PyLaTeX API overhead |

**Deprecated/outdated:**
- `jinja2.Markup` class: Deprecated in Jinja2 3.1+, use `markupsafe.Markup` instead
- `text.usetex = True` in matplotlib: Not needed; matplotlib's built-in mathtext handles most formulas without external LaTeX

## Math-Heavy Source Files Inventory

The following files require full LaTeX treatment in the math verification PDF:

| File | Key Mathematics | Complexity |
|------|----------------|------------|
| `src/evaluation/svd_metrics.py` | 9 SVD metric functions: stable rank, spectral entropy, spectral gaps, condition number, rank-1 residual norm, read-write alignment, Grassmannian distance | HIGH |
| `src/model/attention.py` | Scaled dot-product attention: QK^T/sqrt(d), causal masking (zero-fill vs -inf), softmax | HIGH |
| `src/graph/dcsbm.py` | DCSBM probability matrix: P[i,j] = theta_i * theta_j * omega[b_i, b_j], Bernoulli sampling, density validation with 2-sigma tolerance | HIGH |
| `src/graph/degree_correction.py` | Zipf power-law: theta_i ~ 1/rank^alpha, per-block normalization | MEDIUM |
| `src/analysis/auroc_horizon.py` | Rank-based AUROC (Mann-Whitney U): P(X_viol > X_ctrl), predictive horizon computation | HIGH |
| `src/analysis/statistical_controls.py` | Holm-Bonferroni step-down correction, BCa bootstrap CIs, Cohen's d, Pearson correlation matrix | HIGH |
| `src/analysis/event_extraction.py` | Contamination filtering logic, event stratification | MEDIUM |
| `src/training/trainer.py` | Cosine LR schedule with warmup, cross-entropy loss | MEDIUM |
| `src/model/transformer.py` | Token + positional embeddings, residual scaling 1/sqrt(2*n_layers), WvWo = W_v^T @ W_o | HIGH |
| `src/evaluation/pipeline.py` | AVWo = (A @ V) @ W_o^T, fused evaluation, three SVD targets | HIGH |
| `src/evaluation/behavioral.py` | 4-class behavioral classification, CSR adjacency lookup | MEDIUM |
| `src/graph/jumpers.py` | R-value computation: r = round(scale * w), non-triviality check | LOW |
| `src/graph/validation.py` | Non-trivial path verification via adjacency matrix powers | MEDIUM |
| `src/walk/generator.py` | Random walk generation on directed graph | LOW |
| `src/training/data.py` | Walk chunking into non-overlapping subsequences of size w+1 | LOW |

**Non-math files (appendix only):** `__init__.py` files, `config/experiment.py`, `config/defaults.py`, `config/hashing.py`, `config/serialization.py`, `results/schema.py`, `results/experiment_id.py`, `reproducibility/seed.py`, `reproducibility/git_hash.py`, `training/checkpoint.py`, `walk/cache.py`, `walk/compliance.py`, `walk/corpus.py`, `walk/types.py`, `graph/cache.py`, `graph/types.py`, `model/block.py`, `model/types.py`, all `visualization/` files.

## Open Questions

1. **pdflatex Installation Feasibility**
   - What we know: System has ~303MB free disk space. texlive-latex-base needs ~160MB+. With recommended fonts and latex-extra (for listings, hyperref), total is ~250-300MB. This is very tight.
   - What's unclear: Whether the install will succeed within available disk space, especially after downloading packages.
   - Recommendation: Attempt minimal install with `--no-install-recommends`. If disk is too tight, the .tex file generation should still work standalone -- the user can compile on a machine with pdflatex. The code should generate .tex first, then attempt pdflatex compilation as an optional step with a clear fallback message. Consider clearing apt cache and .venv __pycache__ to free space.

2. **CLI Design for Report Generation**
   - What we know: User delegated this to Claude's discretion. Need commands for single report, comparison report, and math PDF.
   - Recommendation: Module-level functions callable from Python. A lightweight CLI wrapper can be added later if needed. For now, the `src/reporting/` module provides `generate_single_report(result_dir)`, `generate_comparison_report(result_dirs)`, and `generate_math_pdf(output_dir)`.

3. **Sparklines Implementation**
   - What we know: User wants "per-metric rows with sparklines (mini bar charts) showing relative values across experiments."
   - Recommendation: Generate tiny matplotlib figures (e.g., 120x30 pixels) as base64-encoded inline images. This stays consistent with the project's matplotlib-based visualization and avoids adding JavaScript charting libraries.

## Sources

### Primary (HIGH confidence)
- Jinja2 3.1.2 -- verified in project venv (`python -c "import jinja2; print(jinja2.__version__)"`)
- Codebase inspection -- all source files read directly from `/root/Repos/dcsbm-transformer/src/`
- `result.json` schema -- verified from `src/results/schema.py`
- `render.py` data loading -- verified from `src/visualization/render.py`
- `get_git_hash()` -- verified from `src/reproducibility/git_hash.py`
- base64 encoding -- verified working in project venv
- Jinja2 LaTeX-safe delimiters -- verified working in project venv

### Secondary (MEDIUM confidence)
- [PyLaTeX documentation](https://jeltef.github.io/PyLaTeX/current/) -- alternative library reference
- [latexbuild GitHub](https://github.com/pappasam/latexbuild) -- Jinja2 + LaTeX integration patterns
- [LaTeX templates with Python and Jinja2](https://13rac1.com/articles/2015/11/latex-templates-python-and-jinja2-generate-pdfs/) -- LaTeX-safe delimiter pattern source
- [Automated reports with Jinja2](https://nagasudhir.blogspot.com/2023/09/automated-reports-using-python-and.html) -- base64 embedding patterns

### Tertiary (LOW confidence)
- texlive disk space estimates -- approximate; actual size depends on mirror and compression

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- Jinja2, base64, matplotlib all verified installed and working
- Architecture: HIGH -- patterns verified with actual code execution in project venv
- HTML reports: HIGH -- straightforward Jinja2 + base64; all data sources exist and are well-understood
- Math PDF: MEDIUM -- Jinja2 + LaTeX pattern verified, but pdflatex not installed and disk space is tight
- Pitfalls: HIGH -- based on direct codebase analysis and verified experiments

**Research date:** 2026-02-26
**Valid until:** 2026-03-26 (stable domain; no rapidly evolving dependencies)
