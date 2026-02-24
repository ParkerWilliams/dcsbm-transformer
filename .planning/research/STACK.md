# Technology Stack

**Project:** DCSBM Transformer -- SVD Predictive Signals for LLM Hallucination
**Researched:** 2026-02-24
**Overall Confidence:** MEDIUM (versions based on training data through May 2025; unable to verify against live PyPI/docs during this research session -- flag versions for pip install verification before committing pyproject.toml)

---

## Recommended Stack

### Runtime

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Python | 3.11 | Runtime | Specified in project constraints. 3.11 is the sweet spot: full PyTorch support, good performance, stable. Avoid 3.12+ which has had intermittent C extension compatibility issues with scientific stack. | HIGH |
| venv | stdlib | Virtual environment | Project constraint requires venv. No need for conda -- pure Python + PyTorch wheels are sufficient. | HIGH |

### Core Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch | >=2.5,<2.7 | Tensor ops, model training, SVD | The entire pipeline lives in PyTorch: model definition, training loop, attention matrix extraction, `torch.linalg.svd`. Using one framework for everything avoids data transfer overhead between GPU and CPU. PyTorch 2.5+ has mature `torch.compile` support and stable `torch.linalg` API. | HIGH |
| NumPy | >=1.26,<2.1 | CPU array ops, graph adjacency | Needed for NetworkX interop and CPU-side data prep. NumPy 2.0 changed some APIs; pin to allow 2.0 but not bleeding edge to avoid ecosystem breakage. | MEDIUM |
| SciPy | >=1.12 | Sparse matrices, statistical tests | Sparse adjacency matrix storage for DCSBM (n=2000 with low p_out makes dense wasteful). Also provides Mann-Whitney U, Wilson intervals for the statistical tests in result.json. | HIGH |

### Graph Generation

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| NetworkX | >=3.2 | DCSBM graph generation | NetworkX has `stochastic_block_model()` in `networkx.generators.community`. **However, it does NOT have a built-in degree-corrected variant.** The SBM generator supports weighted block probability matrices but not the degree-correction theta parameters. We need a custom DCSBM generator built on top of NetworkX's DiGraph. NetworkX is still the right foundation because: (a) mature directed graph support, (b) random walk utilities, (c) the custom DCSBM layer is 50-80 lines of code using the standard definition (Karrer & Newman 2011). | HIGH |

**Critical note on DCSBM generation:** Do NOT use `networkx.generators.community.stochastic_block_model()` directly -- it generates the planted partition / standard SBM, not the degree-corrected variant. The degree correction (theta parameters that give heterogeneous degree distributions within blocks) must be implemented manually. The implementation is straightforward:

```python
# Pseudocode for DCSBM edge generation
# For each pair (i, j) in different blocks b_i, b_j:
#   P(edge i->j) = theta_i * theta_j * omega[b_i][b_j]
# where theta_i ~ Pareto or drawn from empirical degree distribution
# and omega is the block interaction matrix (p_in on diagonal, p_out off-diagonal)
```

**Why NOT graph-tool:** graph-tool has a proper DCSBM implementation (`graph_tool.generation.generate_sbm` with degree correction), but it requires separate C++ compilation, is not pip-installable on all platforms, and adds deployment friction on RunPod. For n<=2000 vertices, a custom NetworkX-based generator is fast enough (sub-second) and far simpler to deploy.

**Why NOT igraph:** python-igraph has SBM generation but also lacks degree correction. Same custom-code situation as NetworkX but with a less Pythonic API.

### Transformer Architecture

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyTorch (nn.Module) | (same as above) | Model definition | Write the transformer from scratch using `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`. At NanoGPT scale (d_model=64-256, n_layers=2-6, 1 head), there is zero benefit to using a framework like HuggingFace Transformers -- it would add massive dependency weight and make attention matrix extraction harder. The model is ~200 lines of code. | HIGH |

**Why NOT HuggingFace Transformers:** Massive dependency (tokenizers, safetensors, accelerate, etc.) for a model that fits in <1MB. More importantly, extracting the raw QK^T attention matrix before softmax requires hooking into the attention computation, which is straightforward in custom code but requires fighting HF's abstraction layers.

**Why NOT minGPT/nanoGPT directly:** Karpathy's nanoGPT is a good reference but is not a library -- it's a training script. The architecture pattern (GPT block = LayerNorm -> MultiheadAttention -> LayerNorm -> MLP) should be followed, but the code should be purpose-built to expose QK^T at every step without hooks.

**Architecture pattern to follow:**

```python
class SingleHeadAttention(nn.Module):
    """Returns attn_output AND raw QK^T matrix for SVD analysis."""
    def forward(self, x):
        Q = self.W_Q(x)  # (B, T, d)
        K = self.W_K(x)  # (B, T, d)
        V = self.W_V(x)  # (B, T, d)
        qkt = Q @ K.transpose(-2, -1) / math.sqrt(self.d_model)  # (B, T, T)
        # Return qkt BEFORE softmax for SVD analysis
        attn = F.softmax(qkt.masked_fill(causal_mask, float('-inf')), dim=-1)
        return attn @ V, qkt
```

### SVD Computation

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| torch.linalg.svd | (PyTorch) | SVD of QK^T | `torch.linalg.svd(A, full_matrices=False)` runs on GPU, supports batched input (leading batch dimensions), and returns U, S, Vh. With `full_matrices=False` on a (T, T) matrix where T=context window (32-256), this is fast enough per-step. Critical: keep everything on GPU -- do NOT move to CPU for numpy SVD. | HIGH |

**Performance strategy for SVD collection:**

1. **Batch across sequences:** During evaluation, process multiple sequences in a batch. `torch.linalg.svd` accepts `(B, T, T)` input and returns batched `(B, T, T)`, `(B, T)`, `(B, T, T)` -- one SVD per batch element in parallel on GPU.

2. **`full_matrices=False`:** For a (T, T) matrix this returns U as (T, T), S as (T,), Vh as (T, T) -- same as full since the matrix is square. The savings come if QK^T were rectangular, which it is not for self-attention. But it is still good practice and the flag prevents accidental overhead if matrix shapes change.

3. **Compute metrics from S, U, Vh directly on GPU:** All ~20 SVD metrics (condition number, spectral gap, entropy, stable rank, etc.) are simple arithmetic on the singular values and vectors. Compute them as vectorized PyTorch operations, not Python loops.

4. **Amortize over steps:** At each token step t during autoregressive generation, the QK^T matrix is (t, t) growing. For early steps (t < 10), SVD is trivially fast. For late steps (t = 256), SVD of a 256x256 matrix takes ~0.1ms on an RTX 3090. The total overhead for a 256-step sequence is ~10ms of SVD time, negligible compared to the forward pass.

5. **Do NOT use `torch.svd`:** This is the old API. Use `torch.linalg.svd` exclusively.

**Why NOT numpy.linalg.svd:** Requires GPU->CPU transfer, which is the bottleneck. Keep everything on GPU.

**Why NOT scipy.linalg.svd:** Same CPU issue. Also no batching support.

**Why NOT randomized SVD (sklearn.utils.extmath.randomized_svd):** We need ALL singular values for metrics like entropy, stable rank, and condition number. Randomized SVD only gives top-k. It could be used for the low-rank approximation error metric specifically, but the full SVD is fast enough at these matrix sizes (up to 256x256) that the added complexity is not worth it.

### Experiment Orchestration

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Custom job queue (Python) | N/A | Parameter sweep orchestration | The project spec calls for a prioritized job queue where scientifically critical configs run first and budget can be cut at any point. This is 100-200 lines of custom code using `dataclasses` + JSON persistence + a simple priority ordering function. See rationale below. | HIGH |
| dataclasses + JSON | stdlib | Config definition and serialization | Each experiment config is a frozen dataclass serialized to JSON. No framework needed. | HIGH |

**Why NOT Hydra:** Hydra (by Meta) is excellent for config management in large ML projects, but it adds complexity disproportionate to this project's needs. The config space is well-defined (a fixed set of numeric parameters), not the kind of nested config composition Hydra excels at. Hydra's multirun feature does not support priority ordering or budget-aware cutoff. The overhead of learning and debugging Hydra's config resolution is not justified for a single-researcher project.

**Why NOT Weights & Biases (wandb):** wandb is a cloud-hosted experiment tracker. Concerns: (1) adds network dependency on RunPod instances that may have spotty internet, (2) the project already defines its own result.json schema with full reproducibility metadata, (3) $100 budget means we want zero friction -- not signing up for accounts or debugging API keys on rental GPUs. wandb's sweep feature also does not support priority ordering.

**Why NOT Optuna:** Optuna is for hyperparameter optimization (finding the best config). This project is a parameter sweep (running all specified configs to measure a scientific quantity at each point). There is no optimization objective -- we need results at every point in the sweep grid.

**Why NOT Ray Tune:** Overkill. Ray Tune is for distributed hyperparameter search across clusters. This runs on a single GPU.

**Recommended approach:**

```python
@dataclass(frozen=True)
class ExperimentConfig:
    n: int
    w: int
    p_in: float
    p_out: float
    t: int
    r_ratio: float  # r as multiple of w
    # ... etc

    @property
    def priority(self) -> int:
        """Lower = higher priority. Anchor config = 0."""
        if self.is_anchor: return 0
        if self.is_r_sweep: return 1  # Core scientific question
        if self.is_w_sweep: return 2
        # ... etc

class JobQueue:
    def __init__(self, configs: list[ExperimentConfig]):
        self.pending = sorted(configs, key=lambda c: c.priority)
        self.completed = []
        self.persist("queue.json")

    def next(self) -> ExperimentConfig | None:
        return self.pending[0] if self.pending else None
```

### Results Storage and Plotting

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| JSON (stdlib) | N/A | Result storage | Project spec defines result.json schema. JSON is human-readable, git-diffable, and universally parseable. No database needed at this scale. | HIGH |
| matplotlib | >=3.8 | Core plotting engine | Foundation for all plot types in the spec. Mature, well-documented, produces publication-quality figures. | HIGH |
| seaborn | >=0.13 | Statistical visualization | Built on matplotlib. Used for heatmaps (confusion matrix), styled plots, and the `whitegrid` theme specified in the project's plotting guide. | HIGH |

**Why NOT Plotly:** Interactive plots are unnecessary -- all outputs are static PNG/SVG for reports and the PDF. Plotly adds a heavy JS dependency and the HTML reports embed base64 PNGs.

**Why NOT Altair/Vega:** Same reasoning. Static scientific figures are the requirement.

### HTML Report Generation

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Jinja2 | >=3.1 | HTML templating | The report generation code in the spec uses raw string concatenation. Jinja2 templates are cleaner, support loops/conditionals for dynamic sections, and produce more maintainable report code. The project spec's HTML structure maps directly to a Jinja2 template. | HIGH |
| base64 (stdlib) | N/A | Image embedding | Embed PNG figures as base64 in self-contained HTML. Already specified in combined-spec. | HIGH |

**Why NOT a full static site generator (MkDocs, Sphinx):** The reports are single self-contained HTML files. No navigation, no build step, no theme system needed.

### Math Verification PDF

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pylatex | >=1.4 | LaTeX document generation from Python | Specified in the project spec. Generates .tex files programmatically with proper escaping and structure. | MEDIUM |
| pdflatex | system | PDF compilation | Standard LaTeX compiler. Must be installed on the system (`apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended`). | HIGH |

**Alternative if pylatex causes issues:** Raw Jinja2 templates generating .tex files. pylatex's API can be finicky with complex math environments. Having a LaTeX template with Jinja2 placeholders is often more predictable. Recommend trying pylatex first, falling back to raw templates if needed.

### Development and Testing

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytest | >=8.0 | Unit testing | Standard Python testing. Critical for math verification -- every SVD metric function needs unit tests with known analytical solutions. | HIGH |
| pytest-timeout | >=2.2 | Test timeouts | SVD computation tests can hang on degenerate matrices. Timeout prevents CI stalls. | MEDIUM |
| ruff | >=0.4 | Linting and formatting | Single tool replaces flake8 + black + isort. Fast, opinionated, minimal config. | HIGH |
| mypy | >=1.10 | Type checking | The config/dataclass layer benefits from strict typing. Catch shape mismatches early. | MEDIUM |
| tqdm | >=4.66 | Progress bars | Training loops and evaluation sweeps need progress indication. Lightweight. | HIGH |

### System Dependencies (RunPod)

| Dependency | Purpose | Install |
|------------|---------|---------|
| CUDA 12.1+ | GPU compute | Pre-installed on RunPod PyTorch templates |
| texlive-latex-base | PDF compilation | `apt install texlive-latex-base texlive-latex-extra texlive-fonts-recommended` |
| git | Version tracking, code hash | Pre-installed |

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Graph library | NetworkX + custom DCSBM | graph-tool | Not pip-installable, C++ compilation on RunPod is fragile |
| Graph library | NetworkX + custom DCSBM | igraph | No DCSBM either, less Pythonic API, marginal benefit at n<=2000 |
| Model framework | Raw PyTorch nn.Module | HuggingFace Transformers | Massive deps, hides attention internals, overkill for <1MB models |
| Model framework | Raw PyTorch nn.Module | minGPT/nanoGPT repo | Not a library; reference pattern only |
| SVD engine | torch.linalg.svd (GPU) | numpy.linalg.svd | Requires GPU->CPU transfer, no batching |
| SVD engine | torch.linalg.svd (GPU) | scipy.linalg.svd | CPU only, no batching |
| SVD engine | torch.linalg.svd (GPU) | sklearn randomized_svd | Only returns top-k; we need full spectrum |
| Config management | Custom dataclass + JSON | Hydra | Overkill for fixed numeric sweep, no priority queue |
| Experiment tracking | Custom result.json | Weights & Biases | Network dependency, budget friction, project already defines schema |
| Hyperparameter search | Manual priority sweep | Optuna | Not optimization -- we need results at every grid point |
| Distributed training | Single-GPU PyTorch | Ray Tune | Single GPU on RunPod, no distribution needed |
| Plotting | matplotlib + seaborn | Plotly | Static figures required, no interactivity needed |
| HTML reports | Jinja2 templates | MkDocs/Sphinx | Single self-contained HTML files, no site generator needed |
| PDF generation | pylatex | pandoc | pylatex gives programmatic control over LaTeX math environments |

---

## Full Dependency List

### Core (pyproject.toml)

```toml
[project]
name = "dcsbm-transformer"
version = "0.1.0"
requires-python = ">=3.11,<3.13"

dependencies = [
    "torch>=2.5,<2.7",
    "numpy>=1.26,<2.1",
    "scipy>=1.12",
    "networkx>=3.2",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "jinja2>=3.1",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-timeout>=2.2",
    "ruff>=0.4",
    "mypy>=1.10",
]
pdf = [
    "pylatex>=1.4",
]
```

### Installation

```bash
# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA (RunPod)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install project
pip install -e ".[dev,pdf]"

# System deps for PDF generation
apt install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

---

## Architecture-Aware Stack Decisions

### Why everything stays in PyTorch

The pipeline is: Graph -> Walks -> Tensor -> Train -> Generate -> Extract QK^T -> SVD -> Metrics.

After walk generation (which uses NetworkX on CPU), everything from tokenization onward lives on GPU in PyTorch tensors. The critical path is:

1. Token embeddings are `nn.Embedding` (GPU)
2. Forward pass produces QK^T as a PyTorch tensor (GPU)
3. `torch.linalg.svd` operates on that tensor (GPU)
4. Metric computation is PyTorch arithmetic (GPU)
5. Results are `.cpu().numpy()` only at the final storage step

Moving to NumPy or SciPy mid-pipeline would require `tensor.cpu().numpy()` transfers that dominate runtime for the SVD collection loop. This is the single most important stack decision.

### Why custom code over frameworks

At NanoGPT scale with a single attention head and a well-defined synthetic data pipeline, the "framework" is the project itself. Every abstraction layer between us and QK^T is a liability:

- Custom model: ~200 lines, full control over QK^T extraction
- Custom training loop: ~100 lines, standard cross-entropy with eval gates
- Custom data pipeline: ~150 lines, walks from NetworkX graph to PyTorch tensors
- Custom job queue: ~200 lines, priority-ordered configs with budget tracking
- Custom reporting: ~300 lines, Jinja2 templates + matplotlib

Total: ~1000 lines of purpose-built code vs. thousands of lines of framework dependencies that would need customization anyway.

### Seed control strategy

```python
import torch
import numpy as np
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full reproducibility on CUDA (slight perf cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

This must be called before every experiment run. The seed is stored in config and result.json for reproducibility.

---

## Version Verification Notes

**IMPORTANT:** The version ranges above are based on training data through May 2025. Before finalizing pyproject.toml, run the following to verify current stable versions:

```bash
pip index versions torch
pip index versions networkx
pip index versions matplotlib
pip index versions seaborn
pip index versions jinja2
pip index versions pylatex
```

If PyTorch has released 2.7+ with breaking changes to `torch.linalg.svd`, the upper bound should be adjusted. The `torch.linalg` API has been stable since PyTorch 2.0 and is unlikely to break, but verify.

---

## Sources

- PyTorch `torch.linalg.svd` documentation (docs.pytorch.org) -- HIGH confidence on API stability
- NetworkX community generators documentation (networkx.org) -- HIGH confidence that SBM exists, DCSBM does not
- Karrer & Newman (2011) "Stochastic blockmodels with a growing number of classes" -- defines DCSBM
- Karpathy nanoGPT (github.com/karpathy/nanoGPT) -- architectural reference for small transformer
- Project combined-spec.md -- defines result.json schema, plotting guide, reporting guide
- Project PROJECT.md -- defines constraints and anchor configuration

**Confidence notes:**
- Specific library version numbers: MEDIUM (training data, not verified against live PyPI)
- API recommendations (torch.linalg.svd, NetworkX SBM): HIGH (stable APIs, unlikely to have changed)
- Architecture patterns (custom model, custom job queue): HIGH (based on project requirements analysis)
- "Why not" recommendations: HIGH (based on project constraints: single GPU, $100 budget, single researcher)
