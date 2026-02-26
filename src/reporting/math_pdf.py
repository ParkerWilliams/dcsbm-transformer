"""Math verification PDF generator for peer review.

Produces a LaTeX document with per-source-file sections containing
plain-language summaries, code blocks, and LaTeX math formulas. Non-math
files are listed in an appendix. Compiles to PDF via pdflatex if available.
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path

import jinja2

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Math-heavy source file definitions (15 files from RESEARCH.md inventory)
# ---------------------------------------------------------------------------

MATH_SECTIONS: list[dict] = [
    {
        "file_path": "src/evaluation/svd_metrics.py",
        "title": "SVD Metric Functions",
        "summary": (
            "This module implements nine SVD-derived metrics that quantify the "
            "spectral structure of attention matrices at each generation step. "
            "These metrics form the core measurement battery for detecting "
            "instability in the transformer's internal representations. Stable "
            "rank measures effective dimensionality, spectral entropy captures "
            "how evenly singular value mass is distributed, spectral gaps "
            "detect rank-deficiency transitions, condition number flags "
            "numerical ill-conditioning, rank-1 residual norm measures "
            "departure from rank-1 structure, read-write alignment quantifies "
            "the OV circuit's input-output coupling, and Grassmannian distance "
            "tracks how rapidly the dominant subspace rotates between steps. "
            "Together, these metrics provide complementary views of whether "
            "the attention mechanism is operating in a stable regime or "
            "approaching a hallucination-inducing instability."
        ),
        "latex_math": r"""The function \texttt{stable\_rank} computes the ratio of squared Frobenius norm to squared spectral norm:
\begin{equation}
  r_{\text{stable}}(M) = \frac{\|M\|_F^2}{\|M\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2}
\end{equation}
where $\sigma_1 \geq \sigma_2 \geq \cdots$ are the singular values of $M$. A denominator guard $\varepsilon = 10^{-12}$ prevents division by zero.

The function \texttt{spectral\_entropy} computes Shannon entropy over the normalized singular value distribution:
\begin{equation}
  H(\sigma) = -\sum_i p_i \log p_i, \quad p_i = \frac{\sigma_i}{\sum_j \sigma_j}
\end{equation}
The result is clamped to $[0, \infty)$ and a guard $\varepsilon$ is added inside the logarithm.

Spectral gaps measure consecutive singular value differences. The function \texttt{spectral\_gap\_1\_2} computes:
\begin{equation}
  \Delta_{k,k+1} = \sigma_k - \sigma_{k+1}
\end{equation}
for $(k, k+1) \in \{(1,2), (2,3), (4,5)\}$. A large gap indicates a natural rank cutoff.

The function \texttt{condition\_number} computes:
\begin{equation}
  \kappa(M) = \frac{\sigma_1}{\sigma_n + \varepsilon}, \quad \text{capped at } 10^6
\end{equation}

The function \texttt{rank1\_residual\_norm} measures how far $M$ deviates from its best rank-1 approximation:
\begin{equation}
  \rho_1(M) = \frac{\|M - \sigma_1 u_1 v_1^T\|_F}{\|M\|_F} = \frac{\sqrt{\sum_{i \geq 2} \sigma_i^2}}{\sqrt{\sum_i \sigma_i^2}}
\end{equation}

The function \texttt{read\_write\_alignment} computes the absolute cosine between the top left and right singular vectors:
\begin{equation}
  \text{align}(U, V^H) = |u_1 \cdot v_1|
\end{equation}
This is meaningful for the square $W_v W_o$ (OV circuit) matrix where read and write subspaces live in the same space.

The function \texttt{grassmannian\_distance} computes the geodesic distance on the Grassmann manifold between top-$k$ subspaces at consecutive steps. Given $U_{\text{prev}}$ and $U_{\text{curr}}$ (each containing $k=2$ columns), the principal angles $\theta_i$ satisfy:
\begin{equation}
  d_G(U_{\text{prev}}, U_{\text{curr}}) = \sqrt{\sum_{i=1}^{k} \theta_i^2}, \quad \cos\theta_i = \sigma_i(U_{\text{prev}}^T U_{\text{curr}})
\end{equation}""",
    },
    {
        "file_path": "src/model/attention.py",
        "title": "Causal Self-Attention",
        "summary": (
            "This module implements single-head causal self-attention with "
            "transparent extraction of internal matrices for SVD analysis. "
            "The attention mechanism is the core routing component that "
            "determines which tokens attend to which. By exposing the raw "
            "QK transpose matrix before and after masking, the module enables "
            "the three-target SVD analysis pipeline to measure spectral "
            "stability of the routing decisions. The dual masking convention "
            "is critical: zero-fill for SVD analysis (clean input for singular "
            "value decomposition) versus negative-infinity fill for softmax "
            "(proper probability distribution). Both masks operate on the "
            "same raw scaled dot-product matrix."
        ),
        "latex_math": r"""The scaled dot-product attention computes query-key compatibility scores. Given input $x \in \mathbb{R}^{B \times T \times D}$, the projections are $Q = xW_q$, $K = xW_k$, $V = xW_v$, each in $\mathbb{R}^{B \times T \times D}$. The raw score matrix is:
\begin{equation}
  \text{QK}^T = \frac{Q K^T}{\sqrt{d_{\text{model}}}}
\end{equation}

A lower-triangular causal mask $M_{\text{causal}}$ is applied with two different fill values for two purposes:

For softmax attention (forward pass):
\begin{equation}
  A_{ij} = \text{softmax}\!\left(\text{QK}^T \odot M_{\text{causal}} + (-\infty)(1 - M_{\text{causal}})\right)
\end{equation}

For SVD analysis (extraction target):
\begin{equation}
  \text{QK}^T_{\text{SVD}} = \text{QK}^T \odot M_{\text{causal}}
\end{equation}
where future positions are zero-filled rather than $-\infty$-filled, providing a clean matrix for singular value decomposition.

The final output is $y = A V W_o$, projected through a separate output weight matrix $W_o \in \mathbb{R}^{D \times D}$.""",
    },
    {
        "file_path": "src/graph/dcsbm.py",
        "title": "DCSBM Graph Generator",
        "summary": (
            "This module implements the Degree-Corrected Stochastic Block "
            "Model (Karrer and Newman 2011) for generating directed graphs "
            "with configurable community structure and heterogeneous degree "
            "distributions. The DCSBM serves as the synthetic data generator "
            "for the entire experiment: it produces graphs whose community "
            "structure creates predictable patterns that the transformer must "
            "learn, while the degree correction introduces realistic "
            "heterogeneity. The block structure defines which jumper rules "
            "are non-trivial, and the edge probabilities control how difficult "
            "it is for the model to navigate between communities."
        ),
        "latex_math": r"""The function \texttt{build\_probability\_matrix} constructs the edge probability matrix. For vertices $i$ and $j$ in blocks $b_i$ and $b_j$ respectively:
\begin{equation}
  P[i,j] = \theta_i \cdot \theta_j \cdot \omega[b_i, b_j]
\end{equation}
where $\theta_i$ is the degree correction parameter for vertex $i$, and $\omega$ is the $K \times K$ block affinity matrix:
\begin{equation}
  \omega[a,b] = \begin{cases} p_{\text{in}} & \text{if } a = b \\ p_{\text{out}} & \text{if } a \neq b \end{cases}
\end{equation}
The result is clipped to $[0, 1]$ and the diagonal is zeroed (no self-loops).

The function \texttt{sample\_adjacency} draws each edge independently:
\begin{equation}
  A[i,j] \sim \text{Bernoulli}(P[i,j])
\end{equation}

Validation checks edge density per block pair against a 2-sigma tolerance:
\begin{equation}
  |\hat{d}_{ab} - \bar{d}_{ab}| \leq 2\sigma_{ab}, \quad \sigma_{ab} = \sqrt{\frac{\bar{d}_{ab}(1 - \bar{d}_{ab})}{n_{ab}}}
\end{equation}
where $\hat{d}_{ab}$ is the observed density and $\bar{d}_{ab}$ is the expected density for block pair $(a, b)$.""",
    },
    {
        "file_path": "src/graph/degree_correction.py",
        "title": "Degree Correction Parameters",
        "summary": (
            "This module samples degree correction parameters from a Zipf "
            "(power-law) distribution, introducing realistic degree "
            "heterogeneity into the DCSBM graph. In natural language, word "
            "frequencies follow Zipf's law; analogously, vertex degrees in "
            "the synthetic graph follow a power-law distribution. The alpha "
            "exponent is locked at 1.0 (classic Zipf). Per-block normalization "
            "ensures that the expected total degree within each block matches "
            "the uncorrected SBM, so degree correction modulates the variance "
            "of degrees without changing their mean."
        ),
        "latex_math": r"""The function \texttt{sample\_theta} generates degree correction parameters. Within each block $b$, vertices are assigned ranks $1, 2, \ldots, n_b$ (shuffled randomly), and the raw parameter follows Zipf's law:
\begin{equation}
  \theta_i^{\text{raw}} = \frac{1}{\text{rank}(i)^{\alpha}}, \quad \alpha = 1.0
\end{equation}

Per-block normalization ensures the expected degree structure is preserved:
\begin{equation}
  \theta_i = \theta_i^{\text{raw}} \cdot \frac{n_b}{\sum_{j \in \text{block}(b)} \theta_j^{\text{raw}}}
\end{equation}
so that $\sum_{j \in \text{block}(b)} \theta_j = n_b$, matching the block size. This guarantees that the expected row sum of the probability matrix for a vertex in block $b$ is the same as in the uncorrected SBM.""",
    },
    {
        "file_path": "src/analysis/auroc_horizon.py",
        "title": "AUROC and Predictive Horizon Analysis",
        "summary": (
            "This module computes the Area Under the ROC Curve (AUROC) at "
            "each lookback distance to determine how far in advance SVD "
            "metrics can predict rule violations. The AUROC is computed via "
            "the rank-based method equivalent to the Mann-Whitney U statistic, "
            "which measures the probability that a randomly chosen violation "
            "event has a higher metric value than a randomly chosen control "
            "event. The predictive horizon is the furthest lookback distance "
            "where this discriminability exceeds a threshold, indicating that "
            "the SVD metric signal precedes the behavioral outcome. Shuffle "
            "controls validate that the signal is class-label-dependent rather "
            "than a positional artifact."
        ),
        "latex_math": r"""The function \texttt{auroc\_from\_groups} computes AUROC via the Mann-Whitney U statistic. Given violation metric values $X_1, \ldots, X_{n_v}$ and control values $Y_1, \ldots, Y_{n_c}$:
\begin{equation}
  \text{AUROC} = \frac{U}{n_v \cdot n_c}, \quad U = R_v - \frac{n_v(n_v + 1)}{2}
\end{equation}
where $R_v = \sum_{i=1}^{n_v} r_i$ is the sum of ranks of the violation group in the combined sample (ties handled via midrank).

The function \texttt{compute\_auroc\_curve} evaluates AUROC at each lookback distance $j = 1, \ldots, r$. For lookback $j$, metric values are extracted at position $\text{resolution\_step} - j$ for each event.

The function \texttt{compute\_predictive\_horizon} scans from the largest $j$ to the smallest:
\begin{equation}
  h = \max\{j : \text{AUROC}(j) > \tau\}
\end{equation}
where $\tau = 0.75$ is the default threshold. Returns 0 if no lookback exceeds the threshold.

The shuffle control permutes violation/control labels $N = 10{,}000$ times and reports the p-value:
\begin{equation}
  p = \frac{1}{N}\sum_{k=1}^{N} \mathbf{1}\!\left[\max_j \text{AUROC}_k^{\text{shuffle}}(j) \geq \max_j \text{AUROC}^{\text{obs}}(j)\right]
\end{equation}""",
    },
    {
        "file_path": "src/analysis/statistical_controls.py",
        "title": "Statistical Controls",
        "summary": (
            "This module applies rigorous statistical controls to the AUROC "
            "results to ensure publishability. Holm-Bonferroni step-down "
            "correction controls the family-wise error rate across the five "
            "pre-registered primary metrics, preventing inflated significance "
            "from multiple testing. BCa bootstrap confidence intervals provide "
            "non-parametric uncertainty estimates on AUROC values, with "
            "automatic fallback to percentile intervals if the bias-corrected "
            "method fails. Cohen's d effect sizes quantify the practical "
            "magnitude of the metric difference between violation and control "
            "groups. Pearson correlation matrices identify redundant metrics "
            "that may be measuring the same underlying signal."
        ),
        "latex_math": r"""The function \texttt{holm\_bonferroni} implements the step-down multiple comparison correction. Given $m$ p-values sorted as $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$:
\begin{equation}
  p_{(i)}^{\text{adj}} = \max_{k \leq i}\!\left\{\min\!\left((m - k + 1) \cdot p_{(k)},\; 1\right)\right\}
\end{equation}
The monotonicity enforcement (cumulative maximum) ensures that if a less significant test is rejected, all more significant tests are also rejected. Correction applies to exactly 5 primary metrics.

The function \texttt{auroc\_with\_bootstrap\_ci} uses BCa bootstrap for AUROC confidence intervals. The AUROC statistic for each resample is computed as:
\begin{equation}
  \widehat{\text{AUROC}}^{*(b)} = \frac{R_v^{*(b)} - n_v(n_v+1)/2}{n_v \cdot n_c}
\end{equation}
with 10{,}000 resamples and automatic fallback to percentile method if BCa produces NaN.

The function \texttt{cohens\_d} computes the standardized mean difference:
\begin{equation}
  d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}
\end{equation}
Returns NaN for $s_{\text{pooled}} < 10^{-12}$ or fewer than 2 samples per group.

Correlation matrices use Pearson's $r$ between metric vectors. Redundant pairs are flagged when $|r| > 0.9$.""",
    },
    {
        "file_path": "src/analysis/event_extraction.py",
        "title": "Event Extraction and Contamination Filtering",
        "summary": (
            "This module extracts jumper encounter events from generated "
            "sequences and applies contamination filtering before statistical "
            "analysis. Each event records where a jumper vertex was generated, "
            "when the rule resolves, and whether it was followed or violated. "
            "The contamination filter is essential for valid AUROC analysis: "
            "after a violation, the model's subsequent behavior may be affected "
            "by the violation itself, so encounters whose countdown window "
            "overlaps with a prior violation's window are excluded. Events are "
            "then stratified by r-value to ensure AUROC curves are computed "
            "within homogeneous groups."
        ),
        "latex_math": r"""Event extraction scans generated sequences for jumper vertices. For a jumper encounter at step $t$ with jump length $r$:
\begin{equation}
  \text{resolution\_step} = t + r
\end{equation}
The rule outcome is read at index $\text{resolution\_step} - 1$ in the \texttt{rule\_outcome} array. An event is marked as the first violation in its walk when:
\begin{equation}
  \texttt{failure\_index}[\text{walk}] = \text{resolution\_step} - 1
\end{equation}

Contamination filtering tracks $\text{last\_violation\_end}$ per walk. An encounter is excluded if:
\begin{equation}
  \text{encounter\_step} < \text{last\_violation\_end}
\end{equation}
Only violations (not FOLLOWED events) update $\text{last\_violation\_end}$ to their resolution step. The exclusion rate is flagged if it exceeds 30\%.

Stratification groups events by $r$-value. Each group receives independent AUROC analysis to prevent mixing different temporal scales.""",
    },
    {
        "file_path": "src/training/trainer.py",
        "title": "Training Loop and Learning Rate Schedule",
        "summary": (
            "This module provides the training loop for the transformer "
            "language model with AdamW optimization and cosine learning rate "
            "scheduling. The cosine schedule with linear warmup is a standard "
            "technique from modern transformer training: the learning rate "
            "increases linearly during the first 10 percent of training steps "
            "to stabilize early gradients, then decays following a cosine "
            "curve to a minimum ratio. Cross-entropy loss on next-token "
            "prediction drives the model to learn the graph's transition "
            "structure, including the community patterns that create "
            "predictable jumper rules."
        ),
        "latex_math": r"""The function \texttt{cosine\_with\_warmup} computes the learning rate multiplier. For step $t$ with warmup period $t_w$ and total steps $T$:
\begin{equation}
  \lambda(t) = \begin{cases}
    \displaystyle\frac{t}{t_w} & t < t_w \\[8pt]
    \displaystyle\eta_{\min} + \frac{1 - \eta_{\min}}{2}\left(1 + \cos\!\left(\pi \cdot \frac{t - t_w}{T - t_w}\right)\right) & t \geq t_w
  \end{cases}
\end{equation}
where $t_w = 0.1T$ (10\% warmup) and $\eta_{\min} = 0.1$ (minimum LR ratio).

The training objective is next-token prediction via cross-entropy. Given input sequence $x_{1:w}$ and targets $x_{2:w+1}$:
\begin{equation}
  \mathcal{L} = -\frac{1}{w}\sum_{i=1}^{w} \log p_\theta(x_{i+1} \mid x_{1:i})
\end{equation}
where $p_\theta$ is the model's softmax output over the vocabulary. Gradient clipping at $\|\nabla\|_{\max} = 1.0$ prevents exploding gradients.""",
    },
    {
        "file_path": "src/model/transformer.py",
        "title": "Transformer Language Model Architecture",
        "summary": (
            "This module defines the full transformer language model: learned "
            "token and positional embeddings, a stack of transformer blocks "
            "with pre-norm layer normalization, and a linear output head. "
            "The model is deliberately simple (NanoGPT-scale, single-head) to "
            "make the attention routing transparent for SVD analysis. Weight "
            "initialization follows GPT-2 conventions, with a critical "
            "residual scaling factor of 1/sqrt(2n) applied to the output "
            "projection and MLP second layer to prevent the residual stream "
            "norm from growing with depth. The WvWo product (OV circuit) is "
            "exposed as a method for static SVD analysis."
        ),
        "latex_math": r"""Token and positional embeddings are summed:
\begin{equation}
  h^{(0)} = E_{\text{tok}}[x] + E_{\text{pos}}[\text{arange}(T)]
\end{equation}
where $E_{\text{tok}} \in \mathbb{R}^{V \times D}$ and $E_{\text{pos}} \in \mathbb{R}^{T_{\max} \times D}$ are learned embeddings.

Residual scaling prevents norm growth. The output projection $W_o$ and MLP second layer are initialized with standard deviation:
\begin{equation}
  \sigma_{\text{resid}} = \frac{0.02}{\sqrt{2 n_{\text{layers}}}}
\end{equation}
All other weights use $\sigma = 0.02$ (GPT-2 convention).

The WvWo product represents the OV circuit for each layer. Given \texttt{nn.Linear} weight convention where weights are stored as $[d_{\text{out}}, d_{\text{in}}]$:
\begin{equation}
  W_v W_o = W_v^{\text{weight}\top} \cdot W_o^{\text{weight}}
\end{equation}
yielding a $D \times D$ matrix mapping input space through the value projection and output projection. This is input-agnostic (depends only on model weights) and is computed once for static SVD analysis.""",
    },
    {
        "file_path": "src/evaluation/pipeline.py",
        "title": "Fused Evaluation Pipeline",
        "summary": (
            "This module implements the fused evaluation pipeline that "
            "performs autoregressive generation while simultaneously collecting "
            "SVD metrics across three targets and behavioral labels in a single "
            "inference pass. The three SVD targets are QK transpose (routing "
            "decisions), WvWo (OV circuit, static), and AVWo (net residual "
            "update, dynamic). This design avoids the need for separate "
            "generation and analysis passes, which would be both slower and "
            "risk inconsistency. The AVWo computation matches the actual "
            "residual stream contribution computed by nn.Linear internally."
        ),
        "latex_math": r"""The function \texttt{\_compute\_avwo\_for\_layer} computes the net residual update. Given attention weights $A \in \mathbb{R}^{B \times T \times T}$, values $V \in \mathbb{R}^{B \times T \times D}$, and output projection $W_o$:
\begin{equation}
  \text{AVWo} = (A \cdot V) \cdot W_o^{\text{weight}\top}
\end{equation}
This matches \texttt{nn.Linear}'s internal computation $x W^T$, so AVWo represents the actual vector added to the residual stream at each position.

The three SVD targets computed at each generation step $t \geq w$ are:

\textbf{QK\textsuperscript{T} (routing):} The scaled dot-product matrix $QK^T / \sqrt{d}$ with zero-filled causal mask. SVD is computed on the full $T \times T$ matrix.

\textbf{WvWo (OV circuit):} The static weight product $W_v^T W_o$ computed once per layer. All metrics except Grassmannian distance are broadcast to every step (Grassmannian distance is NaN since the matrix is static).

\textbf{AVWo (residual update):} The dynamic matrix $(A \cdot V) \cdot W_o^T$ of shape $B \times T \times D$. SVD captures how the attention-weighted value output evolves across generation steps.

Grassmannian distance is computed between consecutive steps for QK\textsuperscript{T} and AVWo using the top-$k=2$ left singular vectors.""",
    },
    {
        "file_path": "src/evaluation/behavioral.py",
        "title": "Behavioral Classification",
        "summary": (
            "This module classifies each generation step into one of four "
            "behavioral categories based on edge validity against the DCSBM "
            "adjacency matrix and rule compliance against jumper constraints. "
            "Edge validity uses CSR adjacency lookup to check whether each "
            "generated transition is a real edge in the graph. Rule compliance "
            "tracks active jumper constraints as (deadline, target_block) "
            "tuples and checks whether the generated token at the deadline "
            "step is in the correct block. The failure_index annotation marks "
            "the first rule violation per sequence, enabling the AUROC "
            "analysis to focus on uncontaminated first-violation events."
        ),
        "latex_math": r"""The \texttt{classify\_steps} function processes each step $t$ in a generated sequence of length $L$. For consecutive tokens $(u, v)$ at step $t$:

\textbf{Edge validity} uses CSR adjacency lookup:
\begin{equation}
  \text{edge\_valid}[t] = v \in \{\text{indices}[\text{indptr}[u] : \text{indptr}[u+1]]\}
\end{equation}

\textbf{Rule compliance} tracks active constraints. When a jumper vertex $u$ with jump length $r$ and target block $b^*$ is encountered at step $t$:
\begin{equation}
  \text{active\_constraints} \leftarrow \text{active\_constraints} \cup \{(t + r,\; b^*)\}
\end{equation}
At each subsequent step $t'$, if $t' + 1$ equals a constraint deadline:
\begin{equation}
  \text{rule\_outcome}[t'] = \begin{cases}
    \text{FOLLOWED} & \text{if } \text{block}(v) = b^* \\
    \text{VIOLATED} & \text{if } \text{block}(v) \neq b^*
  \end{cases}
\end{equation}

The failure index records the first violation: $\texttt{failure\_index}[b] = \min\{t : \text{rule\_outcome}[b, t] = \text{VIOLATED}\}$, or $-1$ if no violations occur.""",
    },
    {
        "file_path": "src/graph/jumpers.py",
        "title": "Block Jumper Designation",
        "summary": (
            "This module designates jumper vertices within each block of the "
            "DCSBM graph. Jumpers are special vertices that impose navigation "
            "rules: when the walk visits a jumper, it must reach a specified "
            "target block within exactly r steps. The r-values are drawn from "
            "a fixed discrete set of scale factors applied to the context "
            "window size w, cycling globally across all blocks to ensure all "
            "r-values are represented. Each jumper assignment is validated for "
            "non-triviality, meaning both the target block and at least one "
            "non-target block must be reachable at distance r."
        ),
        "latex_math": r"""The function \texttt{compute\_r\_values} maps scale factors to integer jump lengths:
\begin{equation}
  r = \text{round}(\text{scale} \cdot w), \quad \text{scale} \in \{0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0\}
\end{equation}
with $r \geq 1$. The resulting set is deduplicated and sorted.

Jumper designation cycles through the sorted $r$-values using a global counter across all blocks:
\begin{equation}
  r_j = r\text{\_values}[j \bmod |r\text{\_values}|]
\end{equation}
where $j$ is the global jumper index. Each jumper $j$ in block $b$ is assigned a random target block $b^* \neq b$ and validated for non-triviality before inclusion.""",
    },
    {
        "file_path": "src/graph/validation.py",
        "title": "Path Validation for Jumper Non-Triviality",
        "summary": (
            "This module verifies that jumper assignments are non-trivial by "
            "computing which blocks are reachable from a vertex in exactly r "
            "steps. It uses iterative sparse vector-matrix multiplication on "
            "the adjacency matrix, with binary clipping at each iteration to "
            "prevent exponential overflow from path counting. A non-trivial "
            "assignment requires that both the target block and at least one "
            "alternative block are reachable, ensuring the transformer faces "
            "a genuine navigation challenge."
        ),
        "latex_math": r"""The function \texttt{reachable\_blocks\_at\_distance} computes reachability via iterated sparse matrix multiplication. Starting from indicator vector $e_v$ for vertex $v$:
\begin{equation}
  x^{(0)} = e_v, \quad x^{(i+1)} = \min\!\left(x^{(i)} \cdot A,\; 1\right), \quad i = 0, \ldots, r-1
\end{equation}
Binary clipping ($\min(\cdot, 1)$) at each step converts path counts to reachability indicators, preventing exponential growth of path-count values in dense graphs.

The function \texttt{check\_non\_trivial} verifies two conditions for a jumper at vertex $v$ with target block $b^*$ and distance $r$:
\begin{equation}
  b^* \in \mathcal{B}(v, r) \quad \text{and} \quad \mathcal{B}(v, r) \setminus \{b^*\} \neq \emptyset
\end{equation}
where $\mathcal{B}(v, r) = \{\text{block}(u) : x_u^{(r)} > 0\}$ is the set of blocks reachable from $v$ in exactly $r$ steps.""",
    },
    {
        "file_path": "src/walk/generator.py",
        "title": "Random Walk Generation",
        "summary": (
            "This module generates random walks on the directed DCSBM graph "
            "with guided segments at jumper encounters. The two-phase strategy "
            "first generates jumper-seeded walks using path-count weighted "
            "neighbor selection to ensure compliance with jumper rules, then "
            "generates random-start walks in batch using vectorized operations. "
            "Any random walk that encounters a jumper vertex is re-generated "
            "as a guided walk. The walk corpus provides the training and "
            "evaluation data for the transformer."
        ),
        "latex_math": r"""The walk generation follows a directed random walk on graph $G = (V, E)$. At each step, the walker at vertex $u$ transitions to neighbor $v$ selected from the adjacency list.

In unguided mode, the transition is uniform over neighbors:
\begin{equation}
  p(v \mid u) = \frac{1}{\deg^+(u)}, \quad v \in \mathcal{N}^+(u)
\end{equation}
where $\mathcal{N}^+(u)$ is the set of out-neighbors and $\deg^+(u) = |\mathcal{N}^+(u)|$.

In guided mode (active jumper constraint with target block $b^*$ at deadline $d$), the transition is weighted by path-count vectors from the compliance module, biasing the walk toward vertices from which the target block is still reachable.

Vectorized batch generation processes one step at a time across all walks using NumPy operations on CSR index arrays, achieving efficient parallel walk generation for the unguided case.""",
    },
    {
        "file_path": "src/training/data.py",
        "title": "Walk Data Loading",
        "summary": (
            "This module converts numpy walk arrays into PyTorch datasets for "
            "transformer training. Each walk of length L is chunked into "
            "non-overlapping subsequences of size w+1, where w is the context "
            "window. The first w tokens of each chunk serve as input and the "
            "last w tokens (shifted by one position) serve as the target for "
            "next-token prediction. This chunking scheme maximizes data "
            "utilization while ensuring the model sees clean context windows "
            "without overlap artifacts."
        ),
        "latex_math": r"""The \texttt{WalkDataset} class chunks walks into training examples. For a walk of length $L$ and context window $w$:
\begin{equation}
  n_{\text{chunks}} = \left\lfloor \frac{L}{w + 1} \right\rfloor
\end{equation}
Each chunk $c_k$ for $k = 0, \ldots, n_{\text{chunks}} - 1$ extracts a subsequence of length $w + 1$:
\begin{equation}
  c_k = \text{walk}[k(w+1) : (k+1)(w+1)]
\end{equation}
The input is $c_k[0:w]$ and the target is $c_k[1:w+1]$, forming a standard next-token prediction pair.""",
    },
]

# ---------------------------------------------------------------------------
# Non-math source files for the appendix
# ---------------------------------------------------------------------------

APPENDIX_FILES: list[dict] = [
    {"filename": "src/__init__.py", "description": "Top-level package marker"},
    {"filename": "src/analysis/__init__.py", "description": "Analysis subpackage marker"},
    {"filename": "src/config/__init__.py", "description": "Configuration subpackage marker"},
    {"filename": "src/config/experiment.py", "description": "Frozen dataclass experiment configuration with graph, model, and training sub-configs"},
    {"filename": "src/config/defaults.py", "description": "Default configuration values for the anchor experiment"},
    {"filename": "src/config/hashing.py", "description": "Deterministic config hashing for cache key generation"},
    {"filename": "src/config/serialization.py", "description": "Config serialization and deserialization via dacite"},
    {"filename": "src/evaluation/__init__.py", "description": "Evaluation subpackage marker"},
    {"filename": "src/graph/__init__.py", "description": "Graph subpackage marker"},
    {"filename": "src/graph/cache.py", "description": "Graph generation result caching with hash-based keys"},
    {"filename": "src/graph/types.py", "description": "GraphData frozen dataclass holding adjacency, block assignments, and theta"},
    {"filename": "src/model/__init__.py", "description": "Model subpackage marker"},
    {"filename": "src/model/block.py", "description": "Transformer block with pre-norm LayerNorm, attention, and MLP"},
    {"filename": "src/model/types.py", "description": "ExtractionMode enum and ForwardOutput/AttentionInternals dataclasses"},
    {"filename": "src/reporting/__init__.py", "description": "Reporting subpackage exports"},
    {"filename": "src/reporting/embed.py", "description": "Base64 figure encoding for self-contained HTML reports"},
    {"filename": "src/reporting/reproduction.py", "description": "Reproduction block builder with git hash and CLI arguments"},
    {"filename": "src/reproducibility/__init__.py", "description": "Reproducibility subpackage marker"},
    {"filename": "src/reproducibility/git_hash.py", "description": "Git commit hash capture with dirty-state detection"},
    {"filename": "src/reproducibility/seed.py", "description": "Deterministic seeding for random, numpy, torch, and CUDA"},
    {"filename": "src/results/__init__.py", "description": "Results subpackage marker"},
    {"filename": "src/results/experiment_id.py", "description": "Experiment ID generation from config hash and timestamp"},
    {"filename": "src/results/schema.py", "description": "Result JSON schema definition and validation"},
    {"filename": "src/training/__init__.py", "description": "Training subpackage marker"},
    {"filename": "src/training/checkpoint.py", "description": "Model checkpoint save/load with optimizer and scheduler state"},
    {"filename": "src/training/evaluate.py", "description": "Evaluation-time model loading and gate checking"},
    {"filename": "src/training/pipeline.py", "description": "End-to-end training pipeline orchestration"},
    {"filename": "src/visualization/__init__.py", "description": "Visualization subpackage marker"},
    {"filename": "src/visualization/auroc.py", "description": "AUROC curve and predictive horizon plots"},
    {"filename": "src/visualization/confusion.py", "description": "Behavioral classification confusion matrix visualization"},
    {"filename": "src/visualization/distributions.py", "description": "SVD metric distribution box plots across behavioral classes"},
    {"filename": "src/visualization/event_aligned.py", "description": "Event-aligned metric trajectory plots around jumper encounters"},
    {"filename": "src/visualization/heatmap.py", "description": "Predictive horizon heatmap across metrics and r-values"},
    {"filename": "src/visualization/render.py", "description": "Render orchestrator loading result data and dispatching to plot modules"},
    {"filename": "src/visualization/style.py", "description": "Colorblind-safe palette and matplotlib style configuration"},
    {"filename": "src/visualization/training.py", "description": "Training loss curve and learning rate schedule plots"},
    {"filename": "src/walk/__init__.py", "description": "Walk subpackage marker"},
    {"filename": "src/walk/cache.py", "description": "Walk corpus caching with NPZ serialization"},
    {"filename": "src/walk/compliance.py", "description": "Path-count based guided step selection for jumper compliance"},
    {"filename": "src/walk/corpus.py", "description": "Walk corpus management with train/eval splits"},
    {"filename": "src/walk/types.py", "description": "JumperEvent and WalkResult frozen dataclasses"},
]


def _read_source_file(file_path: str) -> str:
    """Read a source file and return its contents.

    Args:
        file_path: Relative path from project root.

    Returns:
        File contents as string, or placeholder if file not found.
    """
    # Try relative to project root (standard usage)
    candidates = [
        Path(file_path),
        Path(__file__).parent.parent.parent / file_path,
    ]
    for p in candidates:
        if p.is_file():
            return p.read_text(encoding="utf-8")
    return f"% Source file not found: {file_path}"


def _create_latex_env() -> jinja2.Environment:
    """Create Jinja2 environment with LaTeX-safe delimiters.

    Uses custom delimiters that do not conflict with LaTeX brace syntax:
    - Block tags: \\BLOCK{...}
    - Variable tags: \\VAR{...}
    - Comment tags: \\#{...}

    Returns:
        Configured Jinja2 Environment with FileSystemLoader.
    """
    template_dir = Path(__file__).parent / "templates"
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
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


def generate_math_pdf(output_dir: str | Path) -> Path:
    """Generate the math verification PDF (or .tex if pdflatex unavailable).

    Builds sections by reading each source file and combining with
    predefined summaries and LaTeX math. Renders the .tex template
    via Jinja2, then attempts pdflatex compilation.

    Args:
        output_dir: Directory to write output files into.

    Returns:
        Path to the generated .pdf (if pdflatex succeeded) or .tex file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build section data
    math_sections = []
    for section_def in MATH_SECTIONS:
        code = _read_source_file(section_def["file_path"])
        math_sections.append(
            {
                "title": section_def["title"],
                "summary": section_def["summary"],
                "code": code,
                "latex_math": section_def["latex_math"],
            }
        )

    # Render template
    env = _create_latex_env()
    template = env.get_template("math_verification.tex")
    tex_content = template.render(
        generation_date=datetime.now().strftime("%Y-%m-%d"),
        math_sections=math_sections,
        appendix_files=APPENDIX_FILES,
    )

    # Write .tex file
    tex_path = output_dir / "math_verification.tex"
    tex_path.write_text(tex_content, encoding="utf-8")
    log.info("Generated LaTeX file: %s", tex_path)

    # Attempt pdflatex compilation (two passes for TOC)
    pdf_path = output_dir / "math_verification.pdf"
    try:
        for pass_num in range(2):
            result = subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    str(output_dir),
                    str(tex_path),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0 and pass_num == 1:
                log.warning(
                    "pdflatex pass %d failed (return code %d). "
                    "LaTeX .tex file preserved at %s",
                    pass_num + 1,
                    result.returncode,
                    tex_path,
                )
                return tex_path

        # Clean up auxiliary files on success
        if pdf_path.exists():
            for ext in (".aux", ".log", ".toc", ".out"):
                aux_file = output_dir / f"math_verification{ext}"
                if aux_file.exists():
                    aux_file.unlink()
            log.info("Generated PDF: %s", pdf_path)
            return pdf_path

    except FileNotFoundError:
        log.warning(
            "pdflatex not found. Install texlive-latex-base to compile "
            "PDF. LaTeX .tex file preserved at %s",
            tex_path,
        )
    except subprocess.TimeoutExpired:
        log.warning(
            "pdflatex timed out. LaTeX .tex file preserved at %s",
            tex_path,
        )

    return tex_path
