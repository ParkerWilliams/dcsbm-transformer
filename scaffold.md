---

## Research Abstract
We want to build a framework to predict LLM errors at the token level using a controlled synthetic environment with known ground truth. The core scientific question is whether instability in the SVD of the QK^T attention matrix precedes and predicts rule violations in a transformer model, and how far ahead of the violation this signal is detectable.
We use a degree-corrected stochastic block model (DCSBM) to generate a directed graph where vertices represent tokens and directed edges represent one token following another. Walks on this graph are the training corpus and represent valid sequences. The graph has block structure where each block can be thought of as a concept or semantic cluster. We impose hidden rules on a subset of vertices called block jumpers. A block jumper vertex v_i in block b has an associated jump length r, meaning that after exactly r steps from v_i, the walk must land in a specific target block different from b. This rule is never encoded for the transformer - it must be learned implicitly from the corpus. The jump length r is the primary experimental variable and is swept relative to the context window size w. Valid paths from v_i to the target block must exist at length r but must not be the only paths, otherwise the rule is trivially learnable from graph topology alone. The in-group and out-group connectivity density parameters of the DCSBM are therefore essential experimental controls for ensuring the experiment is non-trivial.
The governing parameters and their intended sweep ranges are as follows. n is the number of vertices, swept over 200, 500, 1000, 2000. w is the transformer context window size, swept over 32, 64, 128, 256. The DCSBM block structure parameters include number of blocks swept over 4, 8, 16, in-group edge probability p_in swept over 0.15, 0.25, 0.40, and out-group edge probability p_out swept over 0.01, 0.03, 0.07. t is training corpus size, always kept at minimum two orders of magnitude larger than n with values swept over 50k, 200k, 1M, 5M walks. r is the jump length, swept as multiples of w at 0.5w, 0.7w, 0.9w, 1.0w, 1.1w, 1.3w, 1.5w, 2.0w - we expect hallucination rate to increase monotonically with r and to show a step change as r crosses w. l is the walk length included in the training corpus, always set to at least 2w and swept at 2w, 4w, 8w. Number of block jumper vertices per block is swept over 1, 2, 5 as a fraction of block size. Each configuration is run with 3 random seeds. The full sweep is large and should be structured so that the core r vs w interaction experiment runs first with n=500, w=64, t=200k as the baseline configuration, with all other parameters varied around this anchor.
The transformer architecture is NanoGPT scale with d_model swept over 64, 128, 256, n_layers swept over 2, 4, 6, and exactly 1 attention head throughout. The single attention head constraint is intentional and essential - it keeps the QK^T matrix analysis unambiguous and interpretable. Context window w matches the parameter above.
Training sufficiency is a hard gate before any SVD analysis is run. A configuration is considered sufficiently trained only when edge compliance rate exceeds 95 percent and rule compliance rate exceeds 80 percent on held-out walks. If a configuration does not meet this gate after the allocated training budget, it is flagged and excluded from SVD analysis rather than producing noise results.
Evaluation uses three independent layers. First is training signal, which is standard cross-entropy on next token prediction used only to monitor convergence and determine when the sufficiency gate is met, not reported as a result metric. Second is behavioural compliance, where at each step of a generated walk the model output is checked for edge validity meaning the chosen next token corresponds to a valid directed edge in the DCSBM, and rule compliance meaning that at step r from a block jumper vertex the walk lands in the required target block. This produces four outcome classes per step combining edge valid or invalid with rule followed, rule violated, or rule not applicable. The hallucination label in the results schema is this four-class outcome. Third is predictive horizon, which is the core result metric. For each confirmed rule violation event, we look back j steps and ask whether SVD instability metrics were elevated at step t minus j. Sweeping j from 1 to r gives a predictive horizon curve per metric per configuration. The headline result for each metric is the AUROC at each value of j, and the furthest j at which AUROC exceeds 0.75 is the predictive horizon for that metric.
The SVD metrics to be collected at every token step from the QK^T matrix are as follows. Direction of the principal left and right singular vectors tracked as the angle between consecutive steps. Dominant subspace membership tracked as which indices belong to the top-k singular vectors and the set difference between consecutive steps. Principal angles between consecutive dominant subspaces as the canonical Grassmannian distance. Condition number as sigma_1 divided by sigma_n. Spectral gap as sigma_1 minus sigma_2 and generalised gap as sigma_k minus sigma_k+1 for k equal to 2, 4, 8. Singular value entropy computed as negative sum of p_i log p_i where p_i is sigma_i divided by the sum of all singular values, measuring effective rank. Stable rank computed as the squared Frobenius norm divided by the squared spectral norm. Participation ratio computed as the square of the L1 norm of singular values divided by the product of the squared L2 norm and n. Low-rank approximation error at rank k for k equal to 2, 4, 8 computed as the Frobenius norm of QK^T minus its rank-k approximation. Angular velocity of the principal vector as the rate of change of the principal angle between steps. Subspace drift as the Grassmannian distance between dominant subspaces at consecutive steps. Singular value velocity as the per-step change in each of the top-k singular values. Condition number velocity. Alignment of the dominant left singular vector with the token embedding of the current token. Alignment of the dominant right singular vector with the token embedding of the predicted next token. Coherence of the dominant subspace with the full embedding matrix measured as the maximum cosine similarity between any subspace basis vector and any token embedding. Rank of QK^T restricted to the current context window tokens. Variance of singular values across the context window.
All of these metrics are stored as token-level time series in the sequences block of the results JSON per the project schema. Each metric gets its own entry in token_metrics keyed by metric name. The failure_index field marks the confirmed rule violation event and is the alignment anchor for all event-aligned plots.
Results are stored per experimental configuration where each configuration is one combination of all governing parameters plus a random seed. Each configuration gets its own result.json. The comparison report across configurations is the primary deliverable and must support filtering by any subset of parameters and overlaying aligned metric curves across filtered configurations.
The compute budget is 100 USD on RunPod. Use RTX 3090 or RTX 4090 instances. The first run should be a single anchor configuration end-to-end to calibrate wall time before launching the full sweep. Training and SVD collection should be profiled separately. SVD collection at every token step for long sequences is O(d cubed) per step and must be optimised, using torch.linalg.svd with full_matrices=False and batching where possible. The parameter sweep should be structured as a job queue so that the most scientifically critical configurations run first and the budget can be cut at any point without losing the core result.

---

## Project Context

- **Initiative:** AI Health Research — Hospital Mathematics Division
- **Framework:** This project uses [GSD (Get Shit Done)](https://github.com/gsd-build/get-shit-done) for project scaffolding. Read the GSD repo before generating any structure and follow its conventions.
- **Verification Requirement:** All core mathematical logic must be extractable into a peer-review PDF (see below).

---

## Clarification Phase

Before scaffolding anything, conduct a structured clarifying interview. Ask one topic at a time. Do not proceed to GSD invocation until all of the following are resolved with enough specificity:

- [ ] Research question and hypothesis
- [ ] Data sources — type, structure, availability (synthetic vs. clinical vs. sensor)
- [ ] Mathematical / statistical methods expected or preferred
- [ ] Evaluation criteria and success metrics
- [ ] Constraints — privacy, compute budget, reproducibility requirements
- [ ] Required output artifacts (reports, model weights, visualizations, etc.)

Once all are answered, summarize your understanding and ask for confirmation before invoking GSD.

---

## GSD Invocation

After confirmation, invoke GSD to scaffold the project. The generated project must include:

- A `README.md` with the research question, methods summary, and reproduction steps
- A `requirements.txt` (Python 3.11+)
- A `Makefile` with at minimum: `make run`, `make test`, `make pdf`
- Source organized so that math-heavy modules are clearly separated from I/O, config, and orchestration code

---

## Math Verification PDF

After the project is scaffolded and core logic is implemented, produce a PDF for researcher sign-off.

**Process:**

1. Identify all source files containing core mathematical logic. Signals to look for:
   - Numerical methods, matrix operations, statistical models
   - Loss / objective functions, optimization routines
   - Probability distributions, signal processing, transforms
   - Files with math-heavy docstrings or equation comments

2. For each identified file, generate:
   - A 1–2 sentence plain-language summary of what the file does
   - The full code block (monospaced)
   - A LaTeX representation of the mathematics the code implements

   > Use an LLM call (Anthropic API) to generate LaTeX from code — do not use static heuristics. LaTeX accuracy is more important than generation speed.

3. Compile into a single PDF with the following structure:
   - Title page: project name, research question, date
   - Table of contents
   - One section per math-heavy file: summary → code → LaTeX → plain-language description
   - Appendix: list of all other source files (non-math) for completeness

4. The PDF is intended for peer review. Note clearly on the title page that LaTeX was AI-generated and requires researcher sign-off before being treated as ground truth.

**Tooling:** Use `pylatex` or raw LaTeX templating + `pdflatex` for compilation.

---

## Notes for Claude Code

- Read the GSD repo first. Follow its structure and conventions throughout.
- Do not write any code before the clarification phase is complete and confirmed.
- Keep the math extraction and PDF generation as a standalone `make pdf` step — it should be re-runnable at any point after implementation.
- If anything in this prompt is ambiguous, ask before proceeding.
- Always run python commands in a venv

Guiding Principals
## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.