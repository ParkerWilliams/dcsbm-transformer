# Research Scaffolding Tool — Project Prompt

> **Usage:** Replace the block marked `[PASTE ABSTRACT HERE]` with your research direction.
> Then invoke Claude Code with this file as your starting prompt.
> GSD will be invoked once the project structure is clear.

---

## Research Abstract



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