---
phase: quick-2
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/evaluation/pipeline.py
  - src/analysis/auroc_horizon.py
  - src/analysis/statistical_controls.py
autonomous: true
requirements: [QUICK-2]

must_haves:
  truths:
    - "User can see batch-level progress during fused_evaluate (e.g., 'Batch 3/32')"
    - "User can see step-level progress within each batch at regular intervals"
    - "User can see per-stage timing for WvWo pre-computation, batch processing, and behavioral classification"
    - "User can see progress during AUROC analysis (per r-value, per metric)"
    - "User can see progress during statistical controls (bootstrap, Holm-Bonferroni)"
  artifacts:
    - path: "src/evaluation/pipeline.py"
      provides: "Comprehensive progress logging in fused_evaluate and save_evaluation_results"
    - path: "src/analysis/auroc_horizon.py"
      provides: "Progress logging in run_auroc_analysis and run_shuffle_control"
    - path: "src/analysis/statistical_controls.py"
      provides: "Progress logging in apply_statistical_controls"
  key_links:
    - from: "src/evaluation/pipeline.py"
      to: "logging module"
      via: "log.info calls at batch boundaries and timing checkpoints"
      pattern: "log\\.info.*[Bb]atch"
---

<objective>
Add comprehensive progress logging to the evaluation pipeline and post-evaluation analysis stages so the user can track where a long-running experiment is during execution.

Purpose: The user's experiment ran for 17 hours with no visibility into progress. The evaluation pipeline (`fused_evaluate`) processes sequences in batches with per-step SVD computation, but emits zero log messages. The AUROC analysis and statistical controls also run silently. Adding structured progress logging lets the user estimate remaining time and identify bottlenecks.

Output: Updated pipeline.py, auroc_horizon.py, and statistical_controls.py with progress logging using the existing `logging` module pattern already established in the codebase.
</objective>

<execution_context>
@/root/.claude/get-shit-done/workflows/execute-plan.md
@/root/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/evaluation/pipeline.py
@src/analysis/auroc_horizon.py
@src/analysis/statistical_controls.py
@run_experiment.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add progress logging to fused_evaluate in pipeline.py</name>
  <files>src/evaluation/pipeline.py</files>
  <action>
Add progress logging to `fused_evaluate()` using the existing `log` logger (already defined on line 30 as `log = logging.getLogger(__name__)`). Import `time` at the top of the file.

Specific log points to add:

1. **Entry summary** (after line 155, before batch loop): Log the evaluation configuration.
   ```python
   log.info(
       "Evaluation: %d sequences, batch_size=%d, max_steps=%d, w=%d, n_layers=%d, n_heads=%d",
       n_sequences, batch_size, max_steps, w, n_layers, n_heads,
   )
   ```

2. **WvWo pre-computation timing** (wrap lines 173-188 in a timer): Log how long static WvWo SVD takes.
   ```python
   t0 = time.monotonic()
   # ... existing WvWo pre-computation code ...
   log.info("WvWo pre-computation: %.1fs (%d layers x %d heads)", time.monotonic() - t0, n_layers, n_heads)
   ```

3. **Batch progress** (at the start of the batch loop, line 239): Log batch number, progress percentage, and elapsed/ETA.
   Add a `eval_start = time.monotonic()` before the batch loop, then at the top of each batch iteration:
   ```python
   n_batches = (n_sequences + batch_size - 1) // batch_size
   batch_idx = batch_start // batch_size + 1
   elapsed = time.monotonic() - eval_start
   if batch_idx > 1:
       eta = elapsed / (batch_idx - 1) * (n_batches - batch_idx + 1)
       log.info("Batch %d/%d (%.0f%%) — elapsed %.0fs, ETA %.0fs", batch_idx, n_batches, 100 * batch_idx / n_batches, elapsed, eta)
   else:
       log.info("Batch %d/%d — starting", batch_idx, n_batches)
   ```

4. **Step progress within batch** (inside the step loop, line 263): Log every 25% of max_possible_length steps using modular arithmetic. Do NOT log every step — that would flood the output.
   ```python
   step_log_interval = max(1, max_possible_length // 4)
   # Inside the step loop, after SVD collection block:
   if step > 0 and step % step_log_interval == 0:
       log.debug("  Batch %d/%d step %d/%d", batch_idx, n_batches, step, max_possible_length - 1)
   ```
   Use `log.debug` for step-level so it only shows with --verbose.

5. **Behavioral classification** (after line 458, around `classify_steps`): Log batch behavioral classification.
   ```python
   log.debug("  Batch %d/%d: classifying behavioral labels", batch_idx, n_batches)
   ```

6. **Completion summary** (before the return statement): Log total evaluation time and summary stats.
   ```python
   total_time = time.monotonic() - eval_start
   n_violations = int((all_failure_index >= 0).sum())
   log.info(
       "Evaluation complete: %d sequences in %.1fs (%.2f seq/s), %d violations",
       n_sequences, total_time, n_sequences / max(total_time, 0.001), n_violations,
   )
   ```

Also add a log.info line inside `save_evaluation_results` after writing the NPZ file:
   ```python
   log.info("Saved token_metrics.npz (%d metric arrays, %d sequences)", len(npz_data), result.sequence_lengths.shape[0])
   ```

IMPORTANT: Do NOT change any function signatures, return values, or computational logic. Only add logging statements and timing instrumentation.
  </action>
  <verify>
    <automated>cd /root/Repos/dcsbm-transformer && python -m pytest tests/test_evaluation_pipeline.py -x -q 2>&1 | tail -5</automated>
  </verify>
  <done>fused_evaluate logs batch progress with ETA, WvWo timing, step-level debug logs, and completion summary. save_evaluation_results logs NPZ write. All existing tests pass unchanged.</done>
</task>

<task type="auto">
  <name>Task 2: Add progress logging to AUROC analysis and statistical controls</name>
  <files>src/analysis/auroc_horizon.py, src/analysis/statistical_controls.py</files>
  <action>
Add progress logging to the two analysis modules that run after evaluation. These currently have zero logging and can take significant time (AUROC runs 10,000 shuffle permutations per metric).

**In `src/analysis/auroc_horizon.py`:**

1. Add `import logging` and `log = logging.getLogger(__name__)` at the top (after existing imports).

2. In `run_auroc_analysis()` (line 286+):
   - After event extraction (line 325-328): Log event counts.
     ```python
     log.info("AUROC analysis: %d events extracted, %d after contamination filter", len(all_events), len(filtered_events))
     log.info("AUROC analysis: %d r-value groups, %d metrics to analyze", len(by_r), len(metric_keys))
     ```
   - At the start of each r-value loop iteration (line 346): Log which r-value is being processed.
     ```python
     log.info("AUROC r=%d: %d violations, %d controls (tier=%s)", r_val, n_violations, n_controls, tier)
     ```
   - After the metric loop within each r-value (after line 431): Log metric count for that r-value.
     ```python
     log.info("AUROC r=%d: analyzed %d metrics", r_val, len(r_result["by_metric"]))
     ```

3. In `run_shuffle_control()` (line 135+):
   - Log shuffle progress every 2500 permutations (inside the permutation loop, line 178):
     ```python
     if (perm + 1) % 2500 == 0:
         log.debug("  Shuffle control: %d/%d permutations", perm + 1, n_permutations)
     ```

**In `src/analysis/statistical_controls.py`:**

1. Add `import logging` and `log = logging.getLogger(__name__)` at the top (after existing imports, before function definitions).

2. Find the main `apply_statistical_controls()` function and add progress logging:
   - At entry: Log what is being computed.
   - At each major sub-step (Holm-Bonferroni, bootstrap CIs, correlation analysis, metric ranking): Log which step is running.
   - At completion: Log summary.

   The exact log points depend on the function structure, but the pattern should be:
   ```python
   log.info("Statistical controls: starting Holm-Bonferroni correction (%d p-values)", ...)
   log.info("Statistical controls: bootstrap CIs for %d metrics", ...)
   log.info("Statistical controls: correlation/redundancy analysis")
   log.info("Statistical controls: metric importance ranking")
   log.info("Statistical controls: complete")
   ```

IMPORTANT: Do NOT change any function signatures, return values, or computational logic. Only add logging statements.
  </action>
  <verify>
    <automated>cd /root/Repos/dcsbm-transformer && python -m pytest tests/test_auroc_horizon.py tests/test_statistical_controls.py -x -q 2>&1 | tail -5</automated>
  </verify>
  <done>AUROC analysis logs event counts, per-r-value progress, and per-metric processing. Statistical controls logs each major computation step. Shuffle controls log progress every 2500 permutations at debug level. All existing tests pass unchanged.</done>
</task>

</tasks>

<verification>
1. All existing evaluation tests pass: `python -m pytest tests/test_evaluation_pipeline.py -x -q`
2. All AUROC tests pass: `python -m pytest tests/test_auroc_horizon.py -x -q`
3. All statistical control tests pass: `python -m pytest tests/test_statistical_controls.py -x -q`
4. Logging output is visible when running with INFO level: `python -c "import logging; logging.basicConfig(level=logging.INFO); from src.evaluation.pipeline import fused_evaluate; print('import ok')"`
5. No function signatures or return values changed (only additive log statements).
</verification>

<success_criteria>
- fused_evaluate emits batch progress logs with ETA estimates visible at INFO level
- Step-level progress visible at DEBUG level (--verbose)
- AUROC analysis logs per-r-value and per-metric progress at INFO level
- Statistical controls log each major computation phase at INFO level
- All 536+ existing tests continue to pass
- No changes to any function signatures, return values, or computational logic
</success_criteria>

<output>
After completion, create `.planning/quick/2-add-logging-to-evaluation-step/2-SUMMARY.md`
</output>
