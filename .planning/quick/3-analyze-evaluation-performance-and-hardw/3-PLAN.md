---
phase: quick-3
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/profile_evaluation.py
autonomous: true
requirements: [QUICK-3]

must_haves:
  truths:
    - "Running the script produces a clear performance report with per-stage timings"
    - "Report identifies which pipeline stages consume the most wall-clock time"
    - "Report shows GPU utilization and memory usage during inference"
    - "Report flags concrete optimization opportunities with estimated impact"
  artifacts:
    - path: "scripts/profile_evaluation.py"
      provides: "Standalone profiling/benchmarking script for evaluation pipeline"
      min_lines: 300
  key_links:
    - from: "scripts/profile_evaluation.py"
      to: "src/evaluation/pipeline.py"
      via: "imports fused_evaluate and instruments its inner loops"
      pattern: "from src.evaluation.pipeline import"
---

<objective>
Create a profiling/benchmarking script that measures and reports on every stage of the
evaluation pipeline to diagnose a 17+ hour evaluation run.

Purpose: The user ran an evaluation that took 17+ hours and has zero visibility into
what is slow, whether the GPU is utilized, or what can be optimized. This script
provides that visibility.

Output: `scripts/profile_evaluation.py` -- a runnable script producing a structured
performance report to stdout.
</objective>

<execution_context>
@/root/.claude/get-shit-done/workflows/execute-plan.md
@/root/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@src/evaluation/pipeline.py
@src/evaluation/svd_metrics.py
@src/evaluation/behavioral.py
@src/analysis/auroc_horizon.py
@src/analysis/statistical_controls.py
@src/analysis/event_extraction.py
@src/training/evaluate.py
@src/model/transformer.py
@src/config/experiment.py
@run_experiment.py
@config.json

<interfaces>
<!-- Key types and contracts the executor needs -->

From src/evaluation/pipeline.py:
```python
def fused_evaluate(
    model: nn.Module,
    eval_walks: np.ndarray,
    graph_data: GraphData,
    jumpers: list[JumperInfo],
    config: ExperimentConfig,
    device: torch.device,
    batch_size: int = 32,
) -> EvaluationResult:
```

From src/evaluation/svd_metrics.py:
```python
def guard_matrix_for_svd(M: torch.Tensor) -> tuple[torch.Tensor, bool]
def compute_all_metrics(S, U=None, Vh=None) -> dict[str, torch.Tensor]
def grassmannian_distance(U_prev, U_curr, k=2) -> torch.Tensor
```

From src/analysis/auroc_horizon.py:
```python
def run_auroc_analysis(eval_result_data, jumper_map, metric_keys, ...) -> dict
def run_shuffle_control(violation_events, control_events, metric_array, r_value, n_permutations=10_000, ...) -> dict
```

From src/analysis/statistical_controls.py:
```python
def apply_statistical_controls(auroc_results, eval_data, jumper_map, n_bootstrap=10_000, ...) -> dict
```

From src/config/experiment.py:
```python
# Default config: n=500, K=4, d_model=128, n_layers=4, n_heads=1, w=64, walk_length=256
```

From run_experiment.py:
```python
# Pipeline stages (post-training): Evaluation -> Split -> Save NPZ -> AUROC Analysis -> Statistical Controls -> Visualization -> Reporting
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create evaluation pipeline profiling script</name>
  <files>scripts/profile_evaluation.py</files>
  <action>
Create `scripts/profile_evaluation.py` that profiles the full post-training evaluation
pipeline. The script must be self-contained, set up its own graph/model/walks using
the project's existing generation infrastructure, then measure each stage.

**Structure the script in these sections:**

**1. Setup (create small but representative test fixtures):**
- Use the project's existing config defaults (`ExperimentConfig()`) but with a REDUCED
  corpus to keep total profiling time under 5 minutes. Specifically:
  - Use `n_sequences=50` for evaluation (not the full corpus)
  - Use `batch_size=32` (the default from fused_evaluate)
- Generate graph via `src.graph.generate_or_load_graph(config)`
- Generate walks via `src.walk.generate_or_load_walks(graph_data, jumpers, config)`
- Create and randomly initialize model via `src.model.create_model(config)`, move to device
- Put model in eval mode

**2. Profile the `fused_evaluate` function (the main bottleneck):**
- Time the ENTIRE fused_evaluate call
- ALSO instrument individual components by running isolated micro-benchmarks:
  a. **Model forward pass only** (ExtractionMode.SVD_TARGETS): Time a single batch forward
     pass 10 times, report mean/std. Use `torch.cuda.synchronize()` before timing if on GPU.
  b. **SVD computation only**: Extract a representative QK^T matrix from the forward output,
     run `torch.linalg.svd()` on it 100 times, report mean/std. Compare GPU vs CPU SVD time.
  c. **WvWo pre-computation**: Time `model.get_wvwo()` + SVD on the static weight matrices.
  d. **Behavioral classification** (`classify_steps`): Time it on a batch of generated sequences.
  e. **Per-element metric storage loop**: Time the Python for-loop that copies SVD metrics
     into numpy arrays (the `for b_idx in range(B_actual)` loops in pipeline.py lines 349-362).
     Estimate this overhead by timing a synthetic version: create fake metric tensors and
     copy them element-by-element vs. vectorized slice assignment.
- Compute the **estimated time breakdown** at full scale: multiply micro-benchmark times
  by the number of batches and steps in a full run (n_sequences=10000, max_steps=4*w=256).

**3. Profile AUROC analysis:**
- Run `run_auroc_analysis` with `n_shuffle=100` (reduced from 10,000) and time it.
- Extrapolate to n_shuffle=10,000.
- Separately time `run_shuffle_control` for a single metric to show per-metric cost.

**4. Profile statistical controls:**
- Run `apply_statistical_controls` with `n_bootstrap=100` (reduced from 10,000) and time it.
- Extrapolate to n_bootstrap=10,000.

**5. Hardware utilization checks:**
- Check `torch.cuda.is_available()` and report device
- If GPU available:
  - Report GPU name, memory total/allocated/reserved via `torch.cuda.get_device_properties`
    and `torch.cuda.memory_allocated/reserved`
  - During the model forward pass, measure peak GPU memory via `torch.cuda.max_memory_allocated`
  - Check if SVD is running on GPU by verifying tensor device of SVD inputs/outputs
  - Measure GPU utilization via `torch.cuda.utilization()` if available, or suggest
    `nvidia-smi` command
- Report CPU core count via `os.cpu_count()`
- Note whether any operations are unexpectedly on CPU when GPU is available

**6. Optimization opportunity analysis:**
- Based on measured timings, flag the top 3-5 bottlenecks with specific recommendations:
  - **Per-element Python loops**: The `for b_idx in range(B_actual)` loops storing metrics
    (pipeline.py ~lines 349-362, 419-430, 457-470) can be vectorized to
    `svd_metric_arrays[key][batch_start:batch_end, step] = vals_np`
  - **Sequential SVD calls**: 3 SVD decompositions per layer per head per step could
    potentially be batched
  - **CPU-GPU transfers**: `.cpu().numpy()` calls inside the inner step loop force
    synchronization; could be deferred to end of batch
  - **Shuffle control scale**: 10,000 permutations * N metrics * N r-values; check if
    parallelizable or if n_permutations can be reduced
  - **classify_steps pure Python**: Nested Python loops over sequences and steps; could
    be vectorized with numpy

For each bottleneck, estimate the percentage of total time it consumes and the expected
speedup from the fix.

**7. Output format:**
Print a structured report to stdout with clear sections:

```
=== EVALUATION PIPELINE PERFORMANCE PROFILE ===

--- Hardware ---
Device: cuda (NVIDIA A100 / cpu)
GPU Memory: X.X GB total, X.X GB allocated
CPU Cores: N

--- Stage Timings (measured at reduced scale) ---
Stage                          Time (s)    % of Total
fused_evaluate (50 seq)        XX.X        XX%
  - model forward (per batch)  XX.X
  - SVD per step               XX.X
  - WvWo pre-compute           XX.X
  - behavioral classify        XX.X
  - metric storage loops       XX.X
AUROC analysis (100 shuffles)  XX.X        XX%
Statistical controls (100 bs)  XX.X        XX%
Total measured                 XX.X

--- Estimated Full-Scale Timings (10,000 sequences) ---
Stage                          Est. Time    % of Total
fused_evaluate                 XX.X h       XX%
AUROC analysis (10k shuffles)  XX.X h       XX%
Statistical controls (10k bs)  XX.X h       XX%
Total estimated                XX.X h

--- Top Bottlenecks ---
1. [description] -- est. XX% of total, potential Xs speedup
2. ...

--- GPU Utilization ---
SVD running on: GPU/CPU
Peak GPU memory during forward: X.X GB
Tensor devices: [list any CPU tensors that should be on GPU]

--- Recommendations ---
1. [specific actionable recommendation]
2. ...
```

**Implementation notes:**
- Use `argparse` with `--config` pointing to config.json (default: config.json)
- Add `--n-sequences` flag (default 50) to control profiling scale
- Add `--skip-analysis` flag to skip AUROC/stats profiling (just profile fused_evaluate)
- Use `time.monotonic()` for all wall-clock timing
- Use `torch.cuda.synchronize()` before every timing boundary when on GPU
- Wrap each profiling section in try/except so partial results are still printed if
  something fails (e.g., no GPU available)
- Set seeds for reproducibility
- Make the script executable (`#!/usr/bin/env python3`)
  </action>
  <verify>
    <automated>cd /root/Repos/dcsbm-transformer && python -c "import ast; ast.parse(open('scripts/profile_evaluation.py').read()); print('Syntax OK')" && python scripts/profile_evaluation.py --config config.json --n-sequences 10 --skip-analysis 2>&1 | head -80</automated>
  </verify>
  <done>
    - Script parses without syntax errors
    - Running with --n-sequences 10 --skip-analysis produces a performance report showing
      per-stage timings, hardware info, and optimization recommendations
    - Report includes estimated full-scale timings extrapolated from micro-benchmarks
    - Report identifies concrete bottlenecks with estimated impact
  </done>
</task>

</tasks>

<verification>
1. `python scripts/profile_evaluation.py --config config.json --n-sequences 10 --skip-analysis`
   runs to completion and prints a structured report
2. Report includes Hardware, Stage Timings, Estimated Full-Scale, Top Bottlenecks, and
   Recommendations sections
3. Script handles both GPU and CPU-only environments gracefully
</verification>

<success_criteria>
- Script produces a clear, actionable performance report identifying what consumes time
  in the 17+ hour evaluation run
- Hardware utilization (GPU vs CPU, memory) is reported
- Top 3-5 optimization opportunities are identified with estimated impact
- Script runs in under 5 minutes at default settings (50 sequences)
</success_criteria>

<output>
After completion, create `.planning/quick/3-analyze-evaluation-performance-and-hardw/3-SUMMARY.md`
</output>
