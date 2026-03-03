---
phase: quick-4
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - src/evaluation/pipeline.py
  - src/analysis/auroc_horizon.py
autonomous: true
requirements: [PERF-01, PERF-02, PERF-03, PERF-04]

must_haves:
  truths:
    - "Evaluation pipeline produces numerically identical outputs (same NPZ values within float tolerance)"
    - "SVD metric storage uses vectorized slice assignments instead of per-element Python loops"
    - "CPU transfers happen once after generation loop completes, not every step"
    - "AUROC curve computation uses numpy vectorized indexing instead of Python loops"
    - "Shuffle controls reuse pre-extracted values and use vectorized permutation"
  artifacts:
    - path: "src/evaluation/pipeline.py"
      provides: "Optimized fused evaluation with deferred CPU transfers and vectorized storage"
    - path: "src/analysis/auroc_horizon.py"
      provides: "Vectorized AUROC curve computation and optimized shuffle controls"
  key_links:
    - from: "src/evaluation/pipeline.py"
      to: "src/evaluation/svd_metrics.py"
      via: "compute_all_metrics returns batched tensors"
      pattern: "compute_all_metrics"
    - from: "src/analysis/auroc_horizon.py"
      to: "src/analysis/event_extraction.py"
      via: "AnalysisEvent walk_idx and resolution_step fields"
      pattern: "ev\\.walk_idx.*ev\\.resolution_step"
---

<objective>
Optimize the evaluation pipeline for ~10x overall speedup by eliminating Python-level loops and unnecessary GPU-CPU synchronization in the two hottest code paths: SVD metric collection during autoregressive generation, and AUROC analysis with shuffle controls.

Purpose: The user is about to push to a fresh runpod for evaluation. These optimizations reduce wall-clock time substantially while preserving numerical correctness (identical outputs).

Output: Modified `src/evaluation/pipeline.py` and `src/analysis/auroc_horizon.py` with all 4 optimizations applied.
</objective>

<execution_context>
@/root/.claude/get-shit-done/workflows/execute-plan.md
@/root/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/evaluation/pipeline.py
@src/evaluation/svd_metrics.py
@src/analysis/auroc_horizon.py
@src/analysis/event_extraction.py
@tests/test_evaluation_pipeline.py
@tests/test_auroc_horizon.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Defer CPU transfers and vectorize metric storage in pipeline.py</name>
  <files>src/evaluation/pipeline.py</files>
  <action>
Two optimizations to `fused_evaluate()` in `src/evaluation/pipeline.py`:

**Optimization A: Defer CPU transfers (lines ~335-470)**

Currently `.cpu().numpy()` is called inside the step loop for every metric value, every layer, every head, forcing GPU synchronization each time. Instead:

1. Change the `svd_metric_arrays` dict to hold **GPU tensors** during generation, not numpy arrays. Pre-allocate as `torch.full((n_sequences, max_steps - 1), float('nan'), device=device, dtype=torch.float32)` for each key.

2. Similarly, change `spectrum_data` arrays to GPU tensors during generation: `torch.full((n_sequences, max_steps - 1, spectrum_k), float('nan'), device=device, dtype=torch.float16)`.

3. Inside the step loop, store metric values directly as GPU tensors without `.cpu().numpy()`. The metric values from `compute_all_metrics` are already GPU tensors.

4. After the generation loop (after all batches complete), do a single bulk transfer: iterate over all keys and call `.cpu().numpy()` once per array.

5. For the `u_prev` dict, `.clone()` stays on GPU (already correct).

**Optimization B: Vectorize metric storage (lines ~344-470)**

Replace all `for b_idx in range(B_actual)` loops with vectorized slice assignments:

Current pattern (appears ~15 times):
```python
vals_np = val.cpu().numpy()
for b_idx in range(B_actual):
    svd_metric_arrays[key][batch_start + b_idx, step] = vals_np[b_idx] if vals_np.ndim > 0 else vals_np.item()
```

Replace with (now that arrays are GPU tensors):
```python
val_1d = metric_val.mean(dim=-1) if metric_val.dim() > 1 else metric_val
svd_metric_arrays[key][batch_start:batch_end, step] = val_1d
```

This eliminates both the Python loop AND the per-metric `.cpu()` call.

Apply the same vectorization for:
- QKT metric storage (~line 344-362)
- QKT grassmannian distance storage (~line 373-384)
- AVWo metric storage (~line 413-430)
- AVWo grassmannian distance storage (~line 440-452)
- WvWo static metric broadcast (~line 458-471) — for WvWo, the values are scalars from pre-computation; convert them to a tensor or use `fill_` on the batch slice
- Spectrum data storage (~line 335-339) — replace `for b_idx` with `spectrum_data[spec_key][batch_start:batch_end, step, :] = s_top`
- Post-generation behavioral copy (~lines 483-500) — replace `for b_idx` loops with slice assignments: `all_generated[batch_start:batch_end, :seq_len] = gen_np[:, :seq_len]` etc. Note: target_lengths may differ per sequence, so the generated/edge_valid/rule_outcome copy needs care. For `all_generated`, use the max target_length in the batch for the copy since values beyond each sequence's length are zeros anyway. For `all_seq_lengths`, use vectorized: `all_seq_lengths[batch_start:batch_end] = np.minimum(target_lengths, gen_np.shape[1])`. For edge_valid/rule_outcome/failure_index copy, use vectorized slice since `classify_steps` already returns batch-shaped arrays.

**CRITICAL correctness constraints:**
- The WvWo inner loop at line 456 is INCORRECTLY nested — it re-iterates `for layer_idx in range(n_layers): for head_idx in range(n_heads)` inside the OUTER `for layer_idx / for head_idx` loop. This is a bug (quadratic work). Fix by de-indenting the WvWo block so it runs once per step at the same level as the QKT/AVWo blocks, not nested inside them.
- Preserve NaN initialization for warmup positions (steps < w-1).
- Keep the guard_activations dict as Python ints (not tensors).
- Keep the tail extension `for b_idx` loop (line 296-308) as-is — it has conditional logic per sequence that cannot be vectorized.
- Do NOT change `compute_all_metrics`, `guard_matrix_for_svd`, or `grassmannian_distance` — they are already batched over B.
  </action>
  <verify>
Run the evaluation pipeline tests to confirm correctness is preserved:

```
cd /root/Repos/dcsbm-transformer && python -m pytest tests/test_evaluation_pipeline.py -x -v 2>&1 | tail -30
```

All tests must pass with identical behavior.
  </verify>
  <done>
- No `.cpu().numpy()` calls inside the step loop (only after all batches complete)
- No `for b_idx in range(B_actual)` loops for metric storage (replaced with slice assignments)
- WvWo nested loop bug fixed (runs once per step, not n_layers * n_heads times per step)
- All existing tests pass
  </done>
</task>

<task type="auto">
  <name>Task 2: Vectorize AUROC curve computation and optimize shuffle controls in auroc_horizon.py</name>
  <files>src/analysis/auroc_horizon.py</files>
  <action>
Two optimizations to `src/analysis/auroc_horizon.py`:

**Optimization A: Vectorize `compute_auroc_curve` (lines 60-112)**

Current code uses Python loops to extract metric values for each lookback j:
```python
for j in range(1, r_value + 1):
    viol_vals = []
    for ev in violation_events:
        idx = ev.resolution_step - j
        ...
        viol_vals.append(val)
```

Replace with numpy advanced indexing:

1. Pre-extract walk indices and resolution steps into arrays once:
```python
viol_walks = np.array([ev.walk_idx for ev in violation_events])
viol_res = np.array([ev.resolution_step for ev in violation_events])
ctrl_walks = np.array([ev.walk_idx for ev in control_events])
ctrl_res = np.array([ev.resolution_step for ev in control_events])
```

2. For each lookback j, compute indices and extract values vectorized:
```python
for j in range(1, r_value + 1):
    # Violation values
    viol_idx = viol_res - j
    viol_valid = (viol_idx >= 0) & (viol_idx < n_steps)
    viol_vals_all = metric_array[viol_walks[viol_valid], viol_idx[viol_valid]]
    viol_finite = viol_vals_all[np.isfinite(viol_vals_all)]

    # Same for controls
    ctrl_idx = ctrl_res - j
    ctrl_valid = (ctrl_idx >= 0) & (ctrl_idx < n_steps)
    ctrl_vals_all = metric_array[ctrl_walks[ctrl_valid], ctrl_idx[ctrl_valid]]
    ctrl_finite = ctrl_vals_all[np.isfinite(ctrl_vals_all)]

    if len(viol_finite) >= min_per_class and len(ctrl_finite) >= min_per_class:
        aurocs[j - 1] = auroc_from_groups(viol_finite, ctrl_finite)
```

**Optimization B: Vectorize `run_shuffle_control` (lines 138-211)**

Current code does 10,000 permutations, each creating Python lists of AnalysisEvent objects and calling `compute_auroc_curve` which re-extracts values from scratch. Instead:

1. Pre-extract ALL event walk indices and resolution steps into arrays once (combining violations + controls):
```python
all_walks = np.array([ev.walk_idx for ev in all_events])
all_res = np.array([ev.resolution_step for ev in all_events])
```

2. Pre-extract metric values for ALL events at ALL lookback distances into a matrix. Create a 2D array `values_by_j` of shape `(r_value, n_total)` where `values_by_j[j_idx, event_idx]` is the metric value at lookback j for that event. Use NaN where out-of-bounds or non-finite:
```python
n_steps = metric_array.shape[1]
values_by_j = np.full((r_value, n_total), np.nan)
for j in range(1, r_value + 1):
    idx = all_res - j
    valid = (idx >= 0) & (idx < n_steps)
    raw = np.full(n_total, np.nan)
    raw[valid] = metric_array[all_walks[valid], idx[valid]]
    values_by_j[j - 1] = raw
```

3. For each permutation, instead of creating AnalysisEvent lists and calling compute_auroc_curve, just shuffle the column indices and compute AUROC directly from the pre-extracted values:
```python
for perm in range(n_permutations):
    perm_indices = rng.permutation(n_total)
    perm_viol_mask = np.zeros(n_total, dtype=bool)
    perm_viol_mask[perm_indices[:n_viol]] = True

    max_auroc_perm = np.nan
    for j_idx in range(r_value):
        row = values_by_j[j_idx]
        finite_mask = np.isfinite(row)
        viol_vals = row[finite_mask & perm_viol_mask]
        ctrl_vals = row[finite_mask & ~perm_viol_mask]
        if len(viol_vals) >= 2 and len(ctrl_vals) >= 2:
            auc = auroc_from_groups(viol_vals, ctrl_vals)
            if np.isfinite(auc):
                max_auroc_perm = max(max_auroc_perm, auc) if np.isfinite(max_auroc_perm) else auc
    shuffle_max_aurocs[perm] = max_auroc_perm
```

This eliminates: (a) creating 20,000 AnalysisEvent Python objects per permutation, (b) the nested Python loops inside compute_auroc_curve re-extracting values from the metric_array 10,000 times.

4. Also update the observed max AUROC computation to use the same vectorized `compute_auroc_curve` (which was already updated in Optimization A).

**Also vectorize `n_valid_by_lookback` in `run_auroc_analysis` (lines 405-415):**

Replace the nested Python loop:
```python
for j in range(1, r_val + 1):
    n_valid = 0
    for ev in violations + controls:
        ...
```

With vectorized:
```python
all_ev = violations + controls
ev_walks = np.array([ev.walk_idx for ev in all_ev])
ev_res = np.array([ev.resolution_step for ev in all_ev])
n_valid_by_lookback = []
for j in range(1, r_val + 1):
    idx = ev_res - j
    valid = (idx >= 0) & (idx < n_steps)
    vals = metric_array[ev_walks[valid], idx[valid]]
    n_valid_by_lookback.append(int(np.sum(np.isfinite(vals))))
```

**CRITICAL correctness constraints:**
- `auroc_from_groups` must receive the same values in the same order as before (violations first, controls second in the combined rank array). The function itself is already correct.
- The `min_per_class` check in `compute_auroc_curve` must use the count of finite values (not total events).
- Shuffle control p95 logging interval remains at 2500.
- Do NOT change `auroc_from_groups`, `compute_predictive_horizon`, `_is_primary_metric`, `parse_metric_key`, or `_classify_event_count`.
  </action>
  <verify>
Run the AUROC tests to confirm correctness is preserved:

```
cd /root/Repos/dcsbm-transformer && python -m pytest tests/test_auroc_horizon.py -x -v 2>&1 | tail -30
```

All tests must pass with identical behavior.
  </verify>
  <done>
- `compute_auroc_curve` uses numpy advanced indexing instead of Python list-append loops
- `run_shuffle_control` pre-extracts values once and permutes indices, not AnalysisEvent objects
- `n_valid_by_lookback` computation vectorized
- All existing AUROC tests pass
  </done>
</task>

<task type="auto">
  <name>Task 3: Run full test suite to verify no regressions</name>
  <files></files>
  <action>
Run the complete test suite to verify all optimizations preserve correctness across the entire codebase:

```bash
cd /root/Repos/dcsbm-transformer && python -m pytest tests/ -x --timeout=120 -q
```

If any test fails, diagnose and fix the issue in the modified files only. Do not modify test files — the tests define the correctness contract.

Additionally, verify no remaining `for b_idx` loops exist in pipeline.py's step loop (the only acceptable `for b_idx` loop is the tail extension check around line 296):
```bash
grep -n "for b_idx" src/evaluation/pipeline.py
```
Should show at most 1 occurrence (the tail extension loop).

Verify no `.cpu().numpy()` calls inside the step loop:
```bash
# The step loop starts at "for step in range(max_possible_length - 1):"
# and ends before "# After generation: behavioral classification"
# No .cpu().numpy() should appear between those markers
```
  </action>
  <verify>
```
cd /root/Repos/dcsbm-transformer && python -m pytest tests/ -x --timeout=120 -q 2>&1 | tail -20
```

All tests pass. Zero failures.
  </verify>
  <done>
- Full test suite passes with zero failures
- No `for b_idx` loops remain in the metric storage section of pipeline.py
- No `.cpu().numpy()` calls inside the step loop of pipeline.py
- All 4 optimizations confirmed working: deferred CPU transfer, vectorized metric storage, vectorized AUROC curves, optimized shuffle controls
  </done>
</task>

</tasks>

<verification>
1. `python -m pytest tests/test_evaluation_pipeline.py -x -v` -- all pass
2. `python -m pytest tests/test_auroc_horizon.py -x -v` -- all pass
3. `python -m pytest tests/ -x --timeout=120 -q` -- full suite passes
4. `grep -n "\.cpu()\.numpy()" src/evaluation/pipeline.py` -- no occurrences inside step loop
5. `grep -n "for b_idx" src/evaluation/pipeline.py` -- at most 1 occurrence (tail extension)
</verification>

<success_criteria>
- All 536+ existing tests pass unchanged
- pipeline.py: zero .cpu().numpy() inside step loop, zero per-element for loops for metric storage
- auroc_horizon.py: compute_auroc_curve uses numpy indexing, shuffle control pre-extracts values
- WvWo quadratic nesting bug fixed
- Numerical correctness preserved (tests are the oracle)
</success_criteria>

<output>
After completion, create `.planning/quick/4-optimize-evaluation-pipeline-performance/4-SUMMARY.md`
</output>
