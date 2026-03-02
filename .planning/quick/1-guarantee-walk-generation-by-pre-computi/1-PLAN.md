---
phase: quick-1
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - src/walk/compliance.py
  - src/walk/generator.py
  - tests/test_walk_generator.py
autonomous: true
requirements: ["QUICK-1"]

must_haves:
  truths:
    - "Every walk containing a jumper event is compliant by construction (no post-hoc discard)"
    - "Pre-computed viable paths exist for every jumper vertex before walk generation begins"
    - "Walk generation never returns None due to jumper constraint infeasibility"
    - "All existing tests pass with the new approach"
    - "Reproducibility is preserved (same seed produces same walks)"
  artifacts:
    - path: "src/walk/compliance.py"
      provides: "precompute_viable_paths() function and updated precompute_path_counts()"
      exports: ["precompute_viable_paths", "precompute_path_counts"]
    - path: "src/walk/generator.py"
      provides: "Refactored walk generation using path splicing"
      exports: ["generate_single_guided_walk", "generate_walks"]
    - path: "tests/test_walk_generator.py"
      provides: "Updated tests verifying guaranteed compliance"
  key_links:
    - from: "src/walk/compliance.py"
      to: "src/walk/generator.py"
      via: "precompute_viable_paths() called from generate_walks()"
      pattern: "precompute_viable_paths"
    - from: "src/walk/generator.py"
      to: "src/walk/compliance.py"
      via: "viable_paths dict passed to generate_single_guided_walk()"
      pattern: "viable_paths"
---

<objective>
Replace the probabilistic guided-step walk generation with a guaranteed approach that
pre-computes viable jumper paths and splices them into walks during generation.

Purpose: The current approach uses path-count weighted neighbor selection during guided
walk segments. This is probabilistic -- walks can still fail compliance (guided_step
returns None when all weights are zero, or post-hoc block check fails), causing discards
and retries. By pre-computing actual viable r-step walks from each jumper vertex to its
target block, we guarantee compliance by construction and eliminate all discard/retry logic.

Output: Modified compliance.py with path precomputation, refactored generator.py with
path splicing, updated tests proving guaranteed compliance.
</objective>

<execution_context>
@/root/.claude/get-shit-done/workflows/execute-plan.md
@/root/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@src/walk/compliance.py
@src/walk/generator.py
@src/walk/types.py
@src/graph/jumpers.py
@src/graph/types.py
@src/graph/validation.py
@tests/test_walk_generator.py

<interfaces>
<!-- Key types and contracts the executor needs -->

From src/graph/jumpers.py:
```python
@dataclass(frozen=True, slots=True)
class JumperInfo:
    vertex_id: int
    source_block: int
    target_block: int
    r: int  # jump length in steps
```

From src/graph/types.py:
```python
@dataclass(frozen=True)
class GraphData:
    adjacency: scipy.sparse.csr_matrix  # directed adjacency (n x n)
    block_assignments: np.ndarray       # int array length n, vertex -> block
    n: int
    K: int
    block_size: int
    # ...
```

From src/walk/types.py:
```python
@dataclass(frozen=True, slots=True)
class JumperEvent:
    vertex_id: int
    step: int
    target_block: int
    expected_arrival_step: int

@dataclass(frozen=True)
class WalkResult:
    walks: np.ndarray           # int32 (num_walks, walk_length)
    events: list[list[JumperEvent]]
    walk_seeds: np.ndarray      # int64 per-walk seeds
```

From src/walk/compliance.py:
```python
def precompute_path_counts(adj, block_assignments, K, max_r) -> dict[int, list[np.ndarray]]:
    """path_counts[target_block][k][vertex] = normalized path count"""

def guided_step(vertex, active_constraints, step, path_counts, indptr, indices, rng) -> int | None:
    """Weight neighbors by path-count product, return None if infeasible"""
```

Current generate_walks() flow:
1. Builds jumper_map, calls precompute_path_counts()
2. Phase 1: Jumper-seeded guided walks (while loop with discard/retry)
3. Phase 2: Random-start batch walks, re-runs as guided if jumper encountered
4. _validate_walks() does post-hoc compliance checking
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Pre-compute viable jumper paths in compliance.py</name>
  <files>src/walk/compliance.py</files>
  <action>
Add a new function `precompute_viable_paths()` to `compliance.py` that, for each jumper,
collects a pool of viable r-step walks from the jumper vertex that end in the target block.

Function signature:
```python
def precompute_viable_paths(
    adj: csr_matrix,
    block_assignments: np.ndarray,
    jumpers: list[JumperInfo],
    rng: np.random.Generator,
    n_paths_per_jumper: int = 200,
    max_attempts_factor: int = 20,
) -> dict[int, list[np.ndarray]]:
```

Returns `dict[vertex_id, list[np.ndarray]]` where each np.ndarray has shape `(r+1,)` dtype
int32 — the full path including the jumper vertex as first element and the target-block
vertex as last element.

Algorithm:
1. Extract CSR indptr/indices from adj.
2. For each jumper in `jumpers`:
   a. Set `v = jumper.vertex_id`, `r = jumper.r`, `tb = jumper.target_block`.
   b. Run up to `n_paths_per_jumper * max_attempts_factor` random walk attempts from `v`:
      - Walk `r` steps using uniform random neighbor selection from CSR arrays.
      - If `block_assignments[walk[-1]] == tb`, add the walk to the viable pool.
      - Stop early once `n_paths_per_jumper` viable paths are collected.
   c. If zero viable paths found, raise `ValueError` with a message identifying the
      jumper vertex, target block, and r value — this means the jumper was designated
      incorrectly (should not happen given non-triviality checks, but fail loudly).
   d. If fewer than `n_paths_per_jumper` found but at least 1, log a warning and continue.
3. Log summary: number of paths found per jumper, min/max/median across jumpers.

Keep the existing `precompute_path_counts()` and `guided_step()` functions. They are still
used by overlapping-constraint handling (see Task 2). Add the import for JumperInfo at the
top of the file:
```python
from src.graph.jumpers import JumperInfo
```

Also add `precompute_viable_paths` to the module-level exports if any exist.
  </action>
  <verify>
Run: `cd /root/Repos/dcsbm-transformer && python -c "from src.walk.compliance import precompute_viable_paths; print('import ok')"`
  </verify>
  <done>
`precompute_viable_paths()` exists in compliance.py, imports cleanly, accepts the specified
parameters, and returns dict[int, list[np.ndarray]] mapping jumper vertex IDs to pools of
viable r-step paths.
  </done>
</task>

<task type="auto">
  <name>Task 2: Refactor walk generator to use path splicing and update tests</name>
  <files>src/walk/generator.py, tests/test_walk_generator.py</files>
  <action>
**Part A: Refactor generator.py**

Modify `generate_single_guided_walk()` to accept a `viable_paths` parameter and use path
splicing instead of step-by-step guided walking:

1. Update the signature to add `viable_paths: dict[int, list[np.ndarray]]` parameter.
   Remove `path_counts` parameter since it is no longer needed for single-constraint
   guided stepping.

2. Replace the guided walk logic. The new algorithm:
   ```
   walk = np.zeros(walk_length, dtype=np.int32)
   walk[0] = start_vertex
   events = []
   step = 1

   while step < walk_length:
       prev = walk[step - 1]

       if prev in jumper_map and prev in viable_paths:
           jumper = jumper_map[prev]
           # Pick a random viable path for this jumper
           paths = viable_paths[prev]
           path = paths[rng.integers(0, len(paths))]
           # path[0] == prev (the jumper vertex itself)
           # path has length r+1, path[-1] is in target block

           # Record the event
           event = JumperEvent(
               vertex_id=int(prev),
               step=int(step - 1),
               target_block=int(jumper.target_block),
               expected_arrival_step=int(step - 1 + jumper.r),
           )
           events.append(event)

           # Splice the path into the walk (skip path[0] since prev is already placed)
           splice_len = min(jumper.r, walk_length - step)
           walk[step:step + splice_len] = path[1:1 + splice_len]
           step += splice_len
       else:
           # Unguided step: uniform random neighbor
           start_idx = indptr[prev]
           end_idx = indptr[prev + 1]
           degree = end_idx - start_idx
           if degree == 0:
               return None  # dead-end (shouldn't happen)
           offset = int(rng.integers(0, degree))
           walk[step] = indices[start_idx + offset]
           step += 1

   return walk, events
   ```

3. Key behavior changes:
   - The function should NEVER return None due to jumper infeasibility (only dead-end
     vertices, which shouldn't exist in connected DCSBM graphs). Remove the compliance
     post-hoc validation at the end of the function since compliance is now guaranteed
     by construction.
   - When a spliced path passes through another jumper vertex, that jumper's path gets
     spliced in the next iteration of the while loop (when `step` advances past the
     current splice and `walk[step-1]` lands on the nested jumper). Actually, since we
     are splicing a fixed path, if the path passes through another jumper at some
     intermediate position, the walk continues from the END of the current splice.
     The nested jumper is "consumed" by the splice without its own path being applied.
     This is correct behavior: the splice guarantees the FIRST jumper's constraint is
     met. The nested jumper's event should still be recorded though.
   - After splicing, scan `path[1:1+splice_len]` for any intermediate jumper vertices
     and record their events too (with the caveat that the splice does NOT guarantee
     the nested jumper's constraint — but these nested events are informational only,
     since the walk is predetermined through the splice). Actually, for correctness,
     do NOT record events for intermediate jumper vertices encountered during a splice.
     Only the primary jumper that triggered the splice gets an event. Intermediate
     jumpers inside a splice segment have their constraints implicitly "overridden"
     by the splice. This avoids false compliance expectations. The walk continues from
     the splice end, and any jumper encountered AFTER the splice ends will trigger
     its own splice.

4. Update `generate_walks()`:
   - Call `precompute_viable_paths()` (imported from compliance) in addition to the
     existing setup. Pass the master_rng with a derived seed for path precomputation
     (use `np.random.default_rng(seed + 4000)` to keep it independent of walk seeds).
   - Pass `viable_paths` to `generate_single_guided_walk()` instead of `path_counts`.
   - Remove the `precompute_path_counts()` call since it is no longer needed.
   - Remove the import of `precompute_path_counts` and `guided_step` from the imports.
     Add import of `precompute_viable_paths`.
   - In Phase 1 (jumper-seeded walks): remove the while-loop retry logic. Since
     `generate_single_guided_walk` should never return None for jumper infeasibility,
     the discard/retry count should be zero. Keep a safety check: if result is None
     (dead-end vertex), log an error and retry with a different start vertex.
   - In Phase 2 (random-start walks): same change — the re-run as guided should
     always succeed, so the inner while-True retry loop for replacements should
     rarely if ever fire. Keep it as a safety net but log if it does.
   - Remove `path_counts` from `generate_single_guided_walk` call sites.

5. In `_validate_walks()`: keep the compliance validation (section 2) as a
   belt-and-suspenders check. The walks should always pass now, but the validation
   confirms it. This is cheap and provides defense-in-depth.

**Part B: Update tests**

In `tests/test_walk_generator.py`:

1. Update `TestGuidedWalkCompliance.test_guided_walk_compliance()`:
   - Replace `precompute_path_counts` with `precompute_viable_paths` call.
   - Pass `viable_paths` instead of `path_counts` to `generate_single_guided_walk()`.
   - Since walks should now ALWAYS be compliant (never return None for infeasibility),
     assert that ALL generated walks are compliant, not just "some".
   - Update import to use `precompute_viable_paths` instead of `precompute_path_counts`.

2. Update `TestNestedJumperCompliance.test_nested_jumper_compliance()`:
   - Same changes: use `precompute_viable_paths` instead of `precompute_path_counts`.
   - Pass `viable_paths` to the walk generator.

3. Add a new test class `TestViablePathPrecomputation`:
   ```python
   class TestViablePathPrecomputation:
       """Test that viable path precomputation works correctly."""

       def test_viable_paths_exist_for_all_jumpers(self) -> None:
           """Every jumper must have at least one viable path."""
           graph = _make_small_graph(n=20, K=2, seed=123)
           jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3)
           rng = np.random.default_rng(42)
           viable_paths = precompute_viable_paths(
               graph.adjacency, graph.block_assignments, [jumper], rng,
               n_paths_per_jumper=50,
           )
           assert 0 in viable_paths
           assert len(viable_paths[0]) > 0

       def test_viable_paths_reach_target_block(self) -> None:
           """All pre-computed paths must end in the target block."""
           graph = _make_small_graph(n=20, K=2, seed=123)
           jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3)
           rng = np.random.default_rng(42)
           viable_paths = precompute_viable_paths(
               graph.adjacency, graph.block_assignments, [jumper], rng,
               n_paths_per_jumper=50,
           )
           for path in viable_paths[0]:
               assert len(path) == 4  # r+1
               assert path[0] == 0  # starts at jumper vertex
               assert graph.block_assignments[path[-1]] == 1  # ends in target block

       def test_viable_paths_follow_valid_edges(self) -> None:
           """Each step in a pre-computed path must follow a valid edge."""
           graph = _make_small_graph(n=20, K=2, seed=123)
           jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3)
           rng = np.random.default_rng(42)
           viable_paths = precompute_viable_paths(
               graph.adjacency, graph.block_assignments, [jumper], rng,
               n_paths_per_jumper=50,
           )
           indptr = graph.adjacency.indptr
           indices = graph.adjacency.indices
           for path in viable_paths[0]:
               for i in range(len(path) - 1):
                   u, v = path[i], path[i + 1]
                   neighbors = indices[indptr[u]:indptr[u + 1]]
                   assert v in neighbors, f"Edge {u}->{v} not in graph"

       def test_no_discard_with_viable_paths(self) -> None:
           """Walks using viable paths should never be discarded."""
           graph = _make_small_graph(n=20, K=2, seed=123)
           jumper = JumperInfo(vertex_id=0, source_block=0, target_block=1, r=3)
           config = _make_small_config(n=20)
           result = generate_walks(
               graph, [jumper], config, seed=42,
               target_n_walks=30, min_jumper_fraction=0.3,
           )
           # All walks with events must be compliant
           for wi, walk_events in enumerate(result.events):
               for event in walk_events:
                   if event.expected_arrival_step < result.walks.shape[1]:
                       actual_block = graph.block_assignments[
                           result.walks[wi, event.expected_arrival_step]
                       ]
                       assert actual_block == event.target_block, (
                           f"Walk {wi}: expected block {event.target_block} "
                           f"at step {event.expected_arrival_step}, got {actual_block}"
                       )
   ```

4. Update imports at the top of the test file:
   - Remove `precompute_path_counts` from the compliance import.
   - Add `precompute_viable_paths` to the compliance import.
  </action>
  <verify>
Run: `cd /root/Repos/dcsbm-transformer && python -m pytest tests/test_walk_generator.py -x -v 2>&1 | tail -40`
  </verify>
  <done>
All tests in test_walk_generator.py pass including the new TestViablePathPrecomputation
tests. Guided walks are compliant by construction with zero discards. The existing
reproducibility, edge validity, batch unguided, and infeasible walk tests all still pass.
  </done>
</task>

</tasks>

<verification>
```bash
# Run all walk-related tests
cd /root/Repos/dcsbm-transformer && python -m pytest tests/test_walk_generator.py tests/test_walk_corpus.py -x -v

# Run the full test suite to check for regressions
cd /root/Repos/dcsbm-transformer && python -m pytest tests/ -x --timeout=120

# Quick smoke test: generate walks with the anchor config
cd /root/Repos/dcsbm-transformer && python -c "
from src.config.defaults import ANCHOR_CONFIG
from src.graph.dcsbm import generate_dcsbm_graph
from src.graph.jumpers import designate_jumpers
from src.walk.generator import generate_walks
import numpy as np

graph = generate_dcsbm_graph(ANCHOR_CONFIG)
rng = np.random.default_rng(ANCHOR_CONFIG.seed)
jumpers = designate_jumpers(graph, ANCHOR_CONFIG, rng)
result = generate_walks(graph, jumpers, ANCHOR_CONFIG, seed=42, target_n_walks=100)
print(f'Generated {result.walks.shape[0]} walks, {sum(len(e) for e in result.events)} events')
# Verify all events are compliant
for wi, events in enumerate(result.events):
    for ev in events:
        if ev.expected_arrival_step < result.walks.shape[1]:
            actual = graph.block_assignments[result.walks[wi, ev.expected_arrival_step]]
            assert actual == ev.target_block, f'Compliance failure walk {wi}'
print('All walks compliant by construction')
"
```
</verification>

<success_criteria>
- precompute_viable_paths() exists and generates path pools for all jumpers
- generate_single_guided_walk() uses path splicing instead of guided_step()
- Zero discard rate for jumper constraint infeasibility (dead-end discards still possible but should not occur on connected graphs)
- All existing tests pass (edge validity, compliance, reproducibility, batch walks, nested jumpers)
- New tests verify path precomputation correctness (target block, edge validity, coverage)
- Full test suite passes with no regressions
</success_criteria>

<output>
After completion, create `.planning/quick/1-guarantee-walk-generation-by-pre-computi/1-SUMMARY.md`
</output>
