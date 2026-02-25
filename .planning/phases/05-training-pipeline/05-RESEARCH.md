# Phase 5: Training Pipeline - Research

**Researched:** 2026-02-25
**Domain:** PyTorch training loop, checkpointing, compliance evaluation, early stopping
**Confidence:** HIGH

## Summary

Phase 5 implements the training pipeline that takes the walk corpus (Phase 3) and transformer model (Phase 4) and trains via cross-entropy next-token prediction. The core components are: (1) a training loop with AdamW optimizer and cosine LR schedule with warmup, (2) per-epoch compliance evaluation via greedy generation on held-out walks, (3) a hard sufficiency gate (edge >95%, rule >80%) that stops training on pass, (4) checkpoint/resume for preemption recovery, and (5) curve logging to result.json.

All building blocks exist in the codebase: `TransformerLM` with `ExtractionMode.NONE` for lean training, `ExperimentConfig` with training hyperparameters, `generate_corpus` for train/eval split, `write_result` for output, and `set_seed`/`seed_worker` for reproducibility. The implementation is standard PyTorch training with no exotic dependencies.

**Primary recommendation:** Implement as two modules: `src/training/trainer.py` (training loop, optimizer, scheduler, evaluation) and `src/training/checkpoint.py` (save/load/resume logic). Keep the gate logic inline in the trainer rather than a separate module since it's tightly coupled to the training loop's early stopping behavior.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Fixed epochs over the walk corpus (not fixed step count)
- Batch size: 64 sequences
- Peak learning rate: 3e-4 with AdamW optimizer
- Cosine LR schedule with 10% linear warmup
- Max training budget: 50 epochs
- Compliance measured via greedy generation (argmax decoding) on held-out eval walks
- Generate 1000 sequences per evaluation
- Evaluate every epoch (no skipping early epochs)
- Stop training immediately when gate passes (edge >95%, rule >80%)
- Failed configs (gate not passed after 50 epochs) are flagged with failure metadata in result.json and excluded from SVD analysis
- Checkpoint every epoch: model weights, optimizer state, LR scheduler state, epoch counter, RNG states
- Retention: keep last 3 checkpoints + gate-pass checkpoint (rolling window)
- Storage: `results/{experiment_id}/checkpoints/`
- Resume: full restore (weights + optimizer + scheduler + RNG) for seamless continuation after preemption
- Training loss logged every step (full resolution)
- Edge compliance and rule compliance logged every epoch (from gate evaluation)
- No additional metrics (no LR curve, no perplexity, no top-k accuracy)
- Storage: inline arrays in result.json curves block (`curves.train_loss`, `curves.edge_compliance`, `curves.rule_compliance`)
- Console: tqdm progress bar per epoch + one-line epoch summary (loss, compliance)

### Claude's Discretion
- Weight decay value (standard AdamW default ~0.01 expected)
- Gradient clipping threshold
- Cosine schedule min LR
- Data loading / shuffling strategy
- Exact tqdm format and summary line format

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| TRNG-01 | Cross-entropy next-token prediction with AdamW and cosine LR schedule | Standard PyTorch: `nn.CrossEntropyLoss`, `torch.optim.AdamW`, `torch.optim.lr_scheduler.CosineAnnealingLR` with linear warmup via `SequentialLR` or `LambdaLR` |
| TRNG-03 | Checkpoint model weights, optimizer state, training step periodically and on gate pass | `torch.save`/`torch.load` with state dicts; rolling window retention (keep last 3 + gate-pass) |
| TRNG-04 | Log training loss and compliance curves per step, stored in result.json curves block | Accumulate lists in memory; write via existing `write_result` with `curves` key in metrics |
| TRNG-05 | Sufficiency gate: edge compliance >95% and rule compliance >80% on held-out walks | Greedy generation (argmax) on eval walks, compute edge validity and block compliance per step |
| TRNG-06 | Failed configs flagged with failure metadata in result.json, excluded from SVD | Add `gate_passed: false`, `failure_reason`, `final_edge_compliance`, `final_rule_compliance` to result metadata |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.0 | Model, optimizer, scheduler, checkpointing | Already in project deps; provides all training primitives |
| torch.optim.AdamW | built-in | Optimizer with decoupled weight decay | Standard for transformer training |
| torch.optim.lr_scheduler | built-in | Cosine annealing + linear warmup | `CosineAnnealingLR` or `LambdaLR` for combined schedule |
| tqdm | >=4.0 | Progress bars | Standard Python progress display |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.utils.data.DataLoader | built-in | Batched data loading with shuffling | Training data iteration |
| torch.utils.data.TensorDataset | built-in | Wrap numpy walks as torch tensors | Dataset for DataLoader |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom LR schedule | HuggingFace `get_cosine_schedule_with_warmup` | Extra dependency; easy to implement with `LambdaLR` |
| Manual batching | torch DataLoader | DataLoader handles shuffling, worker seeding, pinned memory |

**Installation:**
```bash
pip install tqdm
```
(torch, numpy already installed)

## Architecture Patterns

### Recommended Project Structure
```
src/
├── training/
│   ├── __init__.py          # Public API exports
│   ├── trainer.py           # Training loop, evaluation, gate check
│   ├── checkpoint.py        # Save/load/resume/rolling retention
│   └── data.py              # Dataset wrapping, DataLoader creation
```

### Pattern 1: Epoch-Based Training Loop with Early Stopping
**What:** Train for up to `max_epochs` epochs, evaluating compliance after each. Stop immediately on gate pass.
**When to use:** Fixed-epoch budget with early stopping criterion.
**Example:**
```python
for epoch in range(start_epoch, max_epochs):
    model.train()
    epoch_losses = []
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch[:, :-1], mode=ExtractionMode.NONE)
        loss = F.cross_entropy(
            output.logits.reshape(-1, vocab_size),
            batch[:, 1:].reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # step-level for cosine schedule
        epoch_losses.append(loss.item())

    # Evaluate compliance
    edge_comp, rule_comp = evaluate_compliance(model, eval_walks, graph_data, jumpers)

    # Checkpoint every epoch
    save_checkpoint(...)

    # Gate check
    if edge_comp > 0.95 and rule_comp > 0.80:
        save_gate_checkpoint(...)
        break
```

### Pattern 2: Greedy Generation for Compliance
**What:** Generate sequences via argmax decoding, then check edge validity and rule compliance.
**When to use:** Evaluating whether the model has learned the graph structure and jumper rules.
**Example:**
```python
def greedy_generate(model, start_tokens, length, device):
    """Generate sequences via argmax decoding."""
    model.eval()
    with torch.no_grad():
        generated = start_tokens.clone()  # [B, 1] seed tokens
        for _ in range(length - 1):
            output = model(generated[:, -max_seq_len:], mode=ExtractionMode.NONE)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated
```

### Pattern 3: Rolling Checkpoint Retention
**What:** Keep only the last N checkpoints plus the gate-pass checkpoint.
**When to use:** Preventing disk space exhaustion during long training.
**Example:**
```python
def save_with_retention(checkpoint_dir, epoch, state, max_keep=3):
    path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(state, path)

    # Delete old checkpoints beyond retention window
    all_ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    gate_ckpt = checkpoint_dir / "checkpoint_gate.pt"
    for old in all_ckpts[:-max_keep]:
        if old != gate_ckpt:
            old.unlink()
```

### Anti-Patterns to Avoid
- **Computing compliance during training forward pass:** Compliance requires greedy generation which is separate from teacher-forced training. Never mix these modes.
- **Stepping scheduler per epoch with cosine schedule:** When using cosine annealing, step per optimization step, not per epoch, to get smooth cosine decay. Total steps = num_epochs * steps_per_epoch.
- **Saving raw model instead of state_dict:** Always save `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()` for portability.
- **Forgetting RNG states in checkpoint:** Must save `torch.random.get_rng_state()`, `torch.cuda.get_rng_state_all()`, `numpy.random.get_state()`, `random.getstate()` for perfect resume.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Learning rate warmup + cosine | Custom LR function | `LambdaLR` with piecewise lambda | Edge cases with step counting, resume |
| Gradient clipping | Manual norm computation | `torch.nn.utils.clip_grad_norm_` | Handles edge cases, NaN gradients |
| Batched data iteration | Manual index slicing | `DataLoader` with `TensorDataset` | Handles shuffling, pinning, worker seeding |

**Key insight:** PyTorch's built-in training utilities handle reproducibility edge cases (worker seeding, generator state) that are easy to get wrong in custom code.

## Common Pitfalls

### Pitfall 1: Cosine Schedule Step Granularity
**What goes wrong:** Calling `scheduler.step()` per epoch instead of per optimizer step results in a staircase LR curve with only 50 steps (one per epoch) instead of smooth cosine decay.
**Why it happens:** Confusion between epoch-level and step-level schedulers.
**How to avoid:** Set `T_max = num_epochs * steps_per_epoch` and call `scheduler.step()` after every `optimizer.step()`.
**Warning signs:** LR appears constant for long stretches.

### Pitfall 2: Warmup Step Counting
**What goes wrong:** Linear warmup fraction of 10% should be 10% of total training steps, not 10% of epochs.
**Why it happens:** Mixing epoch and step units.
**How to avoid:** `warmup_steps = int(0.1 * total_steps)` where `total_steps = max_epochs * steps_per_epoch`.

### Pitfall 3: DataLoader Reproducibility
**What goes wrong:** Shuffled DataLoader produces different batches on resume even with same seed.
**Why it happens:** DataLoader's internal generator state is not saved/restored.
**How to avoid:** Create a `torch.Generator()` seeded deterministically per epoch. On resume, recreate from the epoch number. Or save the generator state in checkpoint.

### Pitfall 4: Greedy Generation Context Window
**What goes wrong:** Generating sequences longer than `max_seq_len` causes assertion error in model.
**Why it happens:** The model's positional embeddings are limited to `max_seq_len`.
**How to avoid:** During greedy generation, always slice to `generated[:, -max_seq_len:]` for each forward pass. For compliance evaluation, generate walks of length matching `walk_length` from the config, windowing the context.

### Pitfall 5: Edge Compliance Requires Graph Adjacency
**What goes wrong:** Computing edge compliance without access to the graph adjacency matrix.
**Why it happens:** Training loop doesn't pass graph data to evaluation function.
**How to avoid:** The compliance evaluator needs: generated sequences, graph adjacency (CSR), block assignments, and jumper info. Pass all of these to the evaluation function.

### Pitfall 6: Cross-Entropy Input Shapes
**What goes wrong:** `F.cross_entropy` expects `[N, C]` predictions and `[N]` targets, not `[B, T, C]` and `[B, T]`.
**Why it happens:** Forgetting to reshape before loss computation.
**How to avoid:** Always `logits.reshape(-1, vocab_size)` and `targets.reshape(-1)`.

### Pitfall 7: Checkpoint Resume with Frozen Dataclass
**What goes wrong:** Trying to modify `ExperimentConfig` (frozen dataclass) during resume.
**Why it happens:** Want to store resume metadata in config.
**How to avoid:** Keep resume state in a separate mutable dict in the checkpoint, not in the config.

## Code Examples

### Combined Warmup + Cosine Schedule
```python
import math

def cosine_with_warmup(step, warmup_steps, total_steps, min_lr_ratio=0.1):
    """LambdaLR function: linear warmup then cosine decay."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)
```

### Complete Checkpoint State
```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "torch_rng_state": torch.random.get_rng_state(),
    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    "numpy_rng_state": numpy.random.get_state(),
    "python_rng_state": random.getstate(),
    "train_losses": train_losses,  # all per-step losses so far
    "edge_compliance_history": edge_compliance_history,
    "rule_compliance_history": rule_compliance_history,
    "gate_passed": False,
}
```

### Edge and Rule Compliance Evaluation
```python
def evaluate_compliance(
    generated: torch.Tensor,  # [B, L] generated token sequences
    adjacency: csr_matrix,
    block_assignments: np.ndarray,
    jumpers: list[JumperInfo],
    r: int,
) -> tuple[float, float]:
    """Compute edge compliance and rule compliance on generated sequences."""
    seqs = generated.cpu().numpy()
    indptr = adjacency.indptr
    indices = adjacency.indices

    total_edges = 0
    valid_edges = 0
    total_rule_checks = 0
    rule_compliant = 0

    jumper_set = {j.vertex_id: j for j in jumpers}

    for seq in seqs:
        for t in range(len(seq) - 1):
            u, v = int(seq[t]), int(seq[t + 1])
            neighbors = indices[indptr[u]:indptr[u + 1]]
            total_edges += 1
            if v in neighbors:
                valid_edges += 1

            # Rule check: if u is a jumper, check step t+r
            if u in jumper_set and t + r < len(seq):
                total_rule_checks += 1
                target_block = jumper_set[u].target_block
                actual_block = int(block_assignments[int(seq[t + r])])
                if actual_block == target_block:
                    rule_compliant += 1

    edge_comp = valid_edges / max(1, total_edges)
    rule_comp = rule_compliant / max(1, total_rule_checks)
    return edge_comp, rule_comp
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual LR decay | LambdaLR with cosine function | PyTorch 1.x+ | Cleaner, composable schedules |
| `torch.save(model)` | `torch.save(model.state_dict())` | PyTorch best practice | Portable, version-safe checkpoints |
| Fixed seed at start only | Full RNG state save/restore | Training reproducibility research | Perfect resume after preemption |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_training.py -x` |
| Full suite command | `pytest tests/ -x` |
| Estimated runtime | ~15 seconds (small model, few epochs) |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRNG-01 | Cross-entropy training with AdamW + cosine schedule | unit + integration | `pytest tests/test_training.py::test_training_loss_decreases -x` | No - Wave 0 gap |
| TRNG-03 | Checkpoint save/load/resume | unit | `pytest tests/test_training.py::test_checkpoint_resume -x` | No - Wave 0 gap |
| TRNG-04 | Curve logging to result.json | unit | `pytest tests/test_training.py::test_curve_logging -x` | No - Wave 0 gap |
| TRNG-05 | Sufficiency gate evaluation | unit | `pytest tests/test_training.py::test_sufficiency_gate -x` | No - Wave 0 gap |
| TRNG-06 | Failed config flagging | unit | `pytest tests/test_training.py::test_failed_config_metadata -x` | No - Wave 0 gap |

### Nyquist Sampling Rate
- **Minimum sample interval:** After every committed task -> run: `pytest tests/test_training.py -x`
- **Full suite trigger:** Before merging final task of any plan wave
- **Phase-complete gate:** Full suite green before verification
- **Estimated feedback latency per task:** ~15 seconds

### Wave 0 Gaps (must be created before implementation)
- [ ] `tests/test_training.py` -- covers TRNG-01, TRNG-03, TRNG-04, TRNG-05, TRNG-06
- [ ] `src/training/__init__.py` -- package init

*(tqdm is the only new dependency needed)*

## Open Questions

1. **Walk-to-sequence conversion for DataLoader**
   - What we know: Walks are numpy arrays of shape `(num_walks, walk_length)`. The model's `max_seq_len` is `w` (context window). Walks are length `walk_length` (e.g., 256) but context window is `w` (e.g., 64).
   - What's unclear: Whether to chunk each walk into `walk_length / w` subsequences or use sliding window.
   - Recommendation: Chunk each walk into non-overlapping subsequences of length `w+1` (input is `w` tokens, target is shifted by 1). This is simpler and avoids bias from overlapping windows. Each walk of length `L` produces `floor(L / (w+1))` training sequences, but since `w+1` may not divide evenly, use `floor(L / w) - 1` windows or simply use non-overlapping chunks.

2. **Greedy generation seeding for compliance**
   - What we know: Need to generate 1000 sequences per evaluation. Need seed tokens to start generation.
   - What's unclear: Where seed tokens come from.
   - Recommendation: Use the first token of each eval walk as the seed token. Sample 1000 eval walks, take their first token, generate `walk_length` steps via argmax.

## Sources

### Primary (HIGH confidence)
- PyTorch documentation: `torch.optim.AdamW`, `torch.optim.lr_scheduler`, `torch.save`/`torch.load`
- Existing codebase: `src/model/transformer.py`, `src/config/experiment.py`, `src/results/schema.py`, `src/walk/corpus.py`, `src/reproducibility/seed.py`

### Secondary (MEDIUM confidence)
- NanoGPT reference implementation patterns (Karpathy) for training loop structure
- PyTorch reproducibility guide for RNG state management

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all PyTorch built-ins, no external dependencies beyond tqdm
- Architecture: HIGH - straightforward training loop, well-understood patterns
- Pitfalls: HIGH - common PyTorch training pitfalls, well-documented

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (stable domain, no fast-moving APIs)
