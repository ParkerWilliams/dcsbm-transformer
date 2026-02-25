# Phase 5: Training Pipeline - Context

**Gathered:** 2026-02-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Train the transformer on walk corpus data using cross-entropy next-token prediction. Enforce a sufficiency gate (edge compliance >95%, rule compliance >80%) before any downstream SVD analysis. Checkpoint for resume after preemption. Log training curves to result.json.

Behavioral evaluation, SVD collection, and analysis are separate phases (6-7).

</domain>

<decisions>
## Implementation Decisions

### Training schedule
- Fixed epochs over the walk corpus (not fixed step count)
- Batch size: 64 sequences
- Peak learning rate: 3e-4 with AdamW optimizer
- Cosine LR schedule with 10% linear warmup
- Max training budget: 50 epochs
- Weight decay, gradient clipping: Claude's discretion

### Sufficiency gate
- Compliance measured via greedy generation (argmax decoding) on held-out eval walks
- Generate 1000 sequences per evaluation
- Evaluate every epoch (no skipping early epochs)
- Stop training immediately when gate passes (edge >95%, rule >80%)
- Failed configs (gate not passed after 50 epochs) are flagged with failure metadata in result.json and excluded from SVD analysis

### Checkpoint & resume
- Checkpoint every epoch: model weights, optimizer state, LR scheduler state, epoch counter, RNG states
- Retention: keep last 3 checkpoints + gate-pass checkpoint (rolling window)
- Storage: `results/{experiment_id}/checkpoints/`
- Resume: full restore (weights + optimizer + scheduler + RNG) for seamless continuation after preemption

### Curve logging
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

</decisions>

<specifics>
## Specific Ideas

- 50 epochs is generous — the intent is to give the model plenty of room to learn rule compliance, which is harder than edge compliance
- Greedy generation for compliance (not teacher-forced) because it directly tests what matters: can the model produce valid walks?
- Per-step loss logging enables diagnosing batch-level training dynamics if needed

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-training-pipeline*
*Context gathered: 2026-02-25*
