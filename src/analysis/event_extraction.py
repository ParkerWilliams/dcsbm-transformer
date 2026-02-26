"""Event extraction from evaluation output for predictive horizon analysis.

Extracts jumper encounter events from generated sequences, applies
contamination filtering, and stratifies by r value. Each event records
the walk index, encounter step, resolution step, r value, and outcome
(FOLLOWED or VIOLATED).

Indexing convention (matches behavioral.py):
  - encounter_step: step t where generated[walk, t] is a jumper vertex
  - resolution_step: encounter_step + r (the deadline value in behavioral.py)
  - rule_outcome is recorded at index resolution_step - 1
    (behavioral.py checks: if t + 1 == deadline => rule_outcome[b, t])
  - failure_index[walk] stores the step t where rule_outcome[t] == VIOLATED
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.evaluation.behavioral import RuleOutcome
from src.graph.jumpers import JumperInfo


@dataclass(frozen=True, slots=True)
class AnalysisEvent:
    """A single jumper encounter event for predictive horizon analysis.

    Attributes:
        walk_idx: Index into the sequences array.
        encounter_step: Step where jumper vertex was generated.
        resolution_step: encounter_step + r (where rule resolves).
        r_value: Jump length for this encounter.
        outcome: RuleOutcome.FOLLOWED or RuleOutcome.VIOLATED.
        is_first_violation: True if this is the first violation in its walk.
    """

    walk_idx: int
    encounter_step: int
    resolution_step: int
    r_value: int
    outcome: int
    is_first_violation: bool


def extract_events(
    generated: np.ndarray,
    rule_outcome: np.ndarray,
    failure_index: np.ndarray,
    jumper_map: dict[int, JumperInfo],
) -> list[AnalysisEvent]:
    """Extract all jumper encounter events from evaluation output.

    For each sequence, scans generated tokens to find jumper vertices.
    Cross-references with rule_outcome to determine FOLLOWED/VIOLATED.
    Skips encounters where resolution_step exceeds sequence length.

    Args:
        generated: Generated token IDs, shape [n_sequences, max_steps].
        rule_outcome: Rule compliance per step, shape [n_sequences, max_steps-1].
        failure_index: First rule violation step per sequence, shape [n_sequences].
        jumper_map: Mapping from vertex_id to JumperInfo.

    Returns:
        List of AnalysisEvent records for all encounters found.
    """
    n_sequences, max_steps = generated.shape
    max_outcome_idx = rule_outcome.shape[1]  # max_steps - 1
    events: list[AnalysisEvent] = []

    for walk_idx in range(n_sequences):
        for t in range(max_steps):
            token = int(generated[walk_idx, t])
            if token not in jumper_map:
                continue

            jumper = jumper_map[token]
            resolution_step = t + jumper.r
            outcome_idx = resolution_step - 1  # rule_outcome index

            # Skip if resolution goes beyond available rule_outcome
            if outcome_idx >= max_outcome_idx or outcome_idx < 0:
                continue

            outcome_val = int(rule_outcome[walk_idx, outcome_idx])

            # Only record events where rule actually resolved
            if outcome_val == RuleOutcome.NOT_APPLICABLE:
                continue

            # Determine is_first_violation
            is_first = (
                outcome_val == RuleOutcome.VIOLATED
                and int(failure_index[walk_idx]) == outcome_idx
            )

            events.append(
                AnalysisEvent(
                    walk_idx=walk_idx,
                    encounter_step=t,
                    resolution_step=resolution_step,
                    r_value=jumper.r,
                    outcome=outcome_val,
                    is_first_violation=is_first,
                )
            )

    return events


def filter_contaminated_events(
    events: list[AnalysisEvent],
) -> tuple[list[AnalysisEvent], dict]:
    """Apply contamination filter: exclude encounters whose countdown window
    overlaps with a preceding violation's window in the same walk.

    Group by walk_idx, sort by encounter_step. Track last_violation_end
    (resolution step of most recent violation). Exclude encounters where
    encounter_step < last_violation_end.

    IMPORTANT: Only violation events set last_violation_end, not FOLLOWED
    events (per CONTEXT.md: "Successful prior encounters do not contaminate
    subsequent ones").

    Args:
        events: List of AnalysisEvent records (unfiltered).

    Returns:
        Tuple of (filtered_events, audit_dict) where audit_dict contains:
        - total_encounters: Total events before filtering.
        - excluded_encounters: Number of events excluded.
        - exclusion_rate: Fraction excluded.
        - flagged: True if exclusion_rate > 0.3.
        - per_r: Per-r-value breakdown of exclusions.
    """
    total = len(events)
    if total == 0:
        return [], {
            "total_encounters": 0,
            "excluded_encounters": 0,
            "exclusion_rate": 0.0,
            "flagged": False,
            "per_r": {},
        }

    # Group by walk_idx
    by_walk: dict[int, list[AnalysisEvent]] = defaultdict(list)
    for ev in events:
        by_walk[ev.walk_idx].append(ev)

    filtered: list[AnalysisEvent] = []
    excluded = 0

    # Per-r tracking
    per_r_total: dict[int, int] = defaultdict(int)
    per_r_excluded: dict[int, int] = defaultdict(int)

    for walk_idx in sorted(by_walk.keys()):
        walk_events = sorted(by_walk[walk_idx], key=lambda e: e.encounter_step)
        last_violation_end = -1  # resolution step of most recent violation

        for event in walk_events:
            per_r_total[event.r_value] += 1

            if event.encounter_step < last_violation_end:
                excluded += 1
                per_r_excluded[event.r_value] += 1
                continue

            filtered.append(event)

            # Only violations contaminate subsequent encounters
            if event.outcome == RuleOutcome.VIOLATED:
                last_violation_end = event.resolution_step

    exclusion_rate = excluded / total if total > 0 else 0.0

    # Build per_r breakdown
    per_r: dict[int, dict] = {}
    for r_val in sorted(set(per_r_total.keys())):
        r_total = per_r_total[r_val]
        r_excluded = per_r_excluded.get(r_val, 0)
        per_r[r_val] = {
            "total": r_total,
            "excluded": r_excluded,
            "exclusion_rate": r_excluded / r_total if r_total > 0 else 0.0,
        }

    audit = {
        "total_encounters": total,
        "excluded_encounters": excluded,
        "exclusion_rate": exclusion_rate,
        "flagged": exclusion_rate > 0.3,
        "per_r": per_r,
    }

    return filtered, audit


def stratify_by_r(events: list[AnalysisEvent]) -> dict[int, list[AnalysisEvent]]:
    """Group events by r value.

    Each group gets independent AUROC analysis. Never mix different r values
    in the same curve (per CONTEXT.md).

    Args:
        events: List of AnalysisEvent records.

    Returns:
        Dict mapping r_value -> list of events with that r_value.
    """
    result: dict[int, list[AnalysisEvent]] = defaultdict(list)
    for ev in events:
        result[ev.r_value].append(ev)
    return dict(result)
