"""Reproduction block builder for experiment reports.

Extracts git commit hash and configuration from result.json metadata
to build copy-pasteable reproduction commands. The git hash is captured
at experiment runtime (not at report generation time) per project decision.
"""

from typing import Any


def build_reproduction_block(result: dict[str, Any]) -> dict[str, Any]:
    """Build a reproduction block from experiment result data.

    Extracts the code hash captured at experiment runtime and key
    configuration parameters to assemble git checkout and CLI run
    commands for reproducing the experiment.

    Args:
        result: Parsed result.json dict with ``metadata`` and ``config`` keys.

    Returns:
        Dict with keys:
        - ``checkout_cmd``: git checkout command for the experiment's code version
        - ``run_cmd``: CLI command with seed and key config params
        - ``seed``: The random seed used
        - ``is_dirty``: True if the working tree had uncommitted changes
        - ``dirty_warning``: Warning string if dirty, else None
    """
    metadata = result.get("metadata", {})
    config = result.get("config", {})

    # Extract and clean git hash
    code_hash = metadata.get("code_hash", "unknown")
    is_dirty = code_hash.endswith("-dirty")
    clean_hash = code_hash.removesuffix("-dirty")

    checkout_cmd = f"git checkout {clean_hash}"

    # Extract seed from config (may be nested under training or top-level)
    seed = config.get("seed")
    if seed is None:
        training = config.get("training", {})
        seed = training.get("seed", 42)

    # Build run command with key parameters
    run_parts = ["python -m src.training.pipeline"]
    run_parts.append(f"--seed {seed}")

    # Add key config params if available
    graph = config.get("graph", {})
    training = config.get("training", {})
    model = config.get("model", {})

    if graph.get("n"):
        run_parts.append(f"--n {graph['n']}")
    if graph.get("K"):
        run_parts.append(f"--K {graph['K']}")
    if training.get("num_epochs"):
        run_parts.append(f"--num_epochs {training['num_epochs']}")
    if model.get("d_model"):
        run_parts.append(f"--d_model {model['d_model']}")

    run_cmd = " \\\n  ".join(run_parts)

    dirty_warning = (
        "WARNING: The working tree had uncommitted changes when this "
        "experiment was run. Results may not be exactly reproducible."
    ) if is_dirty else None

    return {
        "checkout_cmd": checkout_cmd,
        "run_cmd": run_cmd,
        "seed": seed,
        "is_dirty": is_dirty,
        "dirty_warning": dirty_warning,
    }
