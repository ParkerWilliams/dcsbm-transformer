"""Result schema validation and writing.

Uses a Python validation function (not jsonschema) to check required fields,
types, and array length consistency before writing result.json files.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config.experiment import ExperimentConfig
from src.config.hashing import full_config_hash, graph_config_hash
from src.reproducibility.git_hash import get_git_hash
from src.results.experiment_id import generate_experiment_id

REQUIRED_TOP_FIELDS = {
    "schema_version",
    "experiment_id",
    "timestamp",
    "description",
    "tags",
    "config",
    "metrics",
}

REQUIRED_METRICS_FIELDS = {"scalars"}


def validate_result(result: dict[str, Any]) -> list[str]:
    """Validate a result dict against the project schema.

    Returns a list of error strings. An empty list means the result is valid.

    Checks:
    - All required top-level fields are present
    - metrics.scalars is present
    - schema_version is a string
    - tags is a list
    - config is a dict
    - timestamp contains 'T' (basic ISO 8601 check)
    - Sequence array lengths match token count
    """
    errors: list[str] = []

    # Required top-level fields
    missing = REQUIRED_TOP_FIELDS - set(result.keys())
    if missing:
        errors.append(f"Missing required top-level fields: {sorted(missing)}")

    # schema_version type check
    if "schema_version" in result and not isinstance(result["schema_version"], str):
        errors.append("schema_version must be a string")

    # tags type check
    if "tags" in result and not isinstance(result["tags"], list):
        errors.append("tags must be a list")

    # config type check
    if "config" in result and not isinstance(result["config"], dict):
        errors.append("config must be a dict")

    # timestamp format check
    if "timestamp" in result:
        ts = result["timestamp"]
        if not isinstance(ts, str):
            errors.append("timestamp must be a string")
        else:
            try:
                datetime.fromisoformat(ts)
            except ValueError:
                errors.append("timestamp must be in ISO 8601 format")

    # metrics.scalars required
    if "metrics" in result:
        if not isinstance(result["metrics"], dict):
            errors.append("metrics must be a dict")
        elif "scalars" not in result["metrics"]:
            errors.append("metrics.scalars is required")

    # Sequence array length consistency
    for seq in result.get("sequences", []):
        tokens = seq.get("tokens", [])
        seq_id = seq.get("sequence_id", "unknown")
        for array_key in ["token_logprobs", "token_entropy"]:
            arr = seq.get(array_key)
            if arr is not None and len(arr) != len(tokens):
                errors.append(
                    f"{array_key} length ({len(arr)}) != tokens length "
                    f"({len(tokens)}) in sequence {seq_id}"
                )

    return errors


def write_result(
    config: ExperimentConfig,
    metrics: dict[str, Any],
    sequences: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    token_metrics: dict[str, dict[str, Any]] | None = None,
    results_dir: str = "results",
) -> str:
    """Write result.json and optional token_metrics.npz.

    Creates a directory at results/{experiment_id}/ containing result.json
    and optionally token_metrics.npz for large per-step metric arrays.

    Args:
        config: The experiment configuration.
        metrics: Metrics dict (must include 'scalars' key).
        sequences: Optional list of sequence dicts with tokens and per-token data.
        metadata: Optional additional metadata to merge into the metadata block.
        token_metrics: Optional dict mapping sequence_id to metric arrays.
            Format: {seq_id: {metric_name: np.ndarray}}
        results_dir: Base directory for result output.

    Returns:
        The generated experiment_id string.

    Raises:
        ValueError: If the assembled result fails validation.
    """
    experiment_id = generate_experiment_id(config)
    out_dir = Path(results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": config.description,
        "tags": list(config.tags),
        "config": asdict(config),
        "metrics": metrics,
        "sequences": sequences or [],
        "metadata": {
            "code_hash": get_git_hash(),
            "config_hash": full_config_hash(config),
            "graph_config_hash": graph_config_hash(config),
            **(metadata or {}),
        },
    }

    # Validate before writing
    errors = validate_result(result)
    if errors:
        raise ValueError(
            "Result validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Write JSON
    result_path = out_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Write npz for large token-level arrays
    if token_metrics:
        npz_path = out_dir / "token_metrics.npz"
        flat: dict[str, Any] = {}
        for seq_id, metrics_dict in token_metrics.items():
            for metric_name, arr in metrics_dict.items():
                flat[f"{seq_id}/{metric_name}"] = arr
        np.savez_compressed(str(npz_path), **flat)

    return experiment_id


def load_result(result_path: str | Path) -> dict[str, Any]:
    """Load and validate a result.json file.

    Args:
        result_path: Path to the result.json file.

    Returns:
        The loaded and validated result dict.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the loaded result fails validation.
    """
    path = Path(result_path)
    with open(path) as f:
        result = json.load(f)

    errors = validate_result(result)
    if errors:
        raise ValueError(
            f"Result validation failed for {path}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return result
