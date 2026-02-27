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

    # Optional split_assignment validation (backward compatible)
    if "metrics" in result and isinstance(result["metrics"], dict):
        scalars = result["metrics"].get("scalars", {})
        if isinstance(scalars, dict) and "split_assignment" in scalars:
            sa = scalars["split_assignment"]
            if not isinstance(sa, dict):
                errors.append("metrics.scalars.split_assignment must be a dict")
            else:
                for field in ["split_seed", "n_exploratory", "n_confirmatory"]:
                    if field not in sa:
                        errors.append(
                            f"metrics.scalars.split_assignment missing field: {field}"
                        )

    # Optional null_model validation (Phase 12, backward compatible)
    if "metrics" in result and isinstance(result["metrics"], dict):
        null_model = result["metrics"].get("null_model")
        if null_model is not None:
            if not isinstance(null_model, dict):
                errors.append("metrics.null_model must be a dict")
            else:
                # Validate required sub-blocks
                for block_name in ["config", "by_lookback", "aggregate"]:
                    if block_name not in null_model:
                        errors.append(
                            f"metrics.null_model missing required block: {block_name}"
                        )
                # Validate config fields
                nm_config = null_model.get("config", {})
                if isinstance(nm_config, dict):
                    for nm_field in [
                        "n_null_walks",
                        "n_violation_walks",
                        "null_seed",
                        "alpha",
                    ]:
                        if nm_field not in nm_config:
                            errors.append(
                                f"metrics.null_model.config missing field: {nm_field}"
                            )
                # Validate aggregate fields
                nm_agg = null_model.get("aggregate", {})
                if isinstance(nm_agg, dict):
                    for nm_field in ["n_lookbacks_tested", "signal_exceeds_noise"]:
                        if nm_field not in nm_agg:
                            errors.append(
                                f"metrics.null_model.aggregate missing field: {nm_field}"
                            )

    # Optional pr_curves validation (Phase 13, backward compatible)
    if "metrics" in result and isinstance(result["metrics"], dict):
        pr_curves = result["metrics"].get("pr_curves")
        if pr_curves is not None:
            if not isinstance(pr_curves, dict):
                errors.append("metrics.pr_curves must be a dict")
            else:
                if "by_r_value" not in pr_curves:
                    errors.append(
                        "metrics.pr_curves missing required block: by_r_value"
                    )

    # Optional calibration validation (Phase 13, backward compatible)
    if "metrics" in result and isinstance(result["metrics"], dict):
        calibration = result["metrics"].get("calibration")
        if calibration is not None:
            if not isinstance(calibration, dict):
                errors.append("metrics.calibration must be a dict")
            else:
                if "by_r_value" not in calibration:
                    errors.append(
                        "metrics.calibration missing required block: by_r_value"
                    )
                cal_config = calibration.get("config", {})
                if isinstance(cal_config, dict) and "n_bins" not in cal_config:
                    errors.append(
                        "metrics.calibration.config missing field: n_bins"
                    )

    # Optional svd_benchmark validation (Phase 13, backward compatible)
    if "metrics" in result and isinstance(result["metrics"], dict):
        svd_bench = result["metrics"].get("svd_benchmark")
        if svd_bench is not None:
            if not isinstance(svd_bench, dict):
                errors.append("metrics.svd_benchmark must be a dict")
            else:
                if "by_target" not in svd_bench:
                    errors.append(
                        "metrics.svd_benchmark missing required block: by_target"
                    )
                by_target = svd_bench.get("by_target", {})
                if isinstance(by_target, dict):
                    for target_name, t_data in by_target.items():
                        if not isinstance(t_data, dict):
                            errors.append(
                                f"metrics.svd_benchmark.by_target.{target_name} "
                                "must be a dict"
                            )
                        else:
                            for field in ["matrix_shape", "full_svd_ms"]:
                                if field not in t_data:
                                    errors.append(
                                        f"metrics.svd_benchmark.by_target."
                                        f"{target_name} missing field: {field}"
                                    )

    # Optional perturbation_bound validation (Phase 14, backward compatible)
    if "metrics" in result and isinstance(result["metrics"], dict):
        pb = result["metrics"].get("perturbation_bound")
        if pb is not None:
            if not isinstance(pb, dict):
                errors.append("metrics.perturbation_bound must be a dict")
            else:
                if "by_magnitude" not in pb:
                    errors.append(
                        "metrics.perturbation_bound missing required block: "
                        "by_magnitude"
                    )
                for pb_field in [
                    "tightness_ratio",
                    "violation_rate",
                    "bound_verified",
                ]:
                    if pb_field not in pb:
                        errors.append(
                            f"metrics.perturbation_bound missing field: {pb_field}"
                        )

    # Optional spectrum_analysis validation (Phase 15, backward compatible)
    if "metrics" in result and isinstance(result["metrics"], dict):
        spectrum_analysis = result["metrics"].get("spectrum_analysis")
        if spectrum_analysis is not None:
            if not isinstance(spectrum_analysis, dict):
                errors.append("metrics.spectrum_analysis must be a dict")
            else:
                if "status" not in spectrum_analysis:
                    errors.append(
                        "metrics.spectrum_analysis missing required field: status"
                    )
                if "by_r_value" not in spectrum_analysis:
                    errors.append(
                        "metrics.spectrum_analysis missing required block: by_r_value"
                    )

    # Optional compliance_curve validation (Phase 15, backward compatible)
    if "metrics" in result and isinstance(result["metrics"], dict):
        compliance_curve = result["metrics"].get("compliance_curve")
        if compliance_curve is not None:
            if not isinstance(compliance_curve, dict):
                errors.append("metrics.compliance_curve must be a dict")
            else:
                if "curve" not in compliance_curve:
                    errors.append(
                        "metrics.compliance_curve missing required block: curve"
                    )

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
