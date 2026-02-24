"""JSON serialization and deserialization for experiment configs."""

import json
from dataclasses import asdict
from typing import Any

from dacite import from_dict, Config as DaciteConfig

from src.config.experiment import ExperimentConfig


def config_to_json(config: ExperimentConfig) -> str:
    """Serialize an ExperimentConfig to a JSON string.

    Uses sorted keys and 2-space indent for human readability and diffability.
    """
    return json.dumps(asdict(config), indent=2, sort_keys=True)


def config_from_json(json_str: str) -> ExperimentConfig:
    """Deserialize a JSON string to an ExperimentConfig.

    Uses dacite with strict=True to reject unknown keys (catches schema drift)
    and cast=[tuple] to convert JSON arrays back to tuples for tags and sweep fields.
    """
    data = json.loads(json_str)
    return from_dict(
        data_class=ExperimentConfig,
        data=data,
        config=DaciteConfig(
            cast=[tuple],
            check_types=True,
            strict=True,
        ),
    )


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert an ExperimentConfig to a plain dictionary."""
    return asdict(config)


def config_from_dict(d: dict[str, Any]) -> ExperimentConfig:
    """Reconstruct an ExperimentConfig from a plain dictionary."""
    return from_dict(
        data_class=ExperimentConfig,
        data=d,
        config=DaciteConfig(
            cast=[tuple],
            check_types=True,
            strict=True,
        ),
    )
