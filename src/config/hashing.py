"""Deterministic config hashing using SHA-256 over sorted JSON."""

import hashlib
import json
from dataclasses import asdict
from typing import Any

from src.config.experiment import ExperimentConfig


def _remove_nested(d: dict[str, Any], field_path: str) -> None:
    """Remove a dotted-path key from a nested dict.

    Example: _remove_nested(d, "graph.n") removes d["graph"]["n"].
    Single-level paths like "seed" remove d["seed"].
    """
    parts = field_path.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            return
        current = current[part]
    current.pop(parts[-1], None)


def config_hash(config: Any, exclude_fields: list[str] | None = None) -> str:
    """Deterministic SHA-256 hash of a config object.

    Args:
        config: Any dataclass instance (or sub-config).
        exclude_fields: Optional list of dotted field paths to exclude.

    Returns:
        First 16 hex characters of the SHA-256 hash.
    """
    d = asdict(config)
    if exclude_fields:
        for field_path in exclude_fields:
            _remove_nested(d, field_path)
    serialized = json.dumps(
        d,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        indent=None,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def graph_config_hash(config: ExperimentConfig) -> str:
    """Hash for graph caching — operates on config.graph only (excludes seed).

    Two configs differing only in seed will produce the same graph hash,
    enabling graph cache sharing across seeds.
    """
    return config_hash(config.graph)


def full_config_hash(config: ExperimentConfig) -> str:
    """Hash for full experiment identity — includes everything including seed."""
    return config_hash(config)
