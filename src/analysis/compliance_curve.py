"""Compliance curve analysis: aggregate compliance across r/w ratio experiments.

Loads multiple result.json files from independent training experiments at
different r values, extracts (r/w ratio, edge compliance, rule compliance,
predictive horizon), and computes aggregate statistics for the compliance
phase transition curve.

Phase 15: Advanced Analysis (COMP-01, COMP-02).
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def load_result_json(result_dir: str | Path) -> dict | None:
    """Load result.json from a result directory.

    Args:
        result_dir: Path to directory containing result.json.

    Returns:
        Parsed dict, or None if file not found or invalid.
    """
    result_path = Path(result_dir) / "result.json"
    if not result_path.exists():
        log.warning("result.json not found in %s", result_dir)
        return None
    try:
        with open(result_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Failed to load %s: %s", result_path, e)
        return None


def extract_compliance_point(result: dict) -> dict | None:
    """Extract a compliance data point from a parsed result.json.

    Args:
        result: Parsed result.json dict.

    Returns:
        Dict with r, w, r_over_w, edge_compliance, rule_compliance,
        predictive_horizon, seed. Returns None if extraction fails.
    """
    try:
        config = result["config"]
        training = config["training"]
        r = training["r"]
        w = training["w"]
        r_over_w = r / w

        metrics = result.get("metrics", {})
        scalars = metrics.get("scalars", {})

        edge_compliance = scalars.get("final_edge_compliance")
        rule_compliance = scalars.get("final_rule_compliance")

        if edge_compliance is None or rule_compliance is None:
            log.warning("Missing compliance scalars in result")
            return None

        # Extract predictive horizon (best across metrics for corresponding r)
        predictive_horizon = None
        pred_horizon_data = metrics.get("predictive_horizon", {})
        by_r_value = pred_horizon_data.get("by_r_value", {})

        # Try exact r match or str(r)
        r_data = by_r_value.get(str(r)) or by_r_value.get(r)
        if r_data:
            by_metric = r_data.get("by_metric", {})
            horizons = []
            for metric_name, metric_data in by_metric.items():
                h = metric_data.get("horizon_j")
                if h is not None and h > 0:
                    horizons.append(h)
            if horizons:
                predictive_horizon = max(horizons)

        seed = config.get("seed", 42)

        return {
            "r": r,
            "w": w,
            "r_over_w": round(r_over_w, 4),
            "edge_compliance": float(edge_compliance),
            "rule_compliance": float(rule_compliance),
            "predictive_horizon": predictive_horizon,
            "seed": seed,
        }
    except (KeyError, TypeError) as e:
        log.warning("Failed to extract compliance point: %s", e)
        return None


def compute_compliance_curve(result_dirs: list[Path | str]) -> list[dict]:
    """Load multiple results and extract compliance points.

    Args:
        result_dirs: List of paths to result directories.

    Returns:
        List of compliance point dicts, sorted by r_over_w ascending.
    """
    points = []
    for result_dir in result_dirs:
        result = load_result_json(result_dir)
        if result is None:
            continue
        point = extract_compliance_point(result)
        if point is not None:
            points.append(point)

    return sorted(points, key=lambda p: p["r_over_w"])


def aggregate_compliance_curve(points: list[dict]) -> dict:
    """Group compliance points by r/w ratio and compute statistics.

    Args:
        points: List of compliance point dicts from compute_compliance_curve.

    Returns:
        Aggregated dict with r_over_w_values, edge_compliance, rule_compliance,
        predictive_horizon (each with mean/std), and n_seeds.
    """
    # Group by r_over_w (rounded to 3 decimal places for floating-point grouping)
    groups: dict[float, list[dict]] = defaultdict(list)
    for p in points:
        key = round(p["r_over_w"], 3)
        groups[key].append(p)

    sorted_keys = sorted(groups.keys())

    r_over_w_values = []
    edge_mean, edge_std = [], []
    rule_mean, rule_std = [], []
    horizon_mean, horizon_std = [], []
    n_seeds_list = []

    for r_w in sorted_keys:
        group = groups[r_w]
        r_over_w_values.append(r_w)
        n_seeds_list.append(len(group))

        edges = [p["edge_compliance"] for p in group]
        rules = [p["rule_compliance"] for p in group]

        edge_mean.append(float(np.mean(edges)))
        edge_std.append(float(np.std(edges)))
        rule_mean.append(float(np.mean(rules)))
        rule_std.append(float(np.std(rules)))

        horizons = [p["predictive_horizon"] for p in group if p["predictive_horizon"] is not None]
        if horizons:
            horizon_mean.append(float(np.mean(horizons)))
            horizon_std.append(float(np.std(horizons)))
        else:
            horizon_mean.append(None)
            horizon_std.append(None)

    return {
        "r_over_w_values": r_over_w_values,
        "edge_compliance": {"mean": edge_mean, "std": edge_std},
        "rule_compliance": {"mean": rule_mean, "std": rule_std},
        "predictive_horizon": {"mean": horizon_mean, "std": horizon_std},
        "n_seeds": n_seeds_list,
    }


def run_compliance_analysis(result_dirs: list[Path | str]) -> dict[str, Any]:
    """Orchestrate compliance curve analysis.

    Args:
        result_dirs: List of paths to result directories.

    Returns:
        Dict with config metadata, aggregated curve, and raw points.
    """
    points = compute_compliance_curve(result_dirs)
    curve = aggregate_compliance_curve(points)

    return {
        "config": {
            "n_result_dirs": len(result_dirs),
            "n_valid_points": len(points),
        },
        "curve": curve,
        "raw_points": points,
    }
