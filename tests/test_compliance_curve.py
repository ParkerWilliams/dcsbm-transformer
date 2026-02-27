"""Tests for compliance curve analysis and visualization.

Phase 15: Advanced Analysis (COMP-01, COMP-02).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.analysis.compliance_curve import (
    aggregate_compliance_curve,
    compute_compliance_curve,
    extract_compliance_point,
    load_result_json,
    run_compliance_analysis,
)


def _make_result_json(r: int, w: int, edge_comp: float, rule_comp: float,
                      seed: int = 42, horizon: dict | None = None) -> dict:
    """Create a synthetic result.json dict."""
    result = {
        "config": {
            "training": {"r": r, "w": w, "walk_length": 256},
            "seed": seed,
        },
        "metrics": {
            "scalars": {
                "final_edge_compliance": edge_comp,
                "final_rule_compliance": rule_comp,
            },
        },
    }
    if horizon is not None:
        result["metrics"]["predictive_horizon"] = horizon
    return result


def _write_result_json(tmp_path: Path, result: dict, subdir: str) -> Path:
    """Write a result.json to a subdirectory and return the directory path."""
    d = tmp_path / subdir
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "result.json", "w") as f:
        json.dump(result, f)
    return d


# ---------------------------------------------------------------------------
# extract_compliance_point tests
# ---------------------------------------------------------------------------


class TestExtractCompliancePoint:
    def test_valid_extraction(self):
        """Extract compliance point from a valid result dict."""
        result = _make_result_json(r=57, w=64, edge_comp=0.98, rule_comp=0.85)
        point = extract_compliance_point(result)

        assert point is not None
        assert point["r"] == 57
        assert point["w"] == 64
        assert abs(point["r_over_w"] - 57 / 64) < 0.01
        assert point["edge_compliance"] == 0.98
        assert point["rule_compliance"] == 0.85
        assert point["seed"] == 42

    def test_missing_compliance(self):
        """Missing compliance scalars should return None."""
        result = {
            "config": {"training": {"r": 57, "w": 64}, "seed": 42},
            "metrics": {"scalars": {}},
        }
        point = extract_compliance_point(result)
        assert point is None

    def test_missing_config(self):
        """Missing config should return None."""
        result = {"metrics": {"scalars": {"final_edge_compliance": 0.9}}}
        point = extract_compliance_point(result)
        assert point is None

    def test_with_predictive_horizon(self):
        """Extract predictive horizon when available."""
        horizon = {
            "by_r_value": {
                "57": {
                    "by_metric": {
                        "qkt.grassmannian_distance": {"horizon_j": 5},
                        "qkt.spectral_gap_1_2": {"horizon_j": 3},
                    }
                }
            }
        }
        result = _make_result_json(r=57, w=64, edge_comp=0.98, rule_comp=0.85,
                                   horizon=horizon)
        point = extract_compliance_point(result)
        assert point is not None
        assert point["predictive_horizon"] == 5  # max across metrics

    def test_without_predictive_horizon(self):
        """No predictive_horizon data should give None horizon."""
        result = _make_result_json(r=57, w=64, edge_comp=0.98, rule_comp=0.85)
        point = extract_compliance_point(result)
        assert point is not None
        assert point["predictive_horizon"] is None


# ---------------------------------------------------------------------------
# compute_compliance_curve tests
# ---------------------------------------------------------------------------


class TestComputeComplianceCurve:
    def test_sorting_by_r_over_w(self, tmp_path):
        """Points should be sorted by r/w ascending."""
        dirs = []
        for r, subdir in [(96, "exp_r96"), (32, "exp_r32"), (64, "exp_r64")]:
            result = _make_result_json(r=r, w=64, edge_comp=0.9, rule_comp=0.8)
            d = _write_result_json(tmp_path, result, subdir)
            dirs.append(d)

        points = compute_compliance_curve(dirs)
        assert len(points) == 3
        assert points[0]["r_over_w"] < points[1]["r_over_w"] < points[2]["r_over_w"]

    def test_missing_dir_skipped(self, tmp_path):
        """Nonexistent directories are skipped."""
        dirs = [tmp_path / "nonexistent"]
        points = compute_compliance_curve(dirs)
        assert len(points) == 0

    def test_invalid_json_skipped(self, tmp_path):
        """Invalid JSON files are skipped."""
        d = tmp_path / "bad"
        d.mkdir()
        (d / "result.json").write_text("not valid json")
        points = compute_compliance_curve([d])
        assert len(points) == 0


# ---------------------------------------------------------------------------
# aggregate_compliance_curve tests
# ---------------------------------------------------------------------------


class TestAggregateComplianceCurve:
    def test_basic_aggregation(self):
        """Aggregate 6 points at 3 r/w values with 2 seeds each."""
        points = [
            {"r_over_w": 0.5, "edge_compliance": 0.98, "rule_compliance": 0.95,
             "predictive_horizon": None, "seed": 42, "r": 32, "w": 64},
            {"r_over_w": 0.5, "edge_compliance": 0.97, "rule_compliance": 0.93,
             "predictive_horizon": None, "seed": 123, "r": 32, "w": 64},
            {"r_over_w": 1.0, "edge_compliance": 0.90, "rule_compliance": 0.75,
             "predictive_horizon": 5, "seed": 42, "r": 64, "w": 64},
            {"r_over_w": 1.0, "edge_compliance": 0.88, "rule_compliance": 0.70,
             "predictive_horizon": 4, "seed": 123, "r": 64, "w": 64},
            {"r_over_w": 1.5, "edge_compliance": 0.80, "rule_compliance": 0.40,
             "predictive_horizon": 2, "seed": 42, "r": 96, "w": 64},
            {"r_over_w": 1.5, "edge_compliance": 0.82, "rule_compliance": 0.42,
             "predictive_horizon": 3, "seed": 123, "r": 96, "w": 64},
        ]

        curve = aggregate_compliance_curve(points)

        assert curve["r_over_w_values"] == [0.5, 1.0, 1.5]
        assert curve["n_seeds"] == [2, 2, 2]
        assert len(curve["rule_compliance"]["mean"]) == 3
        assert len(curve["rule_compliance"]["std"]) == 3

        # Check specific means
        assert abs(curve["rule_compliance"]["mean"][0] - 0.94) < 0.01
        assert abs(curve["rule_compliance"]["mean"][1] - 0.725) < 0.01
        assert abs(curve["rule_compliance"]["mean"][2] - 0.41) < 0.01

    def test_single_seed_zero_std(self):
        """Single seed per r/w should give std=0."""
        points = [
            {"r_over_w": 0.5, "edge_compliance": 0.98, "rule_compliance": 0.95,
             "predictive_horizon": None, "seed": 42, "r": 32, "w": 64},
        ]
        curve = aggregate_compliance_curve(points)
        assert curve["n_seeds"] == [1]
        assert curve["rule_compliance"]["std"][0] == 0.0

    def test_empty_input(self):
        """Empty input should produce empty curve."""
        curve = aggregate_compliance_curve([])
        assert curve["r_over_w_values"] == []
        assert curve["n_seeds"] == []


# ---------------------------------------------------------------------------
# run_compliance_analysis tests
# ---------------------------------------------------------------------------


class TestRunComplianceAnalysis:
    def test_output_structure(self, tmp_path):
        """Verify output dict structure."""
        dirs = []
        for r, subdir in [(32, "exp1"), (64, "exp2")]:
            result = _make_result_json(r=r, w=64, edge_comp=0.9, rule_comp=0.8)
            d = _write_result_json(tmp_path, result, subdir)
            dirs.append(d)

        output = run_compliance_analysis(dirs)

        assert "config" in output
        assert output["config"]["n_result_dirs"] == 2
        assert output["config"]["n_valid_points"] == 2
        assert "curve" in output
        assert "raw_points" in output
        assert len(output["raw_points"]) == 2


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestComplianceVisualization:
    def test_plot_compliance_curve_returns_figure(self):
        """plot_compliance_curve should return a matplotlib Figure."""
        from src.visualization.compliance import plot_compliance_curve

        data = {
            "r_over_w_values": [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0],
            "edge_compliance": {
                "mean": [0.99, 0.98, 0.97, 0.95, 0.92, 0.88, 0.82, 0.70],
                "std": [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05],
            },
            "rule_compliance": {
                "mean": [0.95, 0.92, 0.88, 0.75, 0.55, 0.35, 0.20, 0.10],
                "std": [0.02, 0.02, 0.03, 0.05, 0.05, 0.04, 0.03, 0.02],
            },
            "predictive_horizon": {
                "mean": [None, None, 3, 5, 7, 4, 2, None],
                "std": [None, None, 1, 2, 2, 1, 1, None],
            },
            "n_seeds": [3, 3, 3, 3, 3, 3, 3, 3],
        }

        fig = plot_compliance_curve(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_compliance_curve_dual_axes(self):
        """Figure should have two y-axes when horizon data is present."""
        from src.visualization.compliance import plot_compliance_curve

        data = {
            "r_over_w_values": [0.5, 1.0, 1.5],
            "edge_compliance": {"mean": [0.99, 0.9, 0.8], "std": [0.01, 0.02, 0.03]},
            "rule_compliance": {"mean": [0.95, 0.7, 0.3], "std": [0.02, 0.05, 0.04]},
            "predictive_horizon": {"mean": [None, 5, 2], "std": [None, 2, 1]},
            "n_seeds": [2, 2, 2],
        }

        fig = plot_compliance_curve(data)
        # Should have created a twinx axis
        axes = fig.get_axes()
        assert len(axes) >= 2, "Should have dual axes"
        plt.close(fig)

    def test_plot_compliance_curve_without_horizon(self):
        """Figure should still generate without horizon data."""
        from src.visualization.compliance import plot_compliance_curve

        data = {
            "r_over_w_values": [0.5, 1.0, 1.5],
            "edge_compliance": {"mean": [0.99, 0.9, 0.8], "std": [0.01, 0.02, 0.03]},
            "rule_compliance": {"mean": [0.95, 0.7, 0.3], "std": [0.02, 0.05, 0.04]},
            "predictive_horizon": {"mean": [None, None, None], "std": [None, None, None]},
            "n_seeds": [2, 2, 2],
        }

        fig = plot_compliance_curve(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_compliance_scatter(self):
        """Scatter plot should return a matplotlib Figure."""
        from src.visualization.compliance import plot_compliance_scatter

        points = [
            {"r_over_w": 0.5, "rule_compliance": 0.95, "seed": 42},
            {"r_over_w": 0.5, "rule_compliance": 0.93, "seed": 123},
            {"r_over_w": 1.0, "rule_compliance": 0.70, "seed": 42},
            {"r_over_w": 1.0, "rule_compliance": 0.75, "seed": 123},
        ]

        fig = plot_compliance_scatter(points)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
