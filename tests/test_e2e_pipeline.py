"""Integration tests for the end-to-end experiment pipeline.

Tests the full pipeline from config loading through report generation
using tiny configurations for fast execution.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.config import ExperimentConfig, GraphConfig, ModelConfig, TrainingConfig
from src.config.serialization import config_to_json
from src.results import generate_experiment_id, validate_result


# Tiny config for fast E2E testing.
# n=50 with K=2 gives blocks of 25, enough for connectivity.
# p_in=0.4 ensures strong in-block edges; p_out=0.08 ensures min degree >= 3.
TINY_CONFIG = ExperimentConfig(
    graph=GraphConfig(
        n=50,
        K=2,
        p_in=0.4,
        p_out=0.08,
        n_jumpers_per_block=1,
    ),
    model=ModelConfig(
        d_model=16,
        n_layers=1,
        n_heads=1,
        dropout=0.0,
    ),
    training=TrainingConfig(
        w=8,
        walk_length=16,
        corpus_size=5000,
        r=5,
        learning_rate=3e-4,
        batch_size=32,
        max_steps=50000,
        eval_interval=1000,
        checkpoint_interval=5000,
    ),
    seed=42,
    description="E2E pipeline test",
    tags=("test", "e2e"),
)


def _write_config(tmp_path: Path) -> Path:
    """Write tiny config to a temporary file."""
    config_path = tmp_path / "config.json"
    config_path.write_text(config_to_json(TINY_CONFIG))
    return config_path


class TestDryRun:
    """Tests for --dry-run mode."""

    def test_dry_run_exits_cleanly(self, tmp_path: Path) -> None:
        """Dry run should load config and print plan without executing."""
        config_path = _write_config(tmp_path)
        result = subprocess.run(
            [sys.executable, "run_experiment.py", "--config", str(config_path), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Pipeline plan" in result.stdout
        assert "dry-run" in result.stdout.lower()

    def test_dry_run_shows_stages(self, tmp_path: Path) -> None:
        """Dry run should list all pipeline stages."""
        config_path = _write_config(tmp_path)
        result = subprocess.run(
            [sys.executable, "run_experiment.py", "--config", str(config_path), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        assert "Set seed" in output
        assert "Graph generation" in output
        assert "Walk generation" in output
        assert "Model creation" in output
        assert "Training" in output
        assert "Evaluation" in output
        assert "Analysis" in output
        assert "Visualization" in output
        assert "Reporting" in output

    def test_dry_run_shows_output_paths(self, tmp_path: Path) -> None:
        """Dry run should show expected output paths."""
        config_path = _write_config(tmp_path)
        result = subprocess.run(
            [sys.executable, "run_experiment.py", "--config", str(config_path), "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        # Experiment ID contains a timestamp so we check the slug prefix
        assert "n50_w8_r5_d16_L1_s42_" in output
        assert "result.json" in output
        assert "token_metrics.npz" in output
        assert "report.html" in output


@pytest.fixture(scope="module")
def pipeline_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the full pipeline once for all tests in this module."""
    tmp_path = tmp_path_factory.mktemp("e2e")
    config_path = _write_config(tmp_path)
    results_dir = str(tmp_path / "results")

    from run_experiment import run_pipeline

    output_dir = run_pipeline(config_path, results_dir=results_dir)
    return output_dir


class TestFullPipeline:
    """Tests for the full E2E pipeline execution.

    These tests actually run the pipeline with tiny configs. They may take
    up to 60 seconds due to training and evaluation steps.
    """

    def test_pipeline_completes(self, pipeline_output: Path) -> None:
        """Pipeline should complete without error."""
        assert pipeline_output.exists()

    def test_result_json_exists(self, pipeline_output: Path) -> None:
        """result.json should exist in output directory."""
        result_path = pipeline_output / "result.json"
        assert result_path.exists()

    def test_result_json_validates(self, pipeline_output: Path) -> None:
        """result.json should pass schema validation."""
        result_path = pipeline_output / "result.json"
        with open(result_path) as f:
            result = json.load(f)
        errors = validate_result(result)
        assert errors == [], f"Validation errors: {errors}"

    def test_predictive_horizon_in_result(self, pipeline_output: Path) -> None:
        """result.json should contain predictive_horizon block."""
        result_path = pipeline_output / "result.json"
        with open(result_path) as f:
            result = json.load(f)
        assert "predictive_horizon" in result["metrics"]
        ph = result["metrics"]["predictive_horizon"]
        assert "config" in ph
        assert "by_r_value" in ph

    def test_statistical_controls_in_result(self, pipeline_output: Path) -> None:
        """result.json should contain statistical_controls block."""
        result_path = pipeline_output / "result.json"
        with open(result_path) as f:
            result = json.load(f)
        assert "statistical_controls" in result["metrics"]

    def test_seed_in_metadata(self, pipeline_output: Path) -> None:
        """result.json metadata should contain the seed value."""
        result_path = pipeline_output / "result.json"
        with open(result_path) as f:
            result = json.load(f)
        assert "seed" in result["metadata"]
        assert result["metadata"]["seed"] == TINY_CONFIG.seed

    def test_token_metrics_npz_exists(self, pipeline_output: Path) -> None:
        """token_metrics.npz should exist with SVD metric keys."""
        npz_path = pipeline_output / "token_metrics.npz"
        assert npz_path.exists()
        data = np.load(str(npz_path), allow_pickle=False)
        keys = list(data.keys())
        # Should have dotted SVD metric keys
        svd_keys = [k for k in keys if "." in k and k.split(".")[0] in ("qkt", "avwo", "wvwo")]
        assert len(svd_keys) > 0, f"No SVD keys found. Keys: {keys}"

    def test_npz_key_format_consistency(self, pipeline_output: Path) -> None:
        """NPZ keys should use dotted format, not slash format."""
        npz_path = pipeline_output / "token_metrics.npz"
        data = np.load(str(npz_path), allow_pickle=False)
        for key in data.keys():
            assert "/" not in key, f"Slash in NPZ key: {key}"

    def test_figures_directory_exists(self, pipeline_output: Path) -> None:
        """Figures directory should exist with at least one figure."""
        figures_dir = pipeline_output / "figures"
        assert figures_dir.exists()
        pngs = list(figures_dir.glob("*.png"))
        assert len(pngs) > 0, "No PNG figures generated"

    def test_report_html_exists(self, pipeline_output: Path) -> None:
        """HTML report should exist in output directory."""
        report_path = pipeline_output / "report.html"
        assert report_path.exists()
        content = report_path.read_text()
        assert len(content) > 100, "Report seems too short"

    def test_config_copy_exists(self, pipeline_output: Path) -> None:
        """Config copy should exist in output directory."""
        config_copy = pipeline_output / "config.json"
        assert config_copy.exists()

    def test_dual_key_emission_single_head(self, pipeline_output: Path) -> None:
        """Single-head runs should emit both legacy and per-head NPZ keys."""
        npz_path = pipeline_output / "token_metrics.npz"
        data = np.load(str(npz_path), allow_pickle=False)
        keys = set(data.keys())
        # Legacy format: target.layer_N.metric_name
        legacy_keys = [k for k in keys if k.count(".") == 2 and k.startswith("qkt.layer_")]
        # Per-head format: target.layer_N.head_H.metric_name
        perhead_keys = [k for k in keys if k.count(".") == 3 and "head_" in k and k.startswith("qkt.layer_")]
        # Both should be present for single-head
        assert len(legacy_keys) > 0, f"No legacy keys found. Keys sample: {list(keys)[:10]}"
        assert len(perhead_keys) > 0, f"No per-head keys found. Keys sample: {list(keys)[:10]}"


class TestVisualizationInit:
    """Tests for the visualization __init__.py exports."""

    def test_render_all_importable(self) -> None:
        """render_all should be importable from src.visualization."""
        from src.visualization import render_all
        assert callable(render_all)

    def test_load_result_data_importable(self) -> None:
        """load_result_data should be importable from src.visualization."""
        from src.visualization import load_result_data
        assert callable(load_result_data)

    def test_all_exports(self) -> None:
        """__all__ should be defined with expected exports."""
        import src.visualization
        assert hasattr(src.visualization, "__all__")
        assert "render_all" in src.visualization.__all__
        assert "load_result_data" in src.visualization.__all__
        assert "apply_style" in src.visualization.__all__
        assert "save_figure" in src.visualization.__all__
