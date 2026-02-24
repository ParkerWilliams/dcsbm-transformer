"""Tests for the result schema validation, writing, and experiment ID generation."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from src.config import ANCHOR_CONFIG
from src.results import validate_result, write_result, load_result, generate_experiment_id


class TestValidateResult:
    """validate_result accepts valid dicts and rejects invalid ones."""

    @pytest.fixture
    def valid_result(self):
        return {
            "schema_version": "1.0",
            "experiment_id": "n500_w64_r57_d128_L4_s42_20260224_120000",
            "timestamp": "2026-02-24T12:00:00+00:00",
            "description": "test experiment",
            "tags": ["test"],
            "config": {"graph": {"n": 500}},
            "metrics": {"scalars": {"loss": 0.5}},
        }

    def test_validate_result_valid(self, valid_result):
        errors = validate_result(valid_result)
        assert errors == []

    def test_validate_result_missing_fields(self):
        result = {"schema_version": "1.0"}
        errors = validate_result(result)
        assert len(errors) > 0
        assert any("Missing required" in e for e in errors)

    def test_validate_result_missing_scalars(self, valid_result):
        valid_result["metrics"] = {"curves": {}}
        errors = validate_result(valid_result)
        assert any("scalars" in e for e in errors)

    def test_validate_result_array_length_mismatch(self, valid_result):
        valid_result["sequences"] = [
            {
                "sequence_id": "seq-0",
                "tokens": [1, 2, 3, 4, 5],
                "token_logprobs": [0.1, 0.2, 0.3],  # wrong length
            }
        ]
        errors = validate_result(valid_result)
        assert any("token_logprobs" in e for e in errors)

    def test_validate_result_matching_arrays(self, valid_result):
        valid_result["sequences"] = [
            {
                "sequence_id": "seq-0",
                "tokens": [1, 2, 3],
                "token_logprobs": [0.1, 0.2, 0.3],
                "token_entropy": [0.5, 0.5, 0.5],
            }
        ]
        errors = validate_result(valid_result)
        assert errors == []

    def test_validate_result_bad_schema_version_type(self, valid_result):
        valid_result["schema_version"] = 1.0
        errors = validate_result(valid_result)
        assert any("schema_version" in e for e in errors)

    def test_validate_result_bad_tags_type(self, valid_result):
        valid_result["tags"] = "not-a-list"
        errors = validate_result(valid_result)
        assert any("tags" in e for e in errors)


class TestGenerateExperimentId:
    """Experiment ID follows the scannable slug format."""

    def test_generate_experiment_id_format(self):
        eid = generate_experiment_id(ANCHOR_CONFIG)
        pattern = r"^n500_w64_r57_d128_L4_s42_\d{8}_\d{6}$"
        assert re.match(pattern, eid), f"ID '{eid}' doesn't match expected pattern"

    def test_generate_experiment_id_different_configs(self):
        from dataclasses import replace
        cfg2 = replace(ANCHOR_CONFIG, seed=99)
        eid1 = generate_experiment_id(ANCHOR_CONFIG)
        eid2 = generate_experiment_id(cfg2)
        # Seeds differ, so IDs should differ (in the seed part)
        assert "_s42_" in eid1
        assert "_s99_" in eid2


class TestWriteResult:
    """write_result creates correct directory structure and files."""

    def test_write_result_creates_files(self, tmp_path):
        metrics = {"scalars": {"loss": 0.42}}
        eid = write_result(ANCHOR_CONFIG, metrics, results_dir=str(tmp_path))
        result_dir = tmp_path / eid
        assert result_dir.is_dir()
        result_json = result_dir / "result.json"
        assert result_json.exists()
        data = json.loads(result_json.read_text())
        assert data["schema_version"] == "1.0"
        assert data["experiment_id"] == eid
        assert data["metrics"]["scalars"]["loss"] == 0.42

    def test_write_result_validates_before_write(self, tmp_path):
        bad_metrics = {"no_scalars_here": True}
        with pytest.raises(ValueError, match="validation failed"):
            write_result(ANCHOR_CONFIG, bad_metrics, results_dir=str(tmp_path))
        # Verify no result.json was created (validation fires before write)
        dirs = list(tmp_path.iterdir())
        for d in dirs:
            assert not (d / "result.json").exists()

    def test_write_result_with_token_metrics(self, tmp_path):
        metrics = {"scalars": {"loss": 0.42}}
        token_metrics = {
            "seq-0": {
                "sigma_ratio": np.array([1.0, 2.0, 3.0]),
                "entropy": np.array([0.5, 0.6, 0.7]),
            }
        }
        eid = write_result(
            ANCHOR_CONFIG,
            metrics,
            token_metrics=token_metrics,
            results_dir=str(tmp_path),
        )
        result_dir = tmp_path / eid
        npz_path = result_dir / "token_metrics.npz"
        assert npz_path.exists()
        loaded = np.load(str(npz_path))
        assert "seq-0/sigma_ratio" in loaded
        assert "seq-0/entropy" in loaded
        np.testing.assert_array_equal(loaded["seq-0/sigma_ratio"], [1.0, 2.0, 3.0])

    def test_write_result_metadata(self, tmp_path):
        metrics = {"scalars": {"loss": 0.42}}
        eid = write_result(ANCHOR_CONFIG, metrics, results_dir=str(tmp_path))
        data = json.loads((tmp_path / eid / "result.json").read_text())
        assert "metadata" in data
        assert "config_hash" in data["metadata"]
        assert "graph_config_hash" in data["metadata"]
        assert "code_hash" in data["metadata"]


class TestLoadResult:
    """load_result reads and validates result.json files."""

    def test_load_result(self, tmp_path):
        metrics = {"scalars": {"loss": 0.42}}
        eid = write_result(ANCHOR_CONFIG, metrics, results_dir=str(tmp_path))
        result_path = tmp_path / eid / "result.json"
        data = load_result(result_path)
        assert data["experiment_id"] == eid
        assert data["metrics"]["scalars"]["loss"] == 0.42

    def test_load_result_invalid_file(self, tmp_path):
        bad_file = tmp_path / "bad_result.json"
        bad_file.write_text('{"schema_version": "1.0"}')
        with pytest.raises(ValueError, match="validation failed"):
            load_result(bad_file)
