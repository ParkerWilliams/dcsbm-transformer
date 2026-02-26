"""Tests for the reporting module.

Tests cover: base64 figure embedding, reproduction block builder,
single-experiment report generation with figures and missing sections.
"""

import base64
import json
from pathlib import Path

import pytest


# ── Embed Tests ──────────────────────────────────────────────────────


def test_embed_figure_png(tmp_path):
    """embed_figure returns valid data URI for a PNG file."""
    from src.reporting.embed import embed_figure

    # Create a minimal valid PNG (1x1 pixel, red)
    # PNG header + minimal IHDR + IDAT + IEND
    png_data = (
        b"\x89PNG\r\n\x1a\n"  # PNG signature
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde"
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    png_file = tmp_path / "test.png"
    png_file.write_bytes(png_data)

    result = embed_figure(png_file)

    assert result.startswith("data:image/png;base64,")
    # Verify the base64 portion is valid
    b64_part = result.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert decoded == png_data


def test_embed_figure_svg(tmp_path):
    """embed_figure returns valid data URI for an SVG file."""
    from src.reporting.embed import embed_figure

    svg_content = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><circle cx="5" cy="5" r="4"/></svg>'
    svg_file = tmp_path / "test.svg"
    svg_file.write_bytes(svg_content)

    result = embed_figure(svg_file)

    assert result.startswith("data:image/svg+xml;base64,")
    b64_part = result.split(",", 1)[1]
    decoded = base64.b64decode(b64_part)
    assert decoded == svg_content


def test_embed_figure_missing():
    """embed_figure returns empty string for nonexistent path."""
    from src.reporting.embed import embed_figure

    result = embed_figure(Path("/nonexistent/figure.png"))
    assert result == ""


def test_embed_figure_unsupported_extension(tmp_path):
    """embed_figure returns empty string for unsupported file types."""
    from src.reporting.embed import embed_figure

    txt_file = tmp_path / "test.txt"
    txt_file.write_text("not an image")

    result = embed_figure(txt_file)
    assert result == ""


# ── Reproduction Block Tests ─────────────────────────────────────────


def test_reproduction_block_clean():
    """Build reproduction block from result dict with clean hash."""
    from src.reporting.reproduction import build_reproduction_block

    result = {
        "metadata": {"code_hash": "abc1234"},
        "config": {
            "seed": 42,
            "graph": {"n": 100, "K": 4},
            "training": {"num_epochs": 50},
            "model": {"d_model": 64},
        },
    }

    block = build_reproduction_block(result)

    assert block["checkout_cmd"] == "git checkout abc1234"
    assert block["seed"] == 42
    assert block["is_dirty"] is False
    assert block["dirty_warning"] is None
    assert "--seed 42" in block["run_cmd"]
    assert "--n 100" in block["run_cmd"]
    assert "--K 4" in block["run_cmd"]


def test_reproduction_block_dirty():
    """Build reproduction block from result dict with dirty hash."""
    from src.reporting.reproduction import build_reproduction_block

    result = {
        "metadata": {"code_hash": "def5678-dirty"},
        "config": {
            "seed": 7,
            "graph": {},
            "training": {},
            "model": {},
        },
    }

    block = build_reproduction_block(result)

    assert block["checkout_cmd"] == "git checkout def5678"
    assert block["is_dirty"] is True
    assert block["dirty_warning"] is not None
    assert "uncommitted" in block["dirty_warning"].lower()


def test_reproduction_block_missing_metadata():
    """Reproduction block handles missing metadata gracefully."""
    from src.reporting.reproduction import build_reproduction_block

    result = {"config": {"seed": 1}}

    block = build_reproduction_block(result)

    assert block["checkout_cmd"] == "git checkout unknown"
    assert block["seed"] == 1
    assert block["is_dirty"] is False


# ── Single Report Generation Tests ───────────────────────────────────


def _create_result_dir(tmp_path, *, include_figures=True, include_metrics=True):
    """Create a minimal result directory for testing report generation.

    Args:
        tmp_path: pytest tmp_path fixture value.
        include_figures: Whether to create figure files in figures/ subdir.
        include_metrics: Whether to include predictive_horizon and statistical_controls.

    Returns:
        Path to the created result directory.
    """
    result_dir = tmp_path / "test-experiment"
    result_dir.mkdir()

    result = {
        "schema_version": "1.0",
        "experiment_id": "test-experiment-001",
        "timestamp": "2026-02-26T00:00:00+00:00",
        "description": "Unit test experiment",
        "tags": ["test"],
        "config": {
            "seed": 42,
            "graph": {"n": 100, "K": 4, "p_in": 0.25, "p_out": 0.03},
            "model": {"d_model": 64, "n_layers": 2, "n_heads": 1, "vocab_size": 104},
            "training": {
                "learning_rate": 3e-4,
                "num_epochs": 100,
                "batch_size": 64,
                "w": 16,
                "walk_length": 64,
                "corpus_size": 10000,
            },
        },
        "metrics": {
            "scalars": {
                "final_train_loss": 0.512,
                "edge_compliance": 0.97,
                "rule_compliance": 0.85,
            },
        },
        "sequences": [],
        "metadata": {"code_hash": "abc1234"},
    }

    if include_metrics:
        result["metrics"]["predictive_horizon"] = {
            "by_r_value": {
                "5": {
                    "by_metric": {
                        "qkt.layer_0.stable_rank": {
                            "auroc_by_lookback": [0.5, 0.65, 0.78, 0.82, 0.75],
                            "horizon": 4,
                        }
                    }
                }
            }
        }
        result["metrics"]["statistical_controls"] = {
            "headline_comparison": {
                "primary_metrics": {
                    "qkt.layer_0.stable_rank": {
                        "auroc": 0.82,
                        "p_value_corrected": 0.003,
                        "ci_lower": 0.74,
                        "ci_upper": 0.90,
                        "cohens_d": 1.2,
                    }
                }
            }
        }

    with open(result_dir / "result.json", "w") as f:
        json.dump(result, f)

    if include_figures:
        figures_dir = result_dir / "figures"
        figures_dir.mkdir()

        # Create minimal PNG files (1x1 pixel)
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde"
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        (figures_dir / "training_curves.png").write_bytes(png_data)
        (figures_dir / "confusion_matrix.png").write_bytes(png_data)

    return result_dir


def test_generate_single_report(tmp_path):
    """generate_single_report produces HTML with key content."""
    from src.reporting.single import generate_single_report

    result_dir = _create_result_dir(tmp_path, include_figures=True, include_metrics=True)
    output = generate_single_report(result_dir)

    assert output.exists()
    assert output.name == "report.html"

    html = output.read_text(encoding="utf-8")

    # Check key sections present
    assert "Configuration" in html
    assert "Reproduction" in html
    assert "test-experiment-001" in html

    # Check base64 figures embedded
    assert "data:image/png;base64," in html

    # Check config tables
    assert "d_model" in html
    assert "64" in html

    # Check scalar metrics
    assert "final_train_loss" in html

    # Check statistical tests
    assert "qkt.layer_0.stable_rank" in html

    # Check reproduction block
    assert "git checkout abc1234" in html
    assert "--seed 42" in html


def test_generate_single_report_missing_sections(tmp_path):
    """Report with minimal data shows 'Not available' placeholders."""
    from src.reporting.single import generate_single_report

    result_dir = _create_result_dir(tmp_path, include_figures=False, include_metrics=False)
    output = generate_single_report(result_dir)

    assert output.exists()
    html = output.read_text(encoding="utf-8")

    # Missing sections should show placeholder
    assert "Not available" in html

    # Config tables should still be present
    assert "Configuration" in html
    assert "d_model" in html

    # Reproduction block should still be present
    assert "git checkout abc1234" in html


def test_generate_single_report_custom_output_path(tmp_path):
    """generate_single_report writes to custom output path."""
    from src.reporting.single import generate_single_report

    result_dir = _create_result_dir(tmp_path, include_figures=False, include_metrics=False)
    custom_path = tmp_path / "custom" / "my_report.html"

    output = generate_single_report(result_dir, output_path=custom_path)

    assert output == custom_path
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
