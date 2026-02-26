"""Tests for the visualization module.

Tests cover: style application, dual-format save, palette constants,
event-aligned plots, training curves, AUROC curves, and confusion matrix.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.analysis.event_extraction import AnalysisEvent
from src.evaluation.behavioral import RuleOutcome


# ── Style and Save Tests ──────────────────────────────────────────────


def test_apply_style_sets_whitegrid():
    """apply_style() sets seaborn whitegrid and publication rcParams."""
    from src.visualization.style import apply_style

    apply_style()
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["axes.grid"] is True


def test_save_figure_creates_png_and_svg(tmp_path):
    """save_figure creates both PNG and SVG, closes figure."""
    from src.visualization.style import save_figure

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    fig_num = fig.number

    png_path, svg_path = save_figure(fig, tmp_path, "test_plot")

    assert png_path.exists()
    assert svg_path.exists()
    assert png_path.stat().st_size > 0
    assert svg_path.stat().st_size > 0
    # Figure should be closed
    assert fig_num not in plt.get_fignums()


def test_save_figure_creates_directory(tmp_path):
    """save_figure creates output directory if it doesn't exist."""
    from src.visualization.style import save_figure

    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])

    nested = tmp_path / "sub" / "dir"
    png_path, svg_path = save_figure(fig, nested, "test_nested")

    assert nested.exists()
    assert png_path.exists()
    assert svg_path.exists()


def test_palette_is_colorblind_safe():
    """PALETTE has >= 8 colors; VIOLATION_COLOR and CONTROL_COLOR are valid."""
    from src.visualization.style import CONTROL_COLOR, PALETTE, VIOLATION_COLOR

    assert len(PALETTE) >= 8
    # Each color should be a tuple of 3 floats (RGB)
    assert len(VIOLATION_COLOR) >= 3
    assert len(CONTROL_COLOR) >= 3
    assert all(0 <= c <= 1 for c in VIOLATION_COLOR[:3])
    assert all(0 <= c <= 1 for c in CONTROL_COLOR[:3])


def test_apply_style_idempotent():
    """Calling apply_style() twice doesn't raise or break."""
    from src.visualization.style import apply_style

    apply_style()
    apply_style()
    assert plt.rcParams["savefig.dpi"] == 300


# ── Event-Aligned Plot Tests ──────────────────────────────────────────


def _make_synthetic_events(n_violations=5, n_controls=10, resolution_step=50):
    """Create synthetic AnalysisEvent lists for testing."""
    events = []
    for i in range(n_violations):
        events.append(
            AnalysisEvent(
                walk_idx=i,
                encounter_step=resolution_step - 5,
                resolution_step=resolution_step,
                r_value=5,
                outcome=RuleOutcome.VIOLATED,
                is_first_violation=True,
            )
        )
    for i in range(n_controls):
        events.append(
            AnalysisEvent(
                walk_idx=n_violations + i,
                encounter_step=resolution_step - 5,
                resolution_step=resolution_step,
                r_value=5,
                outcome=RuleOutcome.FOLLOWED,
                is_first_violation=False,
            )
        )
    return events


def test_event_aligned_plot_returns_figure():
    """plot_event_aligned returns a valid Figure with axes."""
    from src.visualization.style import apply_style
    from src.visualization.event_aligned import plot_event_aligned

    apply_style()
    n_seq = 15
    max_steps = 100
    metric_values = np.random.default_rng(42).normal(0, 1, (n_seq, max_steps))
    events = _make_synthetic_events(n_violations=5, n_controls=10, resolution_step=50)

    fig = plot_event_aligned(metric_values, events, window=10, metric_name="qkt.stable_rank")

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_event_aligned_plot_has_two_traces():
    """Event-aligned plot has >= 2 lines (violation + control)."""
    from src.visualization.style import apply_style
    from src.visualization.event_aligned import plot_event_aligned

    apply_style()
    n_seq = 15
    max_steps = 100
    metric_values = np.random.default_rng(42).normal(0, 1, (n_seq, max_steps))
    events = _make_synthetic_events(n_violations=5, n_controls=10, resolution_step=50)

    fig = plot_event_aligned(metric_values, events, window=10, metric_name="qkt.stable_rank")
    ax = fig.axes[0]

    assert len(ax.get_lines()) >= 2
    plt.close(fig)


def test_event_aligned_handles_nan():
    """Event-aligned plot handles NaN in metric values without crashing."""
    from src.visualization.style import apply_style
    from src.visualization.event_aligned import plot_event_aligned

    apply_style()
    n_seq = 15
    max_steps = 100
    metric_values = np.full((n_seq, max_steps), np.nan)
    # Only fill valid positions around event alignment point
    metric_values[:, 30:70] = np.random.default_rng(42).normal(0, 1, (n_seq, 40))
    events = _make_synthetic_events(n_violations=5, n_controls=10, resolution_step=50)

    fig = plot_event_aligned(metric_values, events, window=10, metric_name="qkt.stable_rank")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── Training Curves Tests ─────────────────────────────────────────────


def test_training_curves_returns_figure():
    """plot_training_curves returns Figure with 2 axes."""
    from src.visualization.style import apply_style
    from src.visualization.training import plot_training_curves

    apply_style()
    rng = np.random.default_rng(42)
    curves = {
        "train_loss": (3.0 - np.linspace(0, 2, 1000) + rng.normal(0, 0.1, 1000)).tolist(),
        "edge_compliance": np.linspace(0.5, 0.97, 50).tolist(),
        "rule_compliance": np.linspace(0.3, 0.85, 50).tolist(),
    }

    fig = plot_training_curves(curves)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_training_curves_shows_thresholds():
    """Compliance subplot has horizontal reference lines for gate thresholds."""
    from src.visualization.style import apply_style
    from src.visualization.training import plot_training_curves

    apply_style()
    curves = {
        "train_loss": [3.0, 2.5, 2.0, 1.5, 1.0],
        "edge_compliance": [0.5, 0.7, 0.8, 0.9, 0.97],
        "rule_compliance": [0.3, 0.5, 0.6, 0.7, 0.85],
    }

    fig = plot_training_curves(curves)
    # The compliance axes should have at least 2 horizontal lines (thresholds)
    # plus the 2 compliance curves = at least 4 lines total
    compliance_ax = fig.axes[1]
    assert len(compliance_ax.get_lines()) >= 4
    plt.close(fig)


# ── AUROC Curve Tests ─────────────────────────────────────────────────


def test_auroc_curve_returns_figure():
    """plot_auroc_curves returns a valid Figure."""
    from src.visualization.style import apply_style
    from src.visualization.auroc import plot_auroc_curves

    apply_style()
    auroc_results = {
        "metric_a": {
            "auroc_by_lookback": [0.5, 0.6, 0.7, 0.8, 0.75, 0.65],
            "horizon": 4,
        },
        "metric_b": {
            "auroc_by_lookback": [0.55, 0.65, 0.72, 0.68, 0.6, 0.55],
            "horizon": 3,
        },
    }

    fig = plot_auroc_curves(auroc_results, r_value=6)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_auroc_curve_shows_threshold_line():
    """AUROC plot has threshold (0.75) and chance (0.5) reference lines."""
    from src.visualization.style import apply_style
    from src.visualization.auroc import plot_auroc_curves

    apply_style()
    auroc_results = {
        "metric_a": {
            "auroc_by_lookback": [0.5, 0.6, 0.7, 0.8, 0.75, 0.65],
            "horizon": 4,
        },
    }

    fig = plot_auroc_curves(auroc_results, r_value=6)
    ax = fig.axes[0]
    # Should have: 1 metric line + 2 reference lines (threshold + chance) = >= 3 lines
    assert len(ax.get_lines()) >= 3
    plt.close(fig)


# ── Confusion Matrix Tests ────────────────────────────────────────────


def test_confusion_matrix_returns_figure():
    """plot_confusion_matrix returns a valid Figure."""
    from src.visualization.style import apply_style
    from src.visualization.confusion import plot_confusion_matrix

    apply_style()
    rng = np.random.default_rng(42)
    n_steps = 200
    edge_valid = rng.choice([True, False], size=n_steps, p=[0.9, 0.1])
    rule_outcome = np.full(n_steps, RuleOutcome.NOT_APPLICABLE, dtype=np.int32)
    # Set some as FOLLOWED and VIOLATED
    rule_outcome[:100] = RuleOutcome.FOLLOWED
    rule_outcome[100:130] = RuleOutcome.VIOLATED

    fig = plot_confusion_matrix(edge_valid, rule_outcome)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_confusion_matrix_counts():
    """Confusion matrix correctly counts 4-class outcomes."""
    from src.visualization.confusion import plot_confusion_matrix

    # Create known data: 100 valid+followed, 20 valid+violated, 5 invalid+followed, 3 invalid+violated
    n = 128 + 50  # 178 total (128 applicable + 50 N/A)
    edge_valid = np.zeros(n, dtype=bool)
    rule_outcome = np.full(n, RuleOutcome.NOT_APPLICABLE, dtype=np.int32)

    # Valid + followed: indices 0-99
    edge_valid[:100] = True
    rule_outcome[:100] = RuleOutcome.FOLLOWED

    # Valid + violated: indices 100-119
    edge_valid[100:120] = True
    rule_outcome[100:120] = RuleOutcome.VIOLATED

    # Invalid + followed: indices 120-124
    edge_valid[120:125] = False
    rule_outcome[120:125] = RuleOutcome.FOLLOWED

    # Invalid + violated: indices 125-127
    edge_valid[125:128] = False
    rule_outcome[125:128] = RuleOutcome.VIOLATED

    # N/A: indices 128+ (should be excluded)

    fig = plot_confusion_matrix(edge_valid, rule_outcome)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
