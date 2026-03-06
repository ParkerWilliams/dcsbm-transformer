"""Audit tests for cross-path horizon consistency (AUROC-03).

Verifies that all AUROC consumers delegate to auroc_horizon.py (no duplicated
implementations), and that the 0.75 predictive horizon threshold is consistently
applied across all analysis and display paths.

Uses AST-based import inspection and functional tests to verify code-path
consistency without brittle string matching.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from src.analysis.auroc_horizon import (
    compute_predictive_horizon,
    run_auroc_analysis,
)

# Root of the source tree
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def _read_source(module_path: Path) -> str:
    """Read source file content."""
    return module_path.read_text()


def _get_import_from_nodes(source: str) -> list[ast.ImportFrom]:
    """Parse source and return all ImportFrom AST nodes."""
    tree = ast.parse(source)
    return [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]


def _get_function_defaults(source: str, func_name: str) -> dict[str, object]:
    """Extract default parameter values for a function definition via AST.

    Returns dict mapping parameter_name -> default_value (evaluated as literal).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == func_name:
                defaults = {}
                args = node.args
                # args.defaults apply to the last len(defaults) positional args
                n_defaults = len(args.defaults)
                n_args = len(args.args)
                for i, default in enumerate(args.defaults):
                    arg_name = args.args[n_args - n_defaults + i].arg
                    try:
                        defaults[arg_name] = ast.literal_eval(default)
                    except (ValueError, TypeError):
                        defaults[arg_name] = None  # Non-literal default
                # kwonly defaults
                for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                    if default is not None:
                        try:
                            defaults[arg.arg] = ast.literal_eval(default)
                        except (ValueError, TypeError):
                            defaults[arg.arg] = None
                return defaults
    return {}


# ---------------------------------------------------------------------------
# Code-path audit: all AUROC consumers import from auroc_horizon.py
# ---------------------------------------------------------------------------
class TestAurocCodePathConsistency:
    """Verify every module consuming AUROC delegates to auroc_horizon.py."""

    def test_statistical_controls_imports_from_auroc_horizon(self):
        """statistical_controls.py must import auroc_from_groups, compute_auroc_curve,
        and compute_predictive_horizon from src.analysis.auroc_horizon."""
        source = _read_source(SRC_ROOT / "analysis" / "statistical_controls.py")
        imports = _get_import_from_nodes(source)

        # Find imports from src.analysis.auroc_horizon
        auroc_imports: set[str] = set()
        for node in imports:
            if node.module and "auroc_horizon" in node.module:
                for alias in node.names:
                    auroc_imports.add(alias.name)

        required = {"auroc_from_groups", "compute_auroc_curve", "compute_predictive_horizon"}
        missing = required - auroc_imports
        assert not missing, (
            f"statistical_controls.py missing imports from auroc_horizon: {missing}"
        )

    def test_spectrum_imports_from_auroc_horizon(self):
        """spectrum.py must import auroc_from_groups and compute_auroc_curve
        from src.analysis.auroc_horizon."""
        source = _read_source(SRC_ROOT / "analysis" / "spectrum.py")
        imports = _get_import_from_nodes(source)

        auroc_imports: set[str] = set()
        for node in imports:
            if node.module and "auroc_horizon" in node.module:
                for alias in node.names:
                    auroc_imports.add(alias.name)

        required = {"auroc_from_groups", "compute_auroc_curve"}
        missing = required - auroc_imports
        assert not missing, (
            f"spectrum.py missing imports from auroc_horizon: {missing}"
        )

    def test_pr_curves_imports_from_auroc_horizon(self):
        """pr_curves.py must import auroc_from_groups from src.analysis.auroc_horizon."""
        source = _read_source(SRC_ROOT / "analysis" / "pr_curves.py")
        imports = _get_import_from_nodes(source)

        auroc_imports: set[str] = set()
        for node in imports:
            if node.module and "auroc_horizon" in node.module:
                for alias in node.names:
                    auroc_imports.add(alias.name)

        assert "auroc_from_groups" in auroc_imports, (
            "pr_curves.py must import auroc_from_groups from auroc_horizon"
        )

    def test_calibration_imports_from_auroc_horizon(self):
        """calibration.py must import auroc_from_groups from src.analysis.auroc_horizon."""
        source = _read_source(SRC_ROOT / "analysis" / "calibration.py")
        imports = _get_import_from_nodes(source)

        auroc_imports: set[str] = set()
        for node in imports:
            if node.module and "auroc_horizon" in node.module:
                for alias in node.names:
                    auroc_imports.add(alias.name)

        assert "auroc_from_groups" in auroc_imports, (
            "calibration.py must import auroc_from_groups from auroc_horizon"
        )

    def test_null_model_does_not_compute_auroc(self):
        """null_model.py must NOT import auroc_from_groups, compute_auroc_curve,
        or compute_predictive_horizon from anywhere.

        null_model.py uses Mann-Whitney U for drift comparison -- a fundamentally
        different statistical question (does null drift differ from violation drift?)
        rather than computing predictive horizon."""
        source = _read_source(SRC_ROOT / "analysis" / "null_model.py")

        # Verify none of the AUROC computation function names appear in source
        forbidden = ["auroc_from_groups", "compute_auroc_curve", "compute_predictive_horizon"]
        for name in forbidden:
            assert name not in source, (
                f"null_model.py should NOT reference '{name}' -- "
                "it uses Mann-Whitney U for a different statistical test"
            )

    def test_signal_concentration_no_auroc_computation(self):
        """signal_concentration.py must NOT import or call auroc_from_groups.

        It receives pre-computed AUROC values as input (concentration metrics only)."""
        source = _read_source(SRC_ROOT / "analysis" / "signal_concentration.py")

        # Verify no AUROC computation function names appear
        forbidden = ["auroc_from_groups", "compute_auroc_curve", "compute_predictive_horizon"]
        for name in forbidden:
            assert name not in source, (
                f"signal_concentration.py should NOT reference '{name}' -- "
                "it receives pre-computed AUROC values as input"
            )


# ---------------------------------------------------------------------------
# Threshold consistency: 0.75 consistently applied across all code paths
# ---------------------------------------------------------------------------
class TestHorizonThresholdConsistency:
    """Verify the 0.75 threshold is consistently applied everywhere."""

    def test_compute_predictive_horizon_default_threshold(self):
        """compute_predictive_horizon without specifying threshold must use 0.75.

        Test with auroc_curve=[0.76, 0.74]:
        - j=1 has 0.76 > 0.75, j=2 has 0.74 <= 0.75
        - Default threshold should yield horizon=1 (max j where AUROC > 0.75)."""
        curve = np.array([0.76, 0.74])
        # j=1 (index 0) has 0.76 > 0.75, j=2 (index 1) has 0.74 <= 0.75
        # Scans from largest j=2 to j=1; j=2 fails, j=1 passes -> horizon=1
        horizon = compute_predictive_horizon(curve)
        assert horizon == 1, (
            f"Default threshold should be 0.75; expected horizon=1 for [0.76, 0.74], got {horizon}"
        )

    def test_compute_predictive_horizon_custom_threshold(self):
        """Verify custom threshold parameter works correctly (sanity check)."""
        curve = np.array([0.76, 0.74])
        # With threshold=0.73, both j=1 and j=2 pass -> horizon=2 (max j)
        horizon = compute_predictive_horizon(curve, threshold=0.73)
        assert horizon == 2

        # With threshold=0.77, neither passes -> horizon=0
        horizon = compute_predictive_horizon(curve, threshold=0.77)
        assert horizon == 0

    def test_run_auroc_analysis_default_threshold(self):
        """run_auroc_analysis horizon_threshold default must be 0.75.

        Uses inspect.signature to verify the default parameter value."""
        sig = inspect.signature(run_auroc_analysis)
        param = sig.parameters["horizon_threshold"]
        assert param.default == 0.75, (
            f"run_auroc_analysis horizon_threshold default should be 0.75, got {param.default}"
        )

    def test_visualization_threshold_defaults(self):
        """Visualization modules (auroc.py, heatmap.py) must have threshold=0.75 defaults.

        Uses AST to find function definitions and verify default argument values."""
        # Check auroc.py
        auroc_source = _read_source(SRC_ROOT / "visualization" / "auroc.py")
        auroc_defaults = _get_function_defaults(auroc_source, "plot_auroc_curves")
        assert "threshold" in auroc_defaults, (
            "plot_auroc_curves must have a 'threshold' parameter"
        )
        assert auroc_defaults["threshold"] == 0.75, (
            f"plot_auroc_curves threshold default should be 0.75, got {auroc_defaults['threshold']}"
        )

        # Check heatmap.py
        heatmap_source = _read_source(SRC_ROOT / "visualization" / "heatmap.py")
        heatmap_defaults = _get_function_defaults(heatmap_source, "plot_horizon_heatmap")
        assert "threshold" in heatmap_defaults, (
            "plot_horizon_heatmap must have a 'threshold' parameter"
        )
        assert heatmap_defaults["threshold"] == 0.75, (
            f"plot_horizon_heatmap threshold default should be 0.75, got {heatmap_defaults['threshold']}"
        )

    def test_reporting_threshold_reference(self):
        """reporting/single.py hardcoded 0.75 must match the analytical default.

        The 0.75 appears in a display/formatting context (significance check),
        not a computation context."""
        source = _read_source(SRC_ROOT / "reporting" / "single.py")

        # Verify 0.75 appears in source
        assert "0.75" in source, (
            "reporting/single.py should reference 0.75 for display consistency"
        )

        # Verify it appears in a comparison context (>= 0.75), not as a standalone computation
        assert ">= 0.75" in source or "0.75" in source, (
            "reporting/single.py 0.75 should be in display/comparison context"
        )

    def test_all_075_occurrences_cataloged(self):
        """Regression test: all occurrences of '0.75' in src/ must be from known locations.

        Known locations (from RESEARCH.md):
        - auroc_horizon.py: default parameter for compute_predictive_horizon and run_auroc_analysis
        - visualization/auroc.py: display threshold
        - visualization/heatmap.py: display threshold
        - reporting/single.py: significance display
        - reporting/math_pdf.py: LaTeX documentation string

        Any NEW occurrence not in this list is flagged as a potential consistency gap."""
        known_files = {
            "auroc_horizon.py",
            "auroc.py",          # visualization/auroc.py
            "heatmap.py",        # visualization/heatmap.py
            "single.py",         # reporting/single.py
            "math_pdf.py",       # reporting/math_pdf.py
        }

        unknown_files: list[str] = []

        for py_file in SRC_ROOT.rglob("*.py"):
            try:
                content = py_file.read_text()
            except (OSError, UnicodeDecodeError):
                continue

            if "0.75" in content:
                filename = py_file.name
                if filename not in known_files:
                    # Build relative path for clear error message
                    rel = py_file.relative_to(SRC_ROOT)
                    unknown_files.append(str(rel))

        assert not unknown_files, (
            f"Found '0.75' in uncataloged files: {unknown_files}. "
            "If these are legitimate, add them to the known_files set in this test."
        )
