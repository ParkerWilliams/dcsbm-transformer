"""Publication-quality static figure generation for DCSBM-transformer experiments.

Provides render_all() to generate all figures for a single experiment,
with individual plot modules for each visualization type.
"""

from src.visualization.render import load_result_data, render_all
from src.visualization.style import apply_style, save_figure

__all__ = [
    "render_all",
    "load_result_data",
    "apply_style",
    "save_figure",
]
