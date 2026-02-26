"""Consistent visual style for all project figures.

Sets seaborn whitegrid theme with a colorblind-safe palette.
Provides save_figure() helper for dual PNG/SVG output (PLOT-07, PLOT-08).
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Colorblind-safe palette (seaborn 'colorblind' preset)
PALETTE = sns.color_palette("colorblind", n_colors=8)
VIOLATION_COLOR = PALETTE[3]  # red-ish
CONTROL_COLOR = PALETTE[0]    # blue-ish
BASELINE_COLOR = PALETTE[2]   # green-ish
THRESHOLD_COLOR = (0.5, 0.5, 0.5)  # gray


def apply_style() -> None:
    """Apply project-wide matplotlib/seaborn style (PLOT-07).

    Sets seaborn whitegrid, configures font sizes for publication,
    sets DPI for high-resolution output. Idempotent.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (8, 5),
        "svg.fonttype": "none",  # Embed text as SVG text elements
    })


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> tuple[Path, Path]:
    """Save figure as both PNG (300 dpi) and SVG (PLOT-08).

    Args:
        fig: Matplotlib figure to save.
        output_dir: Directory to write files into. Created if absent.
        name: Base filename (without extension).

    Returns:
        Tuple of (png_path, svg_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{name}.png"
    svg_path = output_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path
