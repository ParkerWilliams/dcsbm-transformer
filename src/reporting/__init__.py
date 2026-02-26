"""Self-contained HTML report generation and math PDF for DCSBM-transformer experiments.

Provides single-experiment report generation with base64-embedded figures,
structured configuration tables, copy-pasteable reproduction blocks,
multi-experiment comparison reports, and a math verification PDF generator
for peer review.
"""

from src.reporting.comparison import generate_comparison_report
from src.reporting.embed import embed_figure
from src.reporting.math_pdf import generate_math_pdf
from src.reporting.reproduction import build_reproduction_block
from src.reporting.single import generate_single_report

__all__ = [
    "generate_single_report",
    "generate_comparison_report",
    "generate_math_pdf",
    "embed_figure",
    "build_reproduction_block",
]
