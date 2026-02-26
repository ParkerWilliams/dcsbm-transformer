"""Self-contained HTML report generation for DCSBM-transformer experiments.

Provides single-experiment report generation with base64-embedded figures,
structured configuration tables, and copy-pasteable reproduction blocks.
"""

from src.reporting.embed import embed_figure
from src.reporting.reproduction import build_reproduction_block
from src.reporting.single import generate_single_report

__all__ = ["generate_single_report", "embed_figure", "build_reproduction_block"]
