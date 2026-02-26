"""Single-experiment HTML report generator.

Produces a self-contained HTML file with base64-embedded figures,
structured configuration tables, scalar metrics, statistical test
results, and a copy-pasteable reproduction block.

Stub — full implementation in Task 2.
"""

from pathlib import Path


def generate_single_report(
    result_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Generate a single-experiment HTML report.

    Args:
        result_dir: Path to results/{experiment_id}/ directory.
        output_path: Where to write the HTML. Defaults to {result_dir}/report.html.

    Returns:
        Path to the generated HTML report file.
    """
    raise NotImplementedError("Full implementation in Task 2")
