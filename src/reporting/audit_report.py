"""Audit report HTML generator.

Produces a self-contained HTML file that renders the v1.2 Mathematical
Audit results.  The report includes a sticky sidebar for navigation,
a summary dashboard with per-category verdict counts, requirement
cards with KaTeX-rendered LaTeX formulas, and verdict badges.

All CSS and JavaScript (including KaTeX) are inlined so the report
opens in any browser without network access.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.reporting.audit_registry import (
    AUDIT_ENTRIES,
    CATEGORIES,
    entries_by_category,
)

log = logging.getLogger(__name__)

# Template directory relative to this file
_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Vendor directory for KaTeX assets
_VENDOR_DIR = Path(__file__).parent / "vendor"


def generate_audit_report(output_path: str | Path | None = None) -> Path:
    """Generate the self-contained v1.2 Mathematical Audit HTML report.

    Loads audit registry data, computes per-category summary statistics,
    inlines KaTeX CSS and JS, then renders the Jinja2 template into a
    single HTML file.

    Args:
        output_path: Where to write the HTML file.  Defaults to
            ``reports/audit_report.html`` relative to the project root.

    Returns:
        Path to the generated HTML report file.
    """
    # Read KaTeX vendor assets
    katex_js = (_VENDOR_DIR / "katex.min.js").read_text(encoding="utf-8")
    katex_css = (_VENDOR_DIR / "katex.min.css").read_text(encoding="utf-8")

    # Group entries by category
    by_cat = entries_by_category()

    # Compute summary statistics
    summary: dict[str, dict[str, int]] = {}
    total_correct = 0
    total_fixed = 0
    total_concern = 0

    for cat in CATEGORIES:
        cat_entries = by_cat[cat]
        correct = sum(1 for e in cat_entries if e["verdict"] == "correct")
        fixed = sum(1 for e in cat_entries if e["verdict"] == "fixed")
        concern = sum(1 for e in cat_entries if e["verdict"] == "concern")
        summary[cat] = {
            "correct": correct,
            "fixed": fixed,
            "concern": concern,
        }
        total_correct += correct
        total_fixed += fixed
        total_concern += concern

    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("audit_report.html")

    # Render template
    generation_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = template.render(
        katex_css=katex_css,
        katex_js=katex_js,
        categories=CATEGORIES,
        entries_by_category=by_cat,
        summary=summary,
        total_correct=total_correct,
        total_fixed=total_fixed,
        total_concern=total_concern,
        generation_date=generation_date,
    )

    # Determine output path
    if output_path is None:
        output_path = Path("reports") / "audit_report.html"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write report
    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    log.info("Audit report written to %s (%.1f KB)", output_path, size_kb)

    return output_path
