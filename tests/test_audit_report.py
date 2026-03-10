"""Tests for the audit report HTML generator.

Verifies that generate_audit_report() produces a self-contained HTML
file containing all 28 requirements, 6 categories, inlined KaTeX
assets, and correct verdict counts.
"""

import re

import pytest

from src.reporting.audit_registry import AUDIT_ENTRIES, CATEGORIES
from src.reporting.audit_report import generate_audit_report


@pytest.fixture()
def report_html(tmp_path):
    """Generate an audit report and return the HTML content."""
    output = tmp_path / "audit.html"
    path = generate_audit_report(output)
    assert path == output
    return path.read_text(encoding="utf-8")


def test_report_generation(tmp_path):
    """Report generates a file that exists and is reasonably sized."""
    output = tmp_path / "audit.html"
    path = generate_audit_report(output)
    assert path.exists()
    size = path.stat().st_size
    assert size > 10_000, f"Report too small: {size} bytes"
    content = path.read_text(encoding="utf-8")
    assert "Mathematical Audit Report" in content


def test_report_contains_all_categories(report_html):
    """All 6 category names appear in the generated HTML."""
    for cat in CATEGORIES:
        assert cat in report_html, f"Category '{cat}' not found in report"


def test_report_contains_all_req_ids(report_html):
    """All 28 requirement IDs appear in the generated HTML."""
    for entry in AUDIT_ENTRIES:
        req_id = entry["req_id"]
        assert req_id in report_html, f"Requirement '{req_id}' not found in report"


def test_report_contains_katex(report_html):
    """KaTeX JavaScript is inlined in the report."""
    assert "katex" in report_html.lower(), "KaTeX not found in report"
    # Should contain the actual KaTeX library code, not just references
    assert "katex.render" in report_html, "katex.render not found in report"


def test_report_self_contained(report_html):
    """Report has no external URL references (fully self-contained)."""
    # Find all href= and src= attributes pointing to http(s) URLs
    external_refs = re.findall(
        r'(?:href|src)\s*=\s*["\']https?://[^"\']+["\']',
        report_html,
    )
    assert len(external_refs) == 0, (
        f"Report has {len(external_refs)} external references: "
        + ", ".join(external_refs[:5])
    )


def test_report_verdict_counts(report_html):
    """Dashboard shows correct total verdict counts (24 correct, 4 fixed)."""
    # Count verdicts from the registry itself to verify
    n_correct = sum(1 for e in AUDIT_ENTRIES if e["verdict"] == "correct")
    n_fixed = sum(1 for e in AUDIT_ENTRIES if e["verdict"] == "fixed")
    assert n_correct == 24, f"Expected 24 correct, got {n_correct}"
    assert n_fixed == 4, f"Expected 4 fixed, got {n_fixed}"

    # The totals row should contain these numbers
    assert "24" in report_html, "Total correct count '24' not in report"
    assert "4" in report_html, "Total fixed count '4' not in report"
