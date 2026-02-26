"""Tests for math verification PDF generation."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.reporting.math_pdf import (
    APPENDIX_FILES,
    MATH_SECTIONS,
    _create_latex_env,
    _read_source_file,
    generate_math_pdf,
)


class TestMathSections:
    """Tests for MATH_SECTIONS data completeness."""

    def test_math_sections_complete(self):
        """Verify MATH_SECTIONS has exactly 15 entries matching inventory."""
        assert len(MATH_SECTIONS) == 15
        for entry in MATH_SECTIONS:
            assert "file_path" in entry
            assert "title" in entry
            assert "summary" in entry
            assert isinstance(entry["summary"], str)
            assert len(entry["summary"]) > 0
            assert "latex_math" in entry
            assert isinstance(entry["latex_math"], str)
            assert len(entry["latex_math"]) > 0

    def test_appendix_files_complete(self):
        """Verify APPENDIX_FILES is non-empty with correct structure."""
        assert len(APPENDIX_FILES) > 0
        for entry in APPENDIX_FILES:
            assert "filename" in entry
            assert "description" in entry
            assert isinstance(entry["filename"], str)
            assert isinstance(entry["description"], str)
            assert len(entry["filename"]) > 0
            assert len(entry["description"]) > 0


class TestLatexEnvironment:
    """Tests for Jinja2 LaTeX environment configuration."""

    def test_latex_env_creation(self):
        """Verify custom LaTeX-safe delimiters are set."""
        env = _create_latex_env()
        assert "BLOCK" in env.block_start_string
        assert "VAR" in env.variable_start_string
        assert env.autoescape is False


class TestSourceFileReading:
    """Tests for source file reading utility."""

    def test_read_source_file_missing(self):
        """Verify _read_source_file with nonexistent path returns placeholder."""
        result = _read_source_file("nonexistent/path/file.py")
        assert "not found" in result.lower()

    def test_read_source_file_existing(self):
        """Verify _read_source_file reads a known existing file."""
        result = _read_source_file("src/evaluation/svd_metrics.py")
        assert "stable_rank" in result


class TestTexGeneration:
    """Tests for .tex file generation."""

    @pytest.fixture
    def tex_output(self, tmp_path):
        """Generate .tex file in temporary directory."""
        result_path = generate_math_pdf(tmp_path)
        content = result_path.read_text(encoding="utf-8")
        return result_path, content

    def test_tex_generation(self, tex_output):
        """Verify .tex file is created with all required content."""
        result_path, content = tex_output
        assert result_path.exists()
        assert result_path.suffix == ".tex"

        # Title
        assert "Mathematical Implementation Verification" in content
        # AI disclaimer footnote (MATH-02)
        assert "researcher sign-off" in content
        # Table of contents
        assert "\\tableofcontents" in content
        # Code blocks
        assert "\\begin{lstlisting}" in content
        # Math environments
        has_math = (
            "\\begin{equation}" in content
            or "\\begin{align}" in content
            or "\\[" in content
        )
        assert has_math, "No LaTeX math environment found in generated .tex"
        # Appendix
        assert "Non-Mathematical Source Files" in content

    def test_tex_contains_all_sections(self, tex_output):
        """Verify .tex contains a section for each MATH_SECTIONS entry title."""
        _, content = tex_output
        for section in MATH_SECTIONS:
            assert (
                f"\\section{{{section['title']}}}" in content
            ), f"Missing section for: {section['title']}"

    def test_generate_math_pdf_no_pdflatex(self, tmp_path):
        """Verify graceful handling when pdflatex is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError("pdflatex")):
            result_path = generate_math_pdf(tmp_path)
        assert result_path.exists()
        assert result_path.suffix == ".tex"
        # Should not crash, should return .tex path
        assert result_path.name == "math_verification.tex"
