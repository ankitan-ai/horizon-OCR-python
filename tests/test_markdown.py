"""Tests for the Markdown report generator."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from docvision.io.markdown import generate_markdown, save_markdown


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_result() -> dict:
    """Minimal but realistic DocVision JSON result."""
    return {
        "id": "abc123",
        "metadata": {
            "filename": "test_invoice.png",
            "file_type": "image",
            "file_size_bytes": 12345,
            "processed_at": "2026-02-10T12:00:00",
            "processing_time_seconds": 3.45,
            "docvision_version": "0.1.0",
        },
        "page_count": 1,
        "pages": [
            {
                "number": 1,
                "metadata": {
                    "width": 800,
                    "height": 600,
                    "dpi": 350,
                    "content_type": "mixed",
                    "readability": "good",
                    "readability_issues": [],
                },
                "layout_regions": [
                    {
                        "id": "lr1",
                        "type": "text",
                        "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 50},
                        "confidence": 0.95,
                        "text_lines": [],
                        "content_type": "printed",
                    },
                    {
                        "id": "lr2",
                        "type": "table",
                        "bbox": {"x1": 10, "y1": 100, "x2": 400, "y2": 300},
                        "confidence": 0.82,
                        "text_lines": [],
                        "content_type": "unknown",
                    },
                ],
                "text_lines": [
                    {
                        "id": "tl1",
                        "text": "INVOICE #12345",
                        "words": [],
                        "polygon": None,
                        "bbox": {"x1": 10, "y1": 10, "x2": 200, "y2": 30},
                        "confidence": 0.99,
                        "source": "trocr",
                        "content_type": "printed",
                    },
                    {
                        "id": "tl2",
                        "text": "TOTAL: $500.00",
                        "words": [],
                        "polygon": None,
                        "bbox": {"x1": 10, "y1": 40, "x2": 200, "y2": 60},
                        "confidence": 0.95,
                        "source": "trocr",
                        "content_type": "printed",
                    },
                ],
                "tables": [
                    {
                        "id": "t1",
                        "page": 1,
                        "bbox": {"x1": 10, "y1": 100, "x2": 400, "y2": 300},
                        "rows": 3,
                        "cols": 3,
                        "cells": [
                            {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "Item", "confidence": 0.90, "source": "trocr", "is_header": True},
                            {"row": 0, "col": 1, "row_span": 1, "col_span": 1, "text": "Qty", "confidence": 0.92, "source": "trocr", "is_header": True},
                            {"row": 0, "col": 2, "row_span": 1, "col_span": 1, "text": "Price", "confidence": 0.91, "source": "trocr", "is_header": True},
                            {"row": 1, "col": 0, "row_span": 1, "col_span": 1, "text": "Widget A", "confidence": 0.88, "source": "trocr", "is_header": False},
                            {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "2", "confidence": 0.85, "source": "trocr", "is_header": False},
                            {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "$250.00", "confidence": 0.87, "source": "trocr", "is_header": False},
                            {"row": 2, "col": 0, "row_span": 1, "col_span": 1, "text": "Widget B", "confidence": 0.45, "source": "trocr", "is_header": False},
                            {"row": 2, "col": 1, "row_span": 1, "col_span": 1, "text": "1", "confidence": 0.40, "source": "trocr", "is_header": False},
                            {"row": 2, "col": 2, "row_span": 1, "col_span": 1, "text": "$250.00", "confidence": 0.90, "source": "trocr", "is_header": False},
                        ],
                        "confidence": 0.82,
                        "has_borders": True,
                    }
                ],
                "raw_text": "INVOICE #12345\nTOTAL: $500.00",
            }
        ],
        "tables": [
            {
                "id": "t1",
                "page": 1,
                "bbox": {"x1": 10, "y1": 100, "x2": 400, "y2": 300},
                "rows": 3,
                "cols": 3,
                "cells": [
                    {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "Item", "confidence": 0.90, "source": "trocr", "is_header": True},
                    {"row": 0, "col": 1, "row_span": 1, "col_span": 1, "text": "Qty", "confidence": 0.92, "source": "trocr", "is_header": True},
                    {"row": 0, "col": 2, "row_span": 1, "col_span": 1, "text": "Price", "confidence": 0.91, "source": "trocr", "is_header": True},
                    {"row": 1, "col": 0, "row_span": 1, "col_span": 1, "text": "Widget A", "confidence": 0.88, "source": "trocr", "is_header": False},
                    {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "2", "confidence": 0.85, "source": "trocr", "is_header": False},
                    {"row": 1, "col": 2, "row_span": 1, "col_span": 1, "text": "$250.00", "confidence": 0.87, "source": "trocr", "is_header": False},
                    {"row": 2, "col": 0, "row_span": 1, "col_span": 1, "text": "Widget B", "confidence": 0.45, "source": "trocr", "is_header": False},
                    {"row": 2, "col": 1, "row_span": 1, "col_span": 1, "text": "1", "confidence": 0.40, "source": "trocr", "is_header": False},
                    {"row": 2, "col": 2, "row_span": 1, "col_span": 1, "text": "$250.00", "confidence": 0.90, "source": "trocr", "is_header": False},
                ],
                "confidence": 0.82,
                "has_borders": True,
            }
        ],
        "fields": [
            {
                "name": "invoice_number",
                "value": "12345",
                "normalized_value": "12345",
                "data_type": "string",
                "confidence": 0.95,
                "status": "confident",
                "page": 1,
                "chosen_source": "donut",
                "candidates": [
                    {"source": "donut", "value": "12345", "confidence": 0.95},
                    {"source": "layoutlmv3", "value": "12345", "confidence": 0.88},
                ],
                "validators": [],
            },
            {
                "name": "total",
                "value": "$500.00",
                "normalized_value": "500.00",
                "data_type": "currency",
                "confidence": 0.92,
                "status": "validated",
                "page": 1,
                "chosen_source": "donut",
                "candidates": [
                    {"source": "donut", "value": "$500.00", "confidence": 0.92},
                ],
                "validators": [
                    {"name": "currency_format", "passed": True, "message": "Valid currency format"},
                ],
            },
        ],
        "validation": {
            "passed": True,
            "total_checks": 1,
            "passed_checks": 1,
            "failed_checks": 0,
            "issues": [],
            "details": [
                {"name": "currency_format", "passed": True, "message": "Valid currency"},
            ],
        },
    }


@pytest.fixture
def minimal_result() -> dict:
    """Bare-minimum result with no pages/tables/fields."""
    return {
        "id": "min1",
        "metadata": {
            "filename": "blank.png",
            "file_type": "image",
            "file_size_bytes": 100,
            "processed_at": "2026-01-01T00:00:00",
            "processing_time_seconds": 0.1,
            "docvision_version": "0.1.0",
        },
        "page_count": 0,
        "pages": [],
        "tables": [],
        "fields": [],
        "validation": {
            "passed": True,
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "issues": [],
            "details": [],
        },
    }


# ── generate_markdown ────────────────────────────────────────────────────

class TestGenerateMarkdown:
    """Tests for generate_markdown()."""

    def test_returns_string(self, sample_result):
        md = generate_markdown(sample_result)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_title_contains_filename(self, sample_result):
        md = generate_markdown(sample_result)
        assert "test_invoice.png" in md

    def test_document_metadata_section(self, sample_result):
        md = generate_markdown(sample_result)
        assert "Document Information" in md
        assert "image" in md
        assert "0.1.0" in md

    def test_page_section(self, sample_result):
        md = generate_markdown(sample_result)
        assert "Page 1" in md
        assert "800 × 600" in md or "800" in md

    def test_layout_regions_rendered(self, sample_result):
        md = generate_markdown(sample_result)
        assert "Layout Regions" in md
        assert "2 region(s) detected" in md

    def test_text_lines_rendered(self, sample_result):
        md = generate_markdown(sample_result)
        assert "INVOICE #12345" in md
        assert "TOTAL: $500.00" in md
        assert "trocr" in md

    def test_table_rendered_as_markdown_table(self, sample_result):
        md = generate_markdown(sample_result)
        # Should contain proper markdown table with header row
        assert "| Item | Qty | Price |" in md
        assert "| Widget A | 2 | $250.00 |" in md
        assert "| Widget B | 1 | $250.00 |" in md

    def test_table_metadata(self, sample_result):
        md = generate_markdown(sample_result)
        assert "3 rows × 3 columns" in md
        assert "Has Borders" in md

    def test_low_confidence_cells_collapsible(self, sample_result):
        md = generate_markdown(sample_result)
        assert "Low-confidence cells" in md
        assert "Widget B" in md  # conf 0.45 should show

    def test_raw_text_rendered(self, sample_result):
        md = generate_markdown(sample_result)
        assert "Raw Text" in md
        assert "INVOICE #12345" in md

    def test_fields_rendered(self, sample_result):
        md = generate_markdown(sample_result)
        assert "Extracted Fields" in md
        assert "invoice_number" in md
        assert "total" in md
        assert "$500.00" in md

    def test_candidates_collapsible(self, sample_result):
        md = generate_markdown(sample_result)
        assert "All Candidates" in md
        assert "donut" in md
        assert "layoutlmv3" in md

    def test_validation_section(self, sample_result):
        md = generate_markdown(sample_result)
        assert "Validation Summary" in md
        assert "Passed" in md

    def test_confidence_badges(self, sample_result):
        md = generate_markdown(sample_result)
        assert "🟢" in md  # high confidence items
        assert "🔴" in md  # low confidence cells (0.40, 0.45)

    def test_footer(self, sample_result):
        md = generate_markdown(sample_result)
        assert "DocVision" in md
        assert "Horizon OCR Pipeline" in md

    def test_minimal_result(self, minimal_result):
        md = generate_markdown(minimal_result)
        assert "blank.png" in md
        assert "Document Information" in md
        # Should not crash with empty data
        assert isinstance(md, str)

    def test_no_pages(self, minimal_result):
        md = generate_markdown(minimal_result)
        # Should not contain page sections
        assert "Page 1" not in md

    def test_empty_fields(self, minimal_result):
        md = generate_markdown(minimal_result)
        # Should not contain fields section
        assert "Extracted Fields" not in md

    def test_pipe_in_text_escaped(self):
        """Pipes in cell text should be escaped."""
        data = {
            "id": "esc1",
            "metadata": {"filename": "pipe.png", "file_type": "image",
                         "file_size_bytes": 0, "processed_at": "", "processing_time_seconds": 0,
                         "docvision_version": "0.1.0"},
            "page_count": 1,
            "pages": [{
                "number": 1,
                "metadata": {"width": 100, "height": 100, "dpi": 72,
                             "content_type": "unknown", "readability": "good",
                             "readability_issues": []},
                "layout_regions": [],
                "text_lines": [{"id": "x", "text": "A|B|C", "words": [],
                                "polygon": None,
                                "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                                "confidence": 0.9, "source": "trocr",
                                "content_type": "unknown"}],
                "tables": [],
                "raw_text": "A|B|C",
            }],
            "tables": [],
            "fields": [],
            "validation": {"passed": True, "total_checks": 0, "passed_checks": 0,
                           "failed_checks": 0, "issues": [], "details": []},
        }
        md = generate_markdown(data)
        # Pipe should be escaped in the table row
        assert "A\\|B\\|C" in md

    def test_readability_icons(self, sample_result):
        md = generate_markdown(sample_result)
        assert "✅" in md  # good readability

    def test_failed_validation(self):
        """Test rendering when validation fails."""
        data = {
            "id": "fail1",
            "metadata": {"filename": "fail.png", "file_type": "image",
                         "file_size_bytes": 0, "processed_at": "", "processing_time_seconds": 0,
                         "docvision_version": "0.1.0"},
            "page_count": 0,
            "pages": [],
            "tables": [],
            "fields": [],
            "validation": {
                "passed": False,
                "total_checks": 2,
                "passed_checks": 1,
                "failed_checks": 1,
                "issues": ["Missing required field: total"],
                "details": [
                    {"name": "required_fields", "passed": False, "message": "total missing"},
                    {"name": "date_format", "passed": True, "message": "OK"},
                ],
            },
        }
        md = generate_markdown(data)
        assert "❌" in md
        assert "Failed" in md
        assert "Missing required field" in md


# ── save_markdown ────────────────────────────────────────────────────────

class TestSaveMarkdown:
    """Tests for save_markdown()."""

    def test_creates_local_subfolder(self, sample_result, tmp_path):
        path = save_markdown(
            data=sample_result,
            output_dir=str(tmp_path / "md_out"),
            processing_mode="local",
            filename_stem="test_invoice.png",
        )
        assert Path(path).exists()
        assert "Local" in path
        assert path.endswith(".md")
        content = Path(path).read_text(encoding="utf-8")
        assert "test_invoice.png" in content

    def test_creates_azure_subfolder(self, sample_result, tmp_path):
        path = save_markdown(
            data=sample_result,
            output_dir=str(tmp_path / "md_out"),
            processing_mode="azure",
            filename_stem="test_invoice.png",
        )
        assert Path(path).exists()
        assert "Azure_Cloud" in path

    def test_file_content_is_valid_markdown(self, sample_result, tmp_path):
        path = save_markdown(
            data=sample_result,
            output_dir=str(tmp_path / "md_out"),
            processing_mode="local",
            filename_stem="test_invoice.png",
        )
        content = Path(path).read_text(encoding="utf-8")
        # Should start with a heading
        assert content.startswith("#")
        # Should contain markdown table separators
        assert "---" in content

    def test_filename_format(self, sample_result, tmp_path):
        path = save_markdown(
            data=sample_result,
            output_dir=str(tmp_path / "md_out"),
            processing_mode="local",
            filename_stem="my_document.pdf",
        )
        assert "my_document.md" in path

    def test_no_doc_id(self, sample_result, tmp_path):
        path = save_markdown(
            data=sample_result,
            output_dir=str(tmp_path / "md_out"),
            processing_mode="local",
            filename_stem="doc.png",
        )
        assert "doc.md" in path

    def test_idempotent(self, sample_result, tmp_path):
        """Saving twice should overwrite without error."""
        for _ in range(2):
            path = save_markdown(
                data=sample_result,
                output_dir=str(tmp_path / "md_out"),
                processing_mode="local",
                filename_stem="doc.png",
            )
        assert Path(path).exists()


# ── Edge cases ───────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and empty data."""

    def test_empty_table_cells(self):
        data = {
            "id": "e1",
            "metadata": {"filename": "e.png", "file_type": "image",
                         "file_size_bytes": 0, "processed_at": "", "processing_time_seconds": 0,
                         "docvision_version": "0.1.0"},
            "page_count": 1,
            "pages": [{
                "number": 1,
                "metadata": {"width": 100, "height": 100, "dpi": 72,
                             "content_type": "unknown", "readability": "poor",
                             "readability_issues": ["low_contrast"]},
                "layout_regions": [],
                "text_lines": [],
                "tables": [{
                    "id": "t1", "page": 1,
                    "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                    "rows": 2, "cols": 2, "cells": [],
                    "confidence": 0.5, "has_borders": False,
                }],
                "raw_text": "",
            }],
            "tables": [],
            "fields": [],
            "validation": {"passed": True, "total_checks": 0, "passed_checks": 0,
                           "failed_checks": 0, "issues": [], "details": []},
        }
        md = generate_markdown(data)
        assert "No cell data available" in md
        assert "low_contrast" in md

    def test_table_no_headers(self):
        """Tables where no cell is marked as header."""
        data = {
            "id": "nh1",
            "metadata": {"filename": "nh.png", "file_type": "image",
                         "file_size_bytes": 0, "processed_at": "", "processing_time_seconds": 0,
                         "docvision_version": "0.1.0"},
            "page_count": 1,
            "pages": [{
                "number": 1,
                "metadata": {"width": 100, "height": 100, "dpi": 72,
                             "content_type": "unknown", "readability": "good",
                             "readability_issues": []},
                "layout_regions": [],
                "text_lines": [],
                "tables": [{
                    "id": "t1", "page": 1,
                    "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
                    "rows": 2, "cols": 2,
                    "cells": [
                        {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "A", "confidence": 0.9, "source": "trocr", "is_header": False},
                        {"row": 0, "col": 1, "row_span": 1, "col_span": 1, "text": "B", "confidence": 0.9, "source": "trocr", "is_header": False},
                        {"row": 1, "col": 0, "row_span": 1, "col_span": 1, "text": "C", "confidence": 0.9, "source": "trocr", "is_header": False},
                        {"row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "D", "confidence": 0.9, "source": "trocr", "is_header": False},
                    ],
                    "confidence": 0.8, "has_borders": True,
                }],
                "raw_text": "",
            }],
            "tables": [],
            "fields": [],
            "validation": {"passed": True, "total_checks": 0, "passed_checks": 0,
                           "failed_checks": 0, "issues": [], "details": []},
        }
        md = generate_markdown(data)
        # Should use generic column headers
        assert "Col 1" in md
        assert "Col 2" in md
        # All rows should be data rows
        assert "| A | B |" in md
        assert "| C | D |" in md

    def test_large_file_size_formatting(self):
        data = {
            "id": "sz",
            "metadata": {"filename": "big.pdf", "file_type": "pdf",
                         "file_size_bytes": 5_242_880, "processed_at": "",
                         "processing_time_seconds": 10.0, "docvision_version": "0.1.0"},
            "page_count": 0, "pages": [], "tables": [], "fields": [],
            "validation": {"passed": True, "total_checks": 0, "passed_checks": 0,
                           "failed_checks": 0, "issues": [], "details": []},
        }
        md = generate_markdown(data)
        assert "5.00 MB" in md

    def test_multiline_text_in_cell_flattened(self):
        """Newlines in cell text should be replaced by spaces."""
        data = {
            "id": "nl",
            "metadata": {"filename": "nl.png", "file_type": "image",
                         "file_size_bytes": 0, "processed_at": "", "processing_time_seconds": 0,
                         "docvision_version": "0.1.0"},
            "page_count": 1,
            "pages": [{
                "number": 1,
                "metadata": {"width": 100, "height": 100, "dpi": 72,
                             "content_type": "unknown", "readability": "good",
                             "readability_issues": []},
                "layout_regions": [],
                "text_lines": [
                    {"id": "x", "text": "line1\nline2", "words": [],
                     "polygon": None,
                     "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                     "confidence": 0.9, "source": "trocr",
                     "content_type": "unknown"}
                ],
                "tables": [],
                "raw_text": "line1\nline2",
            }],
            "tables": [], "fields": [],
            "validation": {"passed": True, "total_checks": 0, "passed_checks": 0,
                           "failed_checks": 0, "issues": [], "details": []},
        }
        md = generate_markdown(data)
        # Inside the text lines table, newlines should be replaced
        lines = md.split("\n")
        text_table_lines = [l for l in lines if "line1" in l and "|" in l]
        for line in text_table_lines:
            assert "\n" not in line.split("|")[2]  # cell should not have raw newline
