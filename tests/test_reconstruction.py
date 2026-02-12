"""Tests for document reconstruction prompt builder."""

import pytest
from docvision.io.reconstruction import (
    build_reconstruction_prompt,
    add_reconstruction_to_document,
    ElementType,
    RenderElement,
    TableGrid,
    _reading_order_key,
    _bbox_to_coords,
    _estimate_font_size,
)


class TestBboxToCoords:
    """Tests for bbox coordinate conversion."""
    
    def test_valid_bbox(self):
        bbox = {"x1": 100.5, "y1": 200.5, "x2": 300.5, "y2": 400.5}
        x, y, w, h = _bbox_to_coords(bbox)
        assert x == 100
        assert y == 200
        assert w == 200
        assert h == 200
    
    def test_none_bbox(self):
        x, y, w, h = _bbox_to_coords(None)
        assert (x, y, w, h) == (0, 0, 0, 0)
    
    def test_empty_bbox(self):
        x, y, w, h = _bbox_to_coords({})
        assert (x, y, w, h) == (0, 0, 0, 0)
    
    def test_negative_dimensions_clamped(self):
        # x2 < x1 should result in width 0
        bbox = {"x1": 300, "y1": 200, "x2": 100, "y2": 400}
        x, y, w, h = _bbox_to_coords(bbox)
        assert w == 0
        assert h == 200


class TestEstimateFontSize:
    """Tests for font size estimation."""
    
    def test_title_size(self):
        assert _estimate_font_size(120) == "title"
    
    def test_large_size(self):
        assert _estimate_font_size(70) == "large"
    
    def test_normal_size(self):
        assert _estimate_font_size(40) == "normal"
    
    def test_small_size(self):
        assert _estimate_font_size(20) == "small"


class TestReadingOrderKey:
    """Tests for reading order sorting."""
    
    def test_page_ordering(self):
        elem1 = {"page": 1, "x": 0, "y": 0}
        elem2 = {"page": 2, "x": 0, "y": 0}
        assert _reading_order_key(elem1) < _reading_order_key(elem2)
    
    def test_y_ordering(self):
        elem1 = {"page": 1, "x": 0, "y": 100}
        elem2 = {"page": 1, "x": 0, "y": 200}
        assert _reading_order_key(elem1) < _reading_order_key(elem2)
    
    def test_x_ordering_within_band(self):
        # Elements in same y-band (within 50px)
        elem1 = {"page": 1, "x": 100, "y": 105}
        elem2 = {"page": 1, "x": 200, "y": 110}
        assert _reading_order_key(elem1) < _reading_order_key(elem2)


class TestBuildReconstructionPrompt:
    """Tests for the main reconstruction prompt builder."""
    
    def test_empty_document(self):
        doc = {"pages": [], "tables": [], "fields": []}
        result = build_reconstruction_prompt(doc)
        
        assert result["version"] == "1.0"
        assert "instruction" in result
        assert result["pages"] == []
        assert result["elements"] == []
        assert result["tables"] == []
        assert result["fields_summary"] == {}
    
    def test_page_dimensions(self):
        doc = {
            "pages": [
                {"number": 1, "metadata": {"width": 4250, "height": 5500, "dpi": 500}},
                {"number": 2, "metadata": {"width": 4250, "height": 5500, "dpi": 500}},
            ],
            "tables": [],
            "fields": []
        }
        result = build_reconstruction_prompt(doc)
        
        assert len(result["pages"]) == 2
        assert result["pages"][0]["page"] == 1
        assert result["pages"][0]["width"] == 4250
        assert result["pages"][0]["height"] == 5500
    
    def test_text_lines_extraction(self):
        doc = {
            "pages": [{
                "number": 1,
                "metadata": {"width": 1000, "height": 1000, "dpi": 300},
                "text_lines": [
                    {
                        "text": "Hello World",
                        "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 240},
                        "confidence": 0.95
                    }
                ],
                "layout_regions": [],
                "tables": []
            }],
            "tables": [],
            "fields": []
        }
        result = build_reconstruction_prompt(doc)
        
        assert len(result["elements"]) == 1
        elem = result["elements"][0]
        assert elem["type"] == "text"
        assert elem["text"] == "Hello World"
        assert elem["x"] == 100
        assert elem["y"] == 200
        assert elem["width"] == 200
        assert elem["height"] == 40
        assert elem["confidence"] == 0.95
    
    def test_duplicate_removal(self):
        doc = {
            "pages": [{
                "number": 1,
                "metadata": {"width": 1000, "height": 1000, "dpi": 300},
                "text_lines": [
                    {"text": "Same Text", "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 240}},
                    {"text": "Same Text", "bbox": {"x1": 100, "y1": 205, "x2": 300, "y2": 245}},  # Near duplicate
                ],
                "layout_regions": [],
                "tables": []
            }],
            "tables": [],
            "fields": []
        }
        result = build_reconstruction_prompt(doc)
        
        # Should keep only one instance
        assert len(result["elements"]) == 1
    
    def test_reading_order_sorting(self):
        doc = {
            "pages": [{
                "number": 1,
                "metadata": {"width": 1000, "height": 1000, "dpi": 300},
                "text_lines": [
                    {"text": "Bottom", "bbox": {"x1": 100, "y1": 500, "x2": 200, "y2": 540}},
                    {"text": "Top", "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 140}},
                    {"text": "Middle", "bbox": {"x1": 100, "y1": 300, "x2": 200, "y2": 340}},
                ],
                "layout_regions": [],
                "tables": []
            }],
            "tables": [],
            "fields": []
        }
        result = build_reconstruction_prompt(doc)
        
        texts = [e["text"] for e in result["elements"]]
        assert texts == ["Top", "Middle", "Bottom"]
    
    def test_table_extraction(self):
        doc = {
            "pages": [{
                "number": 1,
                "metadata": {"width": 1000, "height": 1000, "dpi": 300},
                "text_lines": [],
                "layout_regions": [],
                "tables": [{
                    "bbox": {"x1": 100, "y1": 200, "x2": 500, "y2": 400},
                    "rows": [
                        {"cells": [
                            {"text": "A", "bbox": {"x1": 100, "y1": 200, "x2": 200, "y2": 250}},
                            {"text": "B", "bbox": {"x1": 200, "y1": 200, "x2": 300, "y2": 250}}
                        ]},
                        {"cells": [
                            {"text": "C", "bbox": {"x1": 100, "y1": 250, "x2": 200, "y2": 300}},
                            {"text": "D", "bbox": {"x1": 200, "y1": 250, "x2": 300, "y2": 300}}
                        ]}
                    ]
                }]
            }],
            "tables": [],
            "fields": []
        }
        result = build_reconstruction_prompt(doc)
        
        assert len(result["tables"]) == 1
        table = result["tables"][0]
        assert table["page"] == 1
        assert table["rows"] == 2
        assert table["cols"] == 2
        assert len(table["cells"]) == 4
    
    def test_fields_summary(self):
        doc = {
            "pages": [],
            "tables": [],
            "fields": [
                {"name": "invoice_number", "value": "INV-001"},
                {"name": "total", "value": "$1,234.56"},
                {"name": "empty_field", "value": None},
            ]
        }
        result = build_reconstruction_prompt(doc)
        
        assert result["fields_summary"]["invoice_number"] == "INV-001"
        assert result["fields_summary"]["total"] == "$1,234.56"
        assert "empty_field" not in result["fields_summary"]
    
    def test_title_region_handling(self):
        doc = {
            "pages": [{
                "number": 1,
                "metadata": {"width": 1000, "height": 1000, "dpi": 300},
                "text_lines": [],
                "layout_regions": [{
                    "type": "title",
                    "bbox": {"x1": 100, "y1": 50, "x2": 400, "y2": 100},
                    "confidence": 0.9,
                    "text_lines": [
                        {"text": "Document Title", "confidence": 0.95}
                    ]
                }],
                "tables": []
            }],
            "tables": [],
            "fields": []
        }
        result = build_reconstruction_prompt(doc)
        
        assert len(result["elements"]) == 1
        elem = result["elements"][0]
        assert elem["type"] == "title"
        assert elem["bold"] is True


class TestAddReconstructionToDocument:
    """Tests for in-place document modification."""
    
    def test_adds_reconstruction_key(self):
        doc = {"pages": [], "tables": [], "fields": []}
        result = add_reconstruction_to_document(doc)
        
        assert "reconstruction_prompt" in result
        assert result["reconstruction_prompt"]["version"] == "1.0"
    
    def test_modifies_in_place(self):
        doc = {"pages": [], "tables": [], "fields": []}
        result = add_reconstruction_to_document(doc)
        
        # Should be the same object
        assert result is doc


class TestRenderElement:
    """Tests for the RenderElement model."""
    
    def test_valid_element(self):
        elem = RenderElement(
            type=ElementType.TEXT,
            text="Hello",
            page=1,
            x=100,
            y=200,
            width=150,
            height=40
        )
        assert elem.type == ElementType.TEXT
        assert elem.text == "Hello"
    
    def test_default_values(self):
        elem = RenderElement(
            type=ElementType.TEXT,
            text="Test",
            page=1,
            x=0,
            y=0,
            width=100,
            height=30
        )
        assert elem.confidence == 1.0
        assert elem.font_size is None
        assert elem.bold is False
        assert elem.row is None
        assert elem.col is None


class TestTableGrid:
    """Tests for the TableGrid model."""
    
    def test_valid_table(self):
        grid = TableGrid(
            page=1,
            x=100,
            y=200,
            width=400,
            height=300,
            rows=3,
            cols=4,
            cells=[{"row": 0, "col": 0, "text": "A1"}]
        )
        assert grid.rows == 3
        assert grid.cols == 4
        assert len(grid.cells) == 1
