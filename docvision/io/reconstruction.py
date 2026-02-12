"""
Reconstruction prompt builder for LLM-based PDF reconstruction.

This module generates a flat, LLM-friendly structure from nested document JSON,
enabling accurate PDF reconstruction by providing:
- Deduplicated text elements sorted in reading order
- Explicit table grids with cell positions
- Page dimensions and font size estimates
- Fields summary for semantic context
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ElementType(str, Enum):
    """Type of visual element in the document."""

    TEXT = "text"
    TABLE = "table"
    TITLE = "title"


class RenderElement(BaseModel):
    """A single renderable element for PDF reconstruction."""

    type: ElementType = Field(description="Element type: text, table, or title")
    page: int = Field(description="1-based page number")
    text: str = Field(description="Text content (or table_id reference for tables)")
    x: int = Field(description="Left position in pixels")
    y: int = Field(description="Top position in pixels")
    width: int = Field(description="Width in pixels")
    height: int = Field(description="Height in pixels")
    font_size: float | None = Field(default=None, description="Estimated font size in points")
    confidence: float = Field(default=1.0, description="OCR confidence 0-1")
    bold: bool = Field(default=False, description="Whether text is bold")
    row: int | None = Field(default=None, description="Table row for table elements")
    col: int | None = Field(default=None, description="Table column for table elements")


class TableGrid(BaseModel):
    """Flat representation of a table for reconstruction."""

    page: int = Field(description="1-based page number")
    x: int = Field(description="Left position in pixels")
    y: int = Field(description="Top position in pixels")
    width: int = Field(description="Width in pixels")
    height: int = Field(description="Height in pixels")
    rows: int = Field(description="Number of rows")
    cols: int = Field(description="Number of columns")
    cells: list[dict[str, Any]] = Field(
        description="List of cells with row, col, text, and optional bbox"
    )


def _bbox_to_coords(bbox: dict[str, Any] | list | None) -> tuple[int, int, int, int]:
    """
    Convert various bbox formats to (x, y, width, height).

    Handles:
    - Dict with x, y, width, height
    - Dict with x1, y1, x2, y2
    - Dict with x0, y0, x1, y1
    - List [x0, y0, x1, y1]
    """
    if bbox is None:
        return (0, 0, 0, 0)

    if isinstance(bbox, list) and len(bbox) >= 4:
        x0, y0, x1, y1 = bbox[:4]
        return (int(x0), int(y0), max(0, int(x1 - x0)), max(0, int(y1 - y0)))

    if isinstance(bbox, dict):
        if "width" in bbox:
            return (
                int(bbox.get("x", 0)),
                int(bbox.get("y", 0)),
                int(bbox.get("width", 0)),
                int(bbox.get("height", 0)),
            )
        elif "x1" in bbox and "x2" in bbox:
            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))
            return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
        elif "x0" in bbox:
            x0 = int(bbox.get("x0", 0))
            y0 = int(bbox.get("y0", 0))
            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    return (0, 0, 0, 0)


def _estimate_font_size(height: int) -> str:
    """
    Estimate font size category from bounding box height.

    Returns:
        "title" for heights >= 80
        "large" for heights >= 50
        "normal" for heights >= 30
        "small" for smaller heights
    """
    if height >= 80:
        return "title"
    elif height >= 50:
        return "large"
    elif height >= 30:
        return "normal"
    else:
        return "small"


def _reading_order_key(element: dict[str, Any]) -> tuple[int, int, float]:
    """
    Generate sort key for reading order.

    Sort by: page, y-band (grouped into ~50px bands), then x position.
    This handles multi-column layouts better than pure top-to-bottom.
    """
    page = element.get("page", 1)
    y = element.get("y", 0)
    x = element.get("x", 0)
    # Group into y-bands of 50 pixels to handle slight vertical misalignment
    y_band = int(y // 50)
    return (page, y_band, x)


def build_reconstruction_prompt(document_data: dict[str, Any]) -> dict[str, Any]:
    """
    Build a flat, LLM-friendly reconstruction prompt from document data.

    This function:
    1. Extracts all text elements from nested structures
    2. Deduplicates elements (same text in same y-band)
    3. Sorts elements in reading order (page -> y-band -> x)
    4. Extracts tables as flat cell grids
    5. Includes fields summary for semantic context

    Args:
        document_data: The document JSON (from Azure or Local pipeline)

    Returns:
        dict with reconstruction data for PDF reconstruction
    """
    result: dict[str, Any] = {
        "version": "1.0",
        "instruction": "Use this flat structure to reconstruct the document layout. Elements are sorted in reading order.",
        "pages": [],
        "elements": [],
        "tables": [],
        "fields_summary": {},
    }

    # Extract page dimensions
    pages = document_data.get("pages", [])
    for page_data in pages:
        page_num = page_data.get("number", page_data.get("page_number", 1))
        metadata = page_data.get("metadata", {})
        width = metadata.get("width", page_data.get("width", 612))
        height = metadata.get("height", page_data.get("height", 792))
        result["pages"].append({
            "page": page_num,
            "width": width,
            "height": height,
        })

    # Track seen texts for deduplication
    seen_texts: dict[int, set[tuple[str, int]]] = {}

    # Extract elements from pages
    for page_data in pages:
        page_num = page_data.get("number", page_data.get("page_number", 1))
        if page_num not in seen_texts:
            seen_texts[page_num] = set()

        # Extract from text_lines
        text_lines = page_data.get("text_lines", [])
        for line in text_lines:
            text = line.get("text", "").strip()
            if not text:
                continue

            bbox = line.get("bbox")
            x, y, width, height = _bbox_to_coords(bbox)

            # Deduplication
            y_band = int(y // 50)
            dedup_key = (text, y_band)
            if dedup_key in seen_texts[page_num]:
                continue
            seen_texts[page_num].add(dedup_key)

            confidence = line.get("confidence", 1.0)

            # Extract style if available
            style = line.get("style")
            font_name = None
            font_size = None
            is_bold = False
            is_italic = False
            if style:
                if isinstance(style, dict):
                    font_name = style.get("font_name")
                    font_size = style.get("font_size")
                    is_bold = style.get("bold", False)
                    is_italic = style.get("italic", False)
                elif hasattr(style, "font_name"):
                    font_name = style.font_name
                    font_size = style.font_size
                    is_bold = getattr(style, "bold", False)
                    is_italic = getattr(style, "italic", False)

            result["elements"].append({
                "type": "text",
                "page": page_num,
                "text": text,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "confidence": confidence,
                "font_name": font_name,
                "font_size": font_size,
                "bold": is_bold,
                "italic": is_italic,
            })

        # Extract from layout_regions (titles, etc.)
        layout_regions = page_data.get("layout_regions", [])
        for region in layout_regions:
            region_type = region.get("type", "text")
            bbox = region.get("bbox")
            x, y, width, height = _bbox_to_coords(bbox)
            confidence = region.get("confidence", 1.0)

            # Get text from text_lines within region
            region_lines = region.get("text_lines", [])
            for line in region_lines:
                text = line.get("text", "").strip()
                if not text:
                    continue

                # Use line's bbox if available, otherwise use region bbox
                line_bbox = line.get("bbox")
                if line_bbox:
                    lx, ly, lw, lh = _bbox_to_coords(line_bbox)
                else:
                    lx, ly, lw, lh = x, y, width, height

                # Deduplication - check if already seen
                y_band = int(ly // 50)
                dedup_key = (text, y_band)
                if dedup_key in seen_texts[page_num]:
                    continue
                seen_texts[page_num].add(dedup_key)

                elem_type = "title" if region_type == "title" else "text"

                # Extract style if available
                style = line.get("style")
                font_name = None
                font_size = None
                is_bold = region_type == "title"
                is_italic = False
                if style:
                    if isinstance(style, dict):
                        font_name = style.get("font_name")
                        font_size = style.get("font_size")
                        is_bold = style.get("bold", is_bold)
                        is_italic = style.get("italic", False)
                    elif hasattr(style, "font_name"):
                        font_name = style.font_name
                        font_size = style.font_size
                        is_bold = getattr(style, "bold", is_bold)
                        is_italic = getattr(style, "italic", False)

                result["elements"].append({
                    "type": elem_type,
                    "page": page_num,
                    "text": text,
                    "x": lx,
                    "y": ly,
                    "width": lw,
                    "height": lh,
                    "confidence": line.get("confidence", confidence),
                    "font_name": font_name,
                    "font_size": font_size,
                    "bold": is_bold,
                    "italic": is_italic,
                })

        # Extract from lines (alternative format)
        lines = page_data.get("lines", [])
        for line in lines:
            text = line.get("text", "").strip()
            if not text:
                continue

            bbox = line.get("bbox")
            x, y, width, height = _bbox_to_coords(bbox)

            # Deduplication
            y_band = int(y // 50)
            dedup_key = (text, y_band)
            if dedup_key in seen_texts[page_num]:
                continue
            seen_texts[page_num].add(dedup_key)

            confidence = line.get("confidence", 1.0)

            # Extract style if available
            style = line.get("style")
            font_name = None
            font_size = None
            is_bold = False
            is_italic = False
            if style:
                if isinstance(style, dict):
                    font_name = style.get("font_name")
                    font_size = style.get("font_size")
                    is_bold = style.get("bold", False)
                    is_italic = style.get("italic", False)
                elif hasattr(style, "font_name"):
                    font_name = style.font_name
                    font_size = style.font_size
                    is_bold = getattr(style, "bold", False)
                    is_italic = getattr(style, "italic", False)

            result["elements"].append({
                "type": "text",
                "page": page_num,
                "text": text,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "confidence": confidence,
                "font_name": font_name,
                "font_size": font_size,
                "bold": is_bold,
                "italic": is_italic,
            })

        # Extract tables from page
        page_tables = page_data.get("tables", [])
        for table in page_tables:
            bbox = table.get("bbox", {})
            x, y, width, height = _bbox_to_coords(bbox)

            rows_data = table.get("rows", [])
            cells_list: list[dict[str, Any]] = []
            row_count = 0
            col_count = 0

            if isinstance(rows_data, list):
                row_count = len(rows_data)
                for row_idx, row in enumerate(rows_data):
                    row_cells = row.get("cells", [])
                    col_count = max(col_count, len(row_cells))

                    for col_idx, cell in enumerate(row_cells):
                        if isinstance(cell, dict):
                            cell_text = cell.get("text", "")
                            cell_bbox = cell.get("bbox")
                        else:
                            cell_text = str(cell) if cell else ""
                            cell_bbox = None

                        cell_dict: dict[str, Any] = {
                            "row": row_idx,
                            "col": col_idx,
                            "text": cell_text,
                        }
                        if cell_bbox:
                            cx, cy, cw, ch = _bbox_to_coords(cell_bbox)
                            cell_dict["bbox"] = {"x": cx, "y": cy, "width": cw, "height": ch}
                        cells_list.append(cell_dict)

            result["tables"].append({
                "page": page_num,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "rows": row_count,
                "cols": col_count,
                "cells": cells_list,
            })

    # Extract tables from top-level tables array
    doc_tables = document_data.get("tables", [])
    for table in doc_tables:
        page_num = table.get("page_number", 1)
        bbox = table.get("bbox", {})
        x, y, width, height = _bbox_to_coords(bbox)

        rows_data = table.get("rows", [])
        cells_list: list[dict[str, Any]] = []
        row_count = 0
        col_count = 0

        # Format 1: rows is an int, cells are in separate "cells" array
        if isinstance(rows_data, int):
            row_count = rows_data
            col_count = table.get("cols", 0)
            raw_cells = table.get("cells", [])

            for cell in raw_cells:
                cell_dict: dict[str, Any] = {
                    "row": cell.get("row_index", cell.get("row", 0)),
                    "col": cell.get("col_index", cell.get("col", 0)),
                    "text": cell.get("text", ""),
                }
                cell_bbox = cell.get("bbox")
                if cell_bbox:
                    cx, cy, cw, ch = _bbox_to_coords(cell_bbox)
                    cell_dict["bbox"] = {"x": cx, "y": cy, "width": cw, "height": ch}
                cells_list.append(cell_dict)

        # Format 2: rows is a list of row objects with cells
        elif isinstance(rows_data, list):
            row_count = len(rows_data)
            for row_idx, row in enumerate(rows_data):
                row_cells = row.get("cells", [])
                col_count = max(col_count, len(row_cells))

                for col_idx, cell in enumerate(row_cells):
                    if isinstance(cell, dict):
                        cell_text = cell.get("text", "")
                        cell_bbox = cell.get("bbox")
                    else:
                        cell_text = str(cell) if cell else ""
                        cell_bbox = None

                    cell_dict = {
                        "row": row_idx,
                        "col": col_idx,
                        "text": cell_text,
                    }
                    if cell_bbox:
                        cx, cy, cw, ch = _bbox_to_coords(cell_bbox)
                        cell_dict["bbox"] = {"x": cx, "y": cy, "width": cw, "height": ch}
                    cells_list.append(cell_dict)

        result["tables"].append({
            "page": page_num,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "rows": row_count,
            "cols": col_count,
            "cells": cells_list,
        })

    # Sort elements in reading order
    result["elements"].sort(key=_reading_order_key)

    # Extract fields summary
    fields = document_data.get("fields", [])
    if isinstance(fields, list):
        for field in fields:
            name = field.get("name", "")
            value = field.get("value")
            if name and value is not None:
                result["fields_summary"][name] = value
    elif isinstance(fields, dict):
        for name, field_data in fields.items():
            if isinstance(field_data, dict):
                value = field_data.get("value", field_data.get("content", ""))
            else:
                value = field_data
            if value:
                result["fields_summary"][name] = value

    return result


def add_reconstruction_to_document(document_data: dict[str, Any]) -> dict[str, Any]:
    """
    Add reconstruction_prompt section to document data in-place.

    This is the main entry point called from the orchestrator's _save_output method.

    Args:
        document_data: The document JSON to modify

    Returns:
        The modified document_data with reconstruction_prompt added
    """
    reconstruction = build_reconstruction_prompt(document_data)
    document_data["reconstruction_prompt"] = reconstruction
    return document_data
