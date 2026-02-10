"""
Markdown report generator for DocVision OCR results.

Converts JSON output into a structured, human-readable Markdown file
with proper tables, confidence indicators, and organized sections.
"""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


# ── Confidence helpers ────────────────────────────────────────────────────

def _conf_badge(confidence: float) -> str:
    """Return an emoji badge based on confidence level."""
    if confidence >= 0.9:
        return "🟢"  # high
    if confidence >= 0.7:
        return "🟡"  # medium
    if confidence >= 0.5:
        return "🟠"  # low-medium
    return "🔴"  # low


def _pct(confidence: float) -> str:
    """Format confidence as a percentage string."""
    return f"{confidence * 100:.1f}%"


def _format_bbox(bbox: Optional[Dict[str, Any]]) -> str:
    """Format a bounding box dict as a compact position string."""
    if not bbox:
        return "—"
    x1 = bbox.get("x1", 0)
    y1 = bbox.get("y1", 0)
    x2 = bbox.get("x2", 0)
    y2 = bbox.get("y2", 0)
    return f"`({x1},{y1})→({x2},{y2})`"


# ── Section renderers ────────────────────────────────────────────────────

def _render_metadata(data: Dict[str, Any]) -> str:
    """Render the document metadata block."""
    meta = data.get("metadata", {})
    lines = [
        "## 📄 Document Information\n",
        "| Property | Value |",
        "|----------|-------|",
        f"| **Filename** | `{meta.get('filename', 'N/A')}` |",
        f"| **File Type** | {meta.get('file_type', 'N/A')} |",
        f"| **File Size** | {_format_size(meta.get('file_size_bytes', 0))} |",
        f"| **Processed At** | {meta.get('processed_at', 'N/A')} |",
        f"| **Processing Time** | {meta.get('processing_time_seconds', 0):.2f}s |",
        f"| **DocVision Version** | {meta.get('docvision_version', 'N/A')} |",
        f"| **Page Count** | {data.get('page_count', 0)} |",
        "",
    ]
    return "\n".join(lines)


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def _render_page_metadata(page: Dict[str, Any]) -> str:
    """Render per-page metadata."""
    meta = page.get("metadata", {})
    page_num = page.get("number", "?")
    readability = meta.get("readability", "N/A")
    readability_icon = {"good": "✅", "fair": "⚠️", "poor": "❌"}.get(readability, "❓")

    lines = [
        f"### Page {page_num} — Overview\n",
        "| Property | Value |",
        "|----------|-------|",
        f"| **Dimensions** | {meta.get('width', '?')} × {meta.get('height', '?')} px |",
        f"| **DPI** | {meta.get('dpi', '?')} |",
        f"| **Content Type** | {meta.get('content_type', 'N/A')} |",
        f"| **Readability** | {readability_icon} {readability} |",
    ]

    issues = meta.get("readability_issues", [])
    if issues:
        lines.append(f"| **Readability Issues** | {', '.join(issues)} |")

    lines.append("")
    return "\n".join(lines)


def _render_layout_regions(regions: List[Dict[str, Any]], page_num: int) -> str:
    """Render layout regions as a summary table."""
    if not regions:
        return ""

    lines = [
        f"### 🗺️ Layout Regions (Page {page_num})\n",
        f"*{len(regions)} region(s) detected*\n",
        "| # | Type | Position | Confidence | Content Type |",
        "|---|------|----------|-----------|--------------|",
    ]

    for i, r in enumerate(regions, 1):
        rtype = r.get("type", "unknown")
        conf = r.get("confidence", 0)
        ctype = r.get("content_type", "unknown")
        pos = _format_bbox(r.get("bbox"))
        lines.append(
            f"| {i} | **{rtype}** | {pos} | {_conf_badge(conf)} {_pct(conf)} | {ctype} |"
        )

    lines.append("")
    return "\n".join(lines)


def _render_text_lines(text_lines: List[Dict[str, Any]], page_num: int) -> str:
    """Render extracted text lines."""
    if not text_lines:
        return ""

    lines = [
        f"### 📝 Extracted Text Lines (Page {page_num})\n",
        f"*{len(text_lines)} line(s) detected*\n",
        "| # | Text | Position | Confidence | Source |",
        "|---|------|----------|-----------|--------|",
    ]

    for i, tl in enumerate(text_lines, 1):
        text = _escape_md(tl.get("text", ""))
        conf = tl.get("confidence", 0)
        source = tl.get("source", "N/A")
        pos = _format_bbox(tl.get("bbox"))
        lines.append(
            f"| {i} | {text} | {pos} | {_conf_badge(conf)} {_pct(conf)} | {source} |"
        )

    lines.append("")
    return "\n".join(lines)


def _render_table(table: Dict[str, Any], table_idx: int) -> str:
    """Render a single table as a proper Markdown table."""
    rows = table.get("rows", 0)
    cols = table.get("cols", 0)
    cells = table.get("cells", [])
    conf = table.get("confidence", 0)
    page = table.get("page", "?")
    has_borders = table.get("has_borders", False)

    bbox = table.get("bbox")
    pos_line = f"- **Position:** {_format_bbox(bbox)}" if bbox else ""

    lines = [
        f"#### 📊 Table {table_idx} (Page {page})\n",
        f"- **Size:** {rows} rows × {cols} columns",
        f"- **Confidence:** {_conf_badge(conf)} {_pct(conf)}",
        f"- **Has Borders:** {'Yes' if has_borders else 'No'}",
    ]
    if pos_line:
        lines.append(pos_line)
    lines.append("")

    if not cells or rows == 0 or cols == 0:
        lines.append("*No cell data available.*\n")
        return "\n".join(lines)

    # Build a 2-D grid from cells
    grid: List[List[str]] = [["" for _ in range(cols)] for _ in range(rows)]
    header_rows: set[int] = set()

    for cell in cells:
        r = cell.get("row", 0)
        c = cell.get("col", 0)
        text = _escape_md(cell.get("text", "").strip())
        if cell.get("is_header", False):
            header_rows.add(r)
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = text

    # Render as markdown table
    # Header row
    if header_rows:
        hr = min(header_rows)
        header = grid[hr]
        lines.append("| " + " | ".join(h or " " for h in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        start = hr + 1
    else:
        # Use generic column headers
        lines.append("| " + " | ".join(f"Col {c + 1}" for c in range(cols)) + " |")
        lines.append("| " + " | ".join("---" for _ in range(cols)) + " |")
        start = 0

    for r in range(start, rows):
        row_data = grid[r]
        lines.append("| " + " | ".join(v or " " for v in row_data) + " |")

    lines.append("")

    # Cell-level confidence breakdown
    low_conf_cells = [
        c for c in cells if c.get("confidence", 1.0) < 0.7
    ]
    if low_conf_cells:
        lines.append("<details><summary>⚠️ Low-confidence cells</summary>\n")
        lines.append("| Row | Col | Text | Confidence |")
        lines.append("|-----|-----|------|-----------|")
        for c in low_conf_cells:
            lines.append(
                f"| {c['row']} | {c['col']} | {_escape_md(c.get('text', ''))} "
                f"| {_conf_badge(c['confidence'])} {_pct(c['confidence'])} |"
            )
        lines.append("\n</details>\n")

    return "\n".join(lines)


def _render_tables_section(tables: List[Dict[str, Any]], page_num: int) -> str:
    """Render all tables for a page."""
    if not tables:
        return ""

    lines = [f"### 📊 Tables (Page {page_num})\n"]
    for i, t in enumerate(tables, 1):
        lines.append(_render_table(t, i))

    return "\n".join(lines)


def _render_raw_text(raw_text: str, page_num: int) -> str:
    """Render raw extracted text."""
    if not raw_text or not raw_text.strip():
        return ""

    lines = [
        f"### 📃 Raw Text (Page {page_num})\n",
        "```",
        raw_text.strip(),
        "```",
        "",
    ]
    return "\n".join(lines)


def _render_fields(fields: List[Dict[str, Any]]) -> str:
    """Render extracted fields (KIE results)."""
    if not fields:
        return ""

    lines = [
        "## 🏷️ Extracted Fields\n",
        "| Field | Value | Page | Position | Confidence | Status | Source |",
        "|-------|-------|------|----------|-----------|--------|--------|",
    ]

    for f in fields:
        name = f.get("name", "N/A")
        value = _escape_md(str(f.get("value", "")))
        conf = f.get("confidence", 0)
        status = f.get("status", "N/A")
        source = f.get("chosen_source", "N/A")
        status_icon = {
            "confident": "✅",
            "validated": "✅",
            "uncertain": "⚠️",
            "single_source": "ℹ️",
            "validation_failed": "❌",
        }.get(status, "❓")

        page = f.get("page", "—")
        pos = _format_bbox(f.get("bbox"))
        lines.append(
            f"| **{name}** | {value} | {page} | {pos} | {_conf_badge(conf)} {_pct(conf)} "
            f"| {status_icon} {status} | {source} |"
        )

    lines.append("")

    # Render candidates for each field (collapsible)
    fields_with_candidates = [
        f for f in fields if f.get("candidates")
    ]
    if fields_with_candidates:
        lines.append("<details><summary>🔍 All Candidates (per field)</summary>\n")
        for f in fields_with_candidates:
            name = f.get("name", "N/A")
            lines.append(f"**{name}:**\n")
            lines.append("| Source | Value | Confidence |")
            lines.append("|--------|-------|-----------|")
            for c in f["candidates"]:
                lines.append(
                    f"| {c.get('source', '?')} | "
                    f"{_escape_md(str(c.get('value', '')))} | "
                    f"{_conf_badge(c.get('confidence', 0))} "
                    f"{_pct(c.get('confidence', 0))} |"
                )
            lines.append("")
        lines.append("</details>\n")

    return "\n".join(lines)


def _render_validation(validation: Dict[str, Any]) -> str:
    """Render validation results."""
    if not validation:
        return ""

    passed = validation.get("passed", True)
    total = validation.get("total_checks", 0)
    passed_n = validation.get("passed_checks", 0)
    failed_n = validation.get("failed_checks", 0)
    issues = validation.get("issues", [])

    icon = "✅" if passed else "❌"

    lines = [
        f"## {icon} Validation Summary\n",
        f"- **Overall:** {'Passed' if passed else 'Failed'}",
        f"- **Checks:** {passed_n}/{total} passed, {failed_n} failed",
    ]

    if issues:
        lines.append("\n**Issues:**\n")
        for issue in issues:
            lines.append(f"- ⚠️ {issue}")

    details = validation.get("details", [])
    if details:
        lines.append("\n<details><summary>Detailed check results</summary>\n")
        lines.append("| Check | Result | Message |")
        lines.append("|-------|--------|---------|")
        for d in details:
            r_icon = "✅" if d.get("passed") else "❌"
            lines.append(
                f"| {d.get('name', '?')} | {r_icon} | "
                f"{_escape_md(d.get('message', ''))} |"
            )
        lines.append("\n</details>")

    lines.append("")
    return "\n".join(lines)


def _render_document_tables(tables: List[Dict[str, Any]]) -> str:
    """Render the top-level document tables section."""
    if not tables:
        return ""

    lines = ["## 📊 All Tables (Document Level)\n"]
    for i, t in enumerate(tables, 1):
        lines.append(_render_table(t, i))

    return "\n".join(lines)


def _escape_md(text: str) -> str:
    """Escape special markdown characters in table cells."""
    return (
        text.replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", "")
    )


# ── Main generator ───────────────────────────────────────────────────────

def generate_markdown(data: Dict[str, Any]) -> str:
    """
    Convert a DocVision JSON result dict into a structured Markdown report.

    Args:
        data: The full document JSON dict (same as written to output/*.json).

    Returns:
        Complete Markdown string ready to write to a .md file.
    """
    parts: List[str] = []

    # Title
    filename = data.get("metadata", {}).get("filename", "Document")
    parts.append(f"# 🔎 DocVision OCR Report — `{filename}`\n")
    parts.append(
        f"> Generated on {datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
    )
    parts.append("---\n")

    # Document metadata
    parts.append(_render_metadata(data))

    # Per-page sections
    pages = data.get("pages", [])
    for page in pages:
        page_num = page.get("number", "?")
        parts.append(f"---\n\n## 📑 Page {page_num}\n")

        # Page metadata
        parts.append(_render_page_metadata(page))

        # Layout regions
        parts.append(_render_layout_regions(
            page.get("layout_regions", []), page_num
        ))

        # Text lines
        parts.append(_render_text_lines(
            page.get("text_lines", []), page_num
        ))

        # Tables
        parts.append(_render_tables_section(
            page.get("tables", []), page_num
        ))

        # Raw text
        parts.append(_render_raw_text(
            page.get("raw_text", ""), page_num
        ))

    # Document-level tables (if different from per-page)
    doc_tables = data.get("tables", [])
    if doc_tables:
        parts.append("---\n")
        parts.append(_render_document_tables(doc_tables))

    # Extracted fields
    fields = data.get("fields", [])
    if fields:
        parts.append("---\n")
        parts.append(_render_fields(fields))

    # Validation
    validation = data.get("validation")
    if validation:
        parts.append("---\n")
        parts.append(_render_validation(validation))

    # Footer
    parts.append("---\n")
    parts.append(
        "*Report generated by [DocVision](https://github.com/ankitan-ai/horizon-OCR-python) "
        "— Horizon OCR Pipeline*\n"
    )

    # Filter out empty strings and join
    return "\n".join(p for p in parts if p)


# ── File-saving helper ───────────────────────────────────────────────────

def save_markdown(
    data: Dict[str, Any],
    output_dir: str,
    processing_mode: str = "local",
    filename_stem: Optional[str] = None,
    doc_id: Optional[str] = None,
) -> str:
    """
    Generate and save a Markdown report to the markdown/ directory.

    Mirrors the output/ folder structure with Local/ and Azure_Cloud/ subfolders.

    Args:
        data:             Full document JSON dict.
        output_dir:       Base markdown output directory (e.g. "markdown").
        processing_mode:  "local" or "azure".
        filename_stem:    Original filename stem (without extension).
        doc_id:           Document ID for unique filenames.

    Returns:
        Absolute path of the saved .md file.
    """
    subfolder = "Azure_Cloud" if processing_mode == "azure" else "Local"
    out_path = Path(output_dir) / subfolder
    out_path.mkdir(parents=True, exist_ok=True)

    stem = filename_stem or data.get("metadata", {}).get("filename", "document")
    stem = Path(stem).stem  # strip any extension
    suffix = f"_{doc_id}" if doc_id else ""
    md_filename = f"{stem}{suffix}.md"
    filepath = out_path / md_filename

    md_content = generate_markdown(data)
    filepath.write_text(md_content, encoding="utf-8")

    logger.info(f"Markdown report saved to: {filepath}")
    return str(filepath)
