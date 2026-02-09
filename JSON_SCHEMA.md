# DocVision JSON Output Schema

This document describes the JSON output schema for DocVision document processing results.

## Top-Level Structure

```json
{
  "id": "string",
  "metadata": { ... },
  "page_count": "integer",
  "pages": [ ... ],
  "tables": [ ... ],
  "fields": [ ... ],
  "validation": { ... }
}
```

## Document

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique document identifier |
| `metadata` | DocumentMetadata | Document-level metadata |
| `page_count` | integer | Number of pages in document |
| `pages` | Page[] | Array of page objects |
| `tables` | Table[] | All tables extracted from document |
| `fields` | Field[] | All extracted key-value fields |
| `validation` | ValidationResult | Overall validation summary |

## DocumentMetadata

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Original filename |
| `file_type` | string | File type: "pdf" or "image" |
| `file_size_bytes` | integer | File size in bytes |
| `processed_at` | datetime | Processing timestamp (ISO 8601) |
| `processing_time_seconds` | float | Total processing time |

## Page

| Field | Type | Description |
|-------|------|-------------|
| `number` | integer | Page number (1-indexed) |
| `metadata` | PageMetadata | Page-level metadata |
| `layout_regions` | LayoutRegion[] | Detected layout regions |
| `text_lines` | TextLine[] | Detected text lines with OCR |
| `tables` | Table[] | Tables on this page |
| `raw_text` | string | Concatenated raw text |

## PageMetadata

| Field | Type | Description |
|-------|------|-------------|
| `width` | integer | Page width in pixels |
| `height` | integer | Page height in pixels |
| `dpi` | integer | Rendering DPI |
| `content_type` | string | "printed", "handwritten", "mixed", or "unknown" |
| `readability` | string | "good", "medium", or "poor" |
| `readability_issues` | string[] | List of detected issues |

## Field

The core extraction unit containing key-value pairs with confidence and validation.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Field name/key (e.g., "invoice_number", "total") |
| `value` | any | Extracted value |
| `confidence` | float | Confidence score (0.0-1.0) |
| `status` | FieldStatus | Status flag |
| `candidates` | Candidate[] | All candidates from different sources |
| `source` | string | Primary source engine |
| `page` | integer | Page number where field was found |
| `bbox` | BoundingBox | Bounding box (if available) |
| `validators` | ValidatorResult[] | Validation results |

### FieldStatus Values

| Status | Description |
|--------|-------------|
| `confident` | High confidence (≥0.8), single clear value |
| `uncertain` | Low confidence (<0.3) or conflicting candidates |
| `validated` | Passed all applicable validators |
| `validation_failed` | Failed one or more validators |
| `manual_review` | Flagged for human review |

### Candidate

| Field | Type | Description |
|-------|------|-------------|
| `value` | any | Candidate value |
| `confidence` | float | Confidence score |
| `source` | string | Source engine ("donut", "layoutlmv3", "trocr", "tesseract") |
| `page` | integer | Page number |
| `bbox` | BoundingBox | Location (optional) |

## Table

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique table identifier |
| `bbox` | BoundingBox | Table bounding box |
| `rows` | integer | Number of rows |
| `cols` | integer | Number of columns |
| `cells` | Cell[] | Array of cells |
| `confidence` | float | Detection confidence |
| `page` | integer | Page number |

### Cell

| Field | Type | Description |
|-------|------|-------------|
| `row` | integer | Row index (0-based) |
| `col` | integer | Column index (0-based) |
| `row_span` | integer | Row span (default: 1) |
| `col_span` | integer | Column span (default: 1) |
| `text` | string | Cell text content |
| `bbox` | BoundingBox | Cell bounding box |
| `confidence` | float | OCR confidence |
| `is_header` | boolean | Whether cell is a header |

## TextLine

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique line identifier |
| `bbox` | BoundingBox | Line bounding box |
| `polygon` | Polygon | Precise polygon boundary |
| `text` | string | Recognized text |
| `confidence` | float | OCR confidence |
| `source` | string | OCR engine used |
| `words` | Word[] | Individual words (optional) |

## LayoutRegion

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique region identifier |
| `type` | LayoutRegionType | Region type |
| `bbox` | BoundingBox | Region bounding box |
| `confidence` | float | Detection confidence |

### LayoutRegionType Values

- `text` - Text block/paragraph
- `title` - Title/heading
- `list` - List item
- `table` - Table region
- `figure` - Image/figure
- `caption` - Figure/table caption
- `header` - Page header
- `footer` - Page footer
- `page_number` - Page number
- `other` - Other/unknown

## BoundingBox

| Field | Type | Description |
|-------|------|-------------|
| `x` | integer | Left coordinate |
| `y` | integer | Top coordinate |
| `width` | integer | Box width |
| `height` | integer | Box height |

## Polygon

| Field | Type | Description |
|-------|------|-------------|
| `points` | [int, int][] | Array of [x, y] coordinate pairs |

## ValidationResult

| Field | Type | Description |
|-------|------|-------------|
| `passed` | boolean | Whether all checks passed |
| `total_checks` | integer | Total validation checks run |
| `passed_checks` | integer | Number of passed checks |
| `failed_checks` | integer | Number of failed checks |
| `issues` | string[] | List of issue descriptions |
| `details` | ValidatorResult[] | Detailed results per check |

### ValidatorResult

| Field | Type | Description |
|-------|------|-------------|
| `validator_name` | string | Validator name |
| `passed` | boolean | Whether validation passed |
| `message` | string | Result message |
| `details` | object | Additional details |

## Complete Example

```json
{
  "id": "doc-a1b2c3d4",
  "metadata": {
    "filename": "invoice_001.pdf",
    "file_type": "pdf",
    "file_size_bytes": 245678,
    "processed_at": "2024-01-15T10:30:00.000Z",
    "processing_time_seconds": 2.45
  },
  "page_count": 1,
  "pages": [
    {
      "number": 1,
      "metadata": {
        "width": 2550,
        "height": 3300,
        "dpi": 300,
        "content_type": "printed",
        "readability": "good",
        "readability_issues": []
      },
      "layout_regions": [
        {
          "id": "region-001",
          "type": "title",
          "bbox": {"x": 100, "y": 50, "width": 400, "height": 60},
          "confidence": 0.95
        },
        {
          "id": "region-002",
          "type": "text",
          "bbox": {"x": 100, "y": 150, "width": 600, "height": 200},
          "confidence": 0.92
        },
        {
          "id": "region-003",
          "type": "table",
          "bbox": {"x": 100, "y": 400, "width": 700, "height": 300},
          "confidence": 0.88
        }
      ],
      "text_lines": [
        {
          "id": "line-001",
          "bbox": {"x": 100, "y": 50, "width": 200, "height": 30},
          "polygon": {"points": [[100, 50], [300, 50], [300, 80], [100, 80]]},
          "text": "INVOICE",
          "confidence": 0.98,
          "source": "trocr"
        },
        {
          "id": "line-002",
          "bbox": {"x": 100, "y": 150, "width": 300, "height": 25},
          "text": "Invoice Number: INV-2024-001",
          "confidence": 0.95,
          "source": "trocr"
        }
      ],
      "tables": [],
      "raw_text": "INVOICE\nInvoice Number: INV-2024-001\n..."
    }
  ],
  "tables": [
    {
      "id": "table-001",
      "bbox": {"x": 100, "y": 400, "width": 700, "height": 300},
      "rows": 4,
      "cols": 4,
      "cells": [
        {"row": 0, "col": 0, "text": "Item", "is_header": true, "confidence": 0.95},
        {"row": 0, "col": 1, "text": "Description", "is_header": true, "confidence": 0.94},
        {"row": 0, "col": 2, "text": "Qty", "is_header": true, "confidence": 0.96},
        {"row": 0, "col": 3, "text": "Price", "is_header": true, "confidence": 0.95},
        {"row": 1, "col": 0, "text": "1", "confidence": 0.92},
        {"row": 1, "col": 1, "text": "Widget Pro", "confidence": 0.88},
        {"row": 1, "col": 2, "text": "5", "confidence": 0.94},
        {"row": 1, "col": 3, "text": "$99.00", "confidence": 0.91}
      ],
      "confidence": 0.88,
      "page": 1
    }
  ],
  "fields": [
    {
      "name": "invoice_number",
      "value": "INV-2024-001",
      "confidence": 0.95,
      "status": "confident",
      "candidates": [
        {"value": "INV-2024-001", "confidence": 0.95, "source": "donut"},
        {"value": "INV-2024-001", "confidence": 0.92, "source": "layoutlmv3"}
      ],
      "source": "donut",
      "page": 1,
      "validators": [
        {"validator_name": "non_empty", "passed": true, "message": "Value is not empty"}
      ]
    },
    {
      "name": "invoice_date",
      "value": "2024-01-15",
      "confidence": 0.88,
      "status": "validated",
      "candidates": [
        {"value": "2024-01-15", "confidence": 0.88, "source": "donut"},
        {"value": "January 15, 2024", "confidence": 0.82, "source": "layoutlmv3"}
      ],
      "source": "donut",
      "page": 1,
      "validators": [
        {"validator_name": "date", "passed": true, "message": "Valid date format"}
      ]
    },
    {
      "name": "total_amount",
      "value": "$495.00",
      "confidence": 0.42,
      "status": "uncertain",
      "candidates": [
        {"value": "$495.00", "confidence": 0.42, "source": "donut"},
        {"value": "$495", "confidence": 0.38, "source": "layoutlmv3"},
        {"value": "495.00", "confidence": 0.35, "source": "trocr"}
      ],
      "source": "donut",
      "page": 1,
      "validators": [
        {"validator_name": "amount", "passed": true, "message": "Valid currency amount"}
      ]
    }
  ],
  "validation": {
    "passed": true,
    "total_checks": 4,
    "passed_checks": 4,
    "failed_checks": 0,
    "issues": [],
    "details": [
      {"validator_name": "non_empty", "passed": true},
      {"validator_name": "date", "passed": true},
      {"validator_name": "amount", "passed": true},
      {"validator_name": "consistency", "passed": true}
    ]
  }
}
```

## Field Status Decision Logic

```
if confidence >= 0.8 and no_conflicts:
    status = "confident"
elif confidence < 0.3 or has_conflicts:
    status = "uncertain"
    # All candidates preserved for review

if validators_run:
    if all_validators_passed and status == "confident":
        status = "validated"
    elif any_validator_failed:
        status = "validation_failed"
```

## Source Engine Values

| Value | Description |
|-------|-------------|
| `donut` | Donut OCR-free extraction |
| `layoutlmv3` | LayoutLMv3 token-based KIE |
| `trocr` | TrOCR text recognition |
| `tesseract` | Tesseract OCR backup |
| `fusion` | Rank-and-fuse combined result |
