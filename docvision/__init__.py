"""
DocVision Accuracy-First: Production-ready document AI system
for maximum field-level accuracy on diverse document types.

This package provides an ensemble approach combining:
- OCR-free KIE (Donut)
- Classical OCR pipeline (TrOCR + Tesseract)
- Token-based KIE reranker (LayoutLMv3)
- Rank-and-fuse with validators
"""

# Auto-configure SSL certificates before any network calls
from docvision.ssl_config import configure_ssl_certificates as _configure_ssl  # noqa: E402

__version__ = "0.1.0"
__author__ = "DocVision Team"

from docvision.config import Config, load_config
from docvision.types import (
    Document,
    Page,
    Field,
    Table,
    Cell,
    Candidate,
    ValidationResult,
    ProcessingResult,
)

__all__ = [
    "Config",
    "load_config",
    "Document",
    "Page",
    "Field",
    "Table",
    "Cell",
    "Candidate",
    "ValidationResult",
    "ProcessingResult",
    "__version__",
]
