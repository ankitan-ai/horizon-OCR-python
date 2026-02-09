"""Detection modules for DocVision."""

from docvision.detect.layout_doclaynet import LayoutDetector
from docvision.detect.text_craft import TextDetector
from docvision.detect.table_tatr import TableDetector

__all__ = [
    "LayoutDetector",
    "TextDetector",
    "TableDetector",
]
