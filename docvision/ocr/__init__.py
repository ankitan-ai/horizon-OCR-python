"""OCR modules for DocVision."""

from docvision.ocr.trocr import TrOCRRecognizer
from docvision.ocr.tesseract import TesseractRecognizer
from docvision.ocr.crops import crop_text_region, normalize_crop, batch_crop_regions

__all__ = [
    "TrOCRRecognizer",
    "TesseractRecognizer",
    "crop_text_region",
    "normalize_crop",
    "batch_crop_regions",
]
