"""
PDF loading and rasterization using PyMuPDF.

Converts PDF pages to high-quality images for downstream processing.
Supports configurable DPI and page selection.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Generator
import numpy as np
from loguru import logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available. PDF processing will be disabled.")


class PDFLoader:
    """
    Load and rasterize PDF documents.
    
    Uses PyMuPDF for efficient PDF to image conversion with
    configurable DPI for quality control.
    """
    
    def __init__(self, dpi: int = 350, max_pages: Optional[int] = None):
        """
        Initialize PDF loader.
        
        Args:
            dpi: Resolution for rasterization (300-400 recommended for OCR accuracy)
            max_pages: Maximum pages to process (None = all pages)
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF is required for PDF processing. Install with: pip install PyMuPDF")
        
        self.dpi = dpi
        self.max_pages = max_pages
        self._zoom = dpi / 72.0  # PDF default is 72 DPI
    
    def load(self, pdf_path: str) -> List[np.ndarray]:
        """
        Load all pages from PDF as images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of page images as numpy arrays (BGR format)
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path} at {self.dpi} DPI")
        
        images = []
        doc = fitz.open(pdf_path)
        
        try:
            page_count = len(doc)
            pages_to_process = page_count
            
            if self.max_pages is not None:
                pages_to_process = min(page_count, self.max_pages)
            
            logger.info(f"Processing {pages_to_process}/{page_count} pages")
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                image = self._render_page(page)
                images.append(image)
                logger.debug(f"Rendered page {page_num + 1}: {image.shape}")
        
        finally:
            doc.close()
        
        return images
    
    def load_lazy(self, pdf_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Lazily load pages from PDF (memory efficient for large documents).
        
        Args:
            pdf_path: Path to PDF file
            
        Yields:
            Tuple of (page_number, page_image)
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        
        try:
            page_count = len(doc)
            pages_to_process = page_count
            
            if self.max_pages is not None:
                pages_to_process = min(page_count, self.max_pages)
            
            for page_num in range(pages_to_process):
                page = doc[page_num]
                image = self._render_page(page)
                yield page_num, image
        
        finally:
            doc.close()
    
    def _render_page(self, page: "fitz.Page") -> np.ndarray:
        """
        Render a single PDF page to image.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Page image as numpy array (RGB format)
        """
        # Create transformation matrix for desired DPI
        mat = fitz.Matrix(self._zoom, self._zoom)
        
        # Render page to pixmap
        pixmap = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to numpy array
        image = np.frombuffer(pixmap.samples, dtype=np.uint8)
        image = image.reshape(pixmap.height, pixmap.width, pixmap.n)
        
        # PyMuPDF returns RGB, convert to BGR for OpenCV compatibility
        if pixmap.n == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
    
    def get_page_count(self, pdf_path: str) -> int:
        """
        Get number of pages in PDF without loading all images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages
        """
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    
    def get_metadata(self, pdf_path: str) -> dict:
        """
        Get PDF metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        metadata["page_count"] = len(doc)
        doc.close()
        return metadata
