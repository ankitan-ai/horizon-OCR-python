"""
Pytest configuration and fixtures for DocVision tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Fixtures - Sample Data
# ============================================================================

@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a simple test image with text-like patterns."""
    # Create 800x600 white image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add some black rectangles to simulate text blocks
    # Header area
    img[50:70, 100:700] = 0
    
    # Paragraph lines
    for y in range(150, 350, 25):
        img[y:y+10, 100:650] = 0
    
    # Table area
    for y in range(400, 550, 30):
        img[y:y+2, 100:700] = 0  # Horizontal lines
    for x in range(100, 701, 150):
        img[400:550, x:x+2] = 0  # Vertical lines
    
    return img


@pytest.fixture
def sample_grayscale_image(sample_image) -> np.ndarray:
    """Convert sample image to grayscale."""
    import cv2
    return cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)


@pytest.fixture
def sample_noisy_image(sample_image) -> np.ndarray:
    """Add noise to sample image."""
    noise = np.random.normal(0, 25, sample_image.shape).astype(np.int16)
    noisy = np.clip(sample_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


@pytest.fixture
def sample_skewed_image(sample_image) -> np.ndarray:
    """Rotate sample image slightly to simulate skew."""
    import cv2
    h, w = sample_image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, 5, 1.0)  # 5 degree rotation
    return cv2.warpAffine(sample_image, matrix, (w, h), borderValue=(255, 255, 255))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir) -> Path:
    """Create a minimal test PDF."""
    try:
        import fitz  # PyMuPDF
        
        pdf_path = temp_dir / "test_document.pdf"
        doc = fitz.open()
        
        # Create a page
        page = doc.new_page(width=612, height=792)  # Letter size
        
        # Add some text
        page.insert_text((72, 72), "Test Document", fontsize=24)
        page.insert_text((72, 120), "This is a test invoice.", fontsize=12)
        page.insert_text((72, 150), "Invoice Number: INV-001", fontsize=12)
        page.insert_text((72, 180), "Date: 2024-01-15", fontsize=12)
        page.insert_text((72, 210), "Total: $1,234.56", fontsize=12)
        
        doc.save(str(pdf_path))
        doc.close()
        
        return pdf_path
    except ImportError:
        pytest.skip("PyMuPDF not installed")


@pytest.fixture
def sample_image_path(temp_dir, sample_image) -> Path:
    """Save sample image to temporary file."""
    import cv2
    
    img_path = temp_dir / "test_image.png"
    cv2.imwrite(str(img_path), sample_image)
    return img_path


# ============================================================================
# Fixtures - Configuration
# ============================================================================

@pytest.fixture
def default_config():
    """Get default configuration."""
    from docvision.config import Config
    return Config()


@pytest.fixture
def test_config(temp_dir):
    """Get test configuration with temp directories."""
    from docvision.config import Config
    
    config = Config()
    config.output.dir = str(temp_dir / "output")
    config.artifacts.dir = str(temp_dir / "artifacts")
    config.artifacts.enable = True
    return config


# ============================================================================
# Fixtures - Types
# ============================================================================

@pytest.fixture
def sample_bbox():
    """Create sample bounding box."""
    from docvision.types import BoundingBox
    return BoundingBox(x1=100, y1=50, x2=300, y2=80)


@pytest.fixture
def sample_polygon():
    """Create sample polygon."""
    from docvision.types import Polygon
    return Polygon(points=[
        (100, 50), (300, 50), (300, 80), (100, 80)
    ])


@pytest.fixture
def sample_text_line(sample_bbox, sample_polygon):
    """Create sample text line."""
    from docvision.types import TextLine, SourceEngine
    
    return TextLine(
        id="line-001",
        bbox=sample_bbox,
        polygon=sample_polygon,
        text="Invoice Number: INV-001",
        confidence=0.95,
        source=SourceEngine.TROCR
    )


@pytest.fixture
def sample_field():
    """Create sample extracted field."""
    from docvision.types import Field, FieldStatus, Candidate, SourceEngine
    
    return Field(
        name="invoice_number",
        value="INV-001",
        confidence=0.92,
        status=FieldStatus.CONFIDENT,
        candidates=[
            Candidate(
                value="INV-001",
                confidence=0.92,
                source=SourceEngine.DONUT
            ),
            Candidate(
                value="INV-001",
                confidence=0.88,
                source=SourceEngine.LAYOUTLMV3
            )
        ],
        page=1
    )


@pytest.fixture
def sample_table():
    """Create sample table."""
    from docvision.types import Table, Cell, BoundingBox
    
    cells = [
        Cell(row=0, col=0, text="Item", bbox=BoundingBox(x1=100, y1=400, x2=250, y2=430)),
        Cell(row=0, col=1, text="Qty", bbox=BoundingBox(x1=250, y1=400, x2=400, y2=430)),
        Cell(row=0, col=2, text="Price", bbox=BoundingBox(x1=400, y1=400, x2=550, y2=430)),
        Cell(row=1, col=0, text="Widget", bbox=BoundingBox(x1=100, y1=430, x2=250, y2=460)),
        Cell(row=1, col=1, text="5", bbox=BoundingBox(x1=250, y1=430, x2=400, y2=460)),
        Cell(row=1, col=2, text="$10.00", bbox=BoundingBox(x1=400, y1=430, x2=550, y2=460)),
    ]
    
    return Table(
        id="table-001",
        bbox=BoundingBox(x1=100, y1=400, x2=550, y2=460),
        rows=2,
        cols=3,
        cells=cells,
        confidence=0.85,
        page=1
    )


# ============================================================================
# Fixtures - Mocks
# ============================================================================

@pytest.fixture
def mock_model_available(monkeypatch):
    """Mock model availability checks."""
    def always_available(*args, **kwargs):
        return True
    
    # Mock common availability checks
    monkeypatch.setattr("os.path.exists", always_available)


# ============================================================================
# Test helpers
# ============================================================================

def assert_bbox_valid(bbox):
    """Assert bounding box is valid."""
    assert bbox.x >= 0
    assert bbox.y >= 0
    assert bbox.width > 0
    assert bbox.height > 0


def assert_confidence_valid(confidence):
    """Assert confidence score is valid."""
    assert 0.0 <= confidence <= 1.0


def assert_text_not_empty(text):
    """Assert text is not empty."""
    assert text is not None
    assert len(text.strip()) > 0
