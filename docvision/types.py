"""
Pydantic types and schemas for DocVision.

Defines the complete JSON output schema with support for:
- Per-page details with layout regions
- Table structures with cells
- Dynamic field extraction with all candidates
- Validation results and confidence trails
"""

from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field as PydanticField
import uuid


class BoundingBox(BaseModel):
    """Bounding box coordinates (x1, y1, x2, y2) in pixels."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])


class Polygon(BaseModel):
    """Polygon defined by list of (x, y) points."""
    points: List[tuple[float, float]]
    
    @property
    def bounding_box(self) -> BoundingBox:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return BoundingBox(
            x1=min(xs), y1=min(ys),
            x2=max(xs), y2=max(ys)
        )


class ContentType(str, Enum):
    """Type of content detected in a region."""
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LayoutRegionType(str, Enum):
    """Type of layout region (DocLayNet classes)."""
    HEADER = "header"
    FOOTER = "footer"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    LOGO = "logo"
    TITLE = "title"
    LIST = "list"
    CAPTION = "caption"
    PAGE_NUMBER = "page_number"
    SIGNATURE = "signature"
    STAMP = "stamp"
    UNKNOWN = "unknown"


class FieldStatus(str, Enum):
    """Status of extracted field confidence."""
    CONFIDENT = "confident"  # High confidence from multiple sources
    UNCERTAIN = "uncertain"  # Low confidence, all candidates included
    SINGLE_SOURCE = "single_source"  # Only one source had this field
    VALIDATED = "validated"  # Passed validator checks
    VALIDATION_FAILED = "validation_failed"  # Failed validator checks


class SourceEngine(str, Enum):
    """Source engine that produced the extraction."""
    DONUT = "donut"
    LAYOUTLMV3 = "layoutlmv3"
    TROCR = "trocr"
    TESSERACT = "tesseract"
    PPSTRUCTURE = "ppstructure"
    ENSEMBLE = "ensemble"
    VALIDATOR = "validator"
    AZURE_DOC_INTELLIGENCE = "azure_doc_intelligence"
    GPT_VISION = "gpt_vision"


class Word(BaseModel):
    """Individual word with position and confidence."""
    text: str
    bbox: BoundingBox
    confidence: float = PydanticField(ge=0.0, le=1.0)
    source: SourceEngine = SourceEngine.TROCR
    content_type: ContentType = ContentType.UNKNOWN


class TextLine(BaseModel):
    """Text line containing words."""
    id: str = PydanticField(default_factory=lambda: str(uuid.uuid4())[:8])
    text: str
    words: List[Word] = PydanticField(default_factory=list)
    polygon: Optional[Polygon] = None
    bbox: BoundingBox
    confidence: float = PydanticField(ge=0.0, le=1.0)
    source: SourceEngine = SourceEngine.TROCR
    content_type: ContentType = ContentType.UNKNOWN


class LayoutRegion(BaseModel):
    """Layout region detected on a page."""
    id: str = PydanticField(default_factory=lambda: str(uuid.uuid4())[:8])
    type: LayoutRegionType
    bbox: BoundingBox
    confidence: float = PydanticField(ge=0.0, le=1.0)
    text_lines: List[TextLine] = PydanticField(default_factory=list)
    content_type: ContentType = ContentType.UNKNOWN


class Cell(BaseModel):
    """Table cell with position and content."""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: Optional[BoundingBox] = None
    confidence: float = PydanticField(ge=0.0, le=1.0, default=0.0)
    source: SourceEngine = SourceEngine.TROCR
    is_header: bool = False


class Table(BaseModel):
    """Table structure with cells."""
    id: str = PydanticField(default_factory=lambda: str(uuid.uuid4())[:8])
    page: int
    bbox: BoundingBox
    rows: int
    cols: int
    cells: List[Cell] = PydanticField(default_factory=list)
    confidence: float = PydanticField(ge=0.0, le=1.0)
    has_borders: bool = True
    
    def get_cell(self, row: int, col: int) -> Optional[Cell]:
        """Get cell at specific position."""
        for cell in self.cells:
            if cell.row == row and cell.col == col:
                return cell
        return None
    
    def get_row_texts(self, row: int) -> List[str]:
        """Get all text values in a row."""
        return [c.text for c in self.cells if c.row == row]
    
    def get_col_texts(self, col: int) -> List[str]:
        """Get all text values in a column."""
        return [c.text for c in self.cells if c.col == col]


class Candidate(BaseModel):
    """Candidate value from a specific source."""
    source: SourceEngine
    value: Any
    confidence: float = PydanticField(ge=0.0, le=1.0)
    bbox: Optional[BoundingBox] = None
    page: Optional[int] = None
    raw_value: Optional[str] = None  # Original value before normalization
    metadata: Dict[str, Any] = PydanticField(default_factory=dict)


class ValidatorResult(BaseModel):
    """Result of a validator check."""
    name: str  # e.g., "regex", "date_format", "currency", "total_check"
    passed: bool
    message: Optional[str] = None
    details: Dict[str, Any] = PydanticField(default_factory=dict)


class Field(BaseModel):
    """Extracted field with all candidates and validation."""
    name: str
    value: Optional[Any] = None
    normalized_value: Optional[Any] = None  # e.g., date -> ISO format
    data_type: str = "string"  # string, number, date, currency, etc.
    confidence: float = PydanticField(ge=0.0, le=1.0, default=0.0)
    status: FieldStatus = FieldStatus.UNCERTAIN
    page: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    chosen_source: Optional[SourceEngine] = None
    candidates: List[Candidate] = PydanticField(default_factory=list)
    validators: List[ValidatorResult] = PydanticField(default_factory=list)
    
    @property
    def is_confident(self) -> bool:
        return self.status in [FieldStatus.CONFIDENT, FieldStatus.VALIDATED]
    
    @property
    def validation_passed(self) -> bool:
        return all(v.passed for v in self.validators)


class PageMetadata(BaseModel):
    """Metadata about a page."""
    width: int
    height: int
    dpi: int = 350
    content_type: ContentType = ContentType.UNKNOWN
    readability: Literal["good", "fair", "poor"] = "good"
    readability_issues: List[str] = PydanticField(default_factory=list)


class Page(BaseModel):
    """Single page with all extracted content."""
    number: int  # 1-indexed
    metadata: PageMetadata
    layout_regions: List[LayoutRegion] = PydanticField(default_factory=list)
    text_lines: List[TextLine] = PydanticField(default_factory=list)
    tables: List[Table] = PydanticField(default_factory=list)
    raw_text: str = ""  # Full page text (reading order)
    
    @property
    def word_count(self) -> int:
        return len(self.raw_text.split())


class ValidationResult(BaseModel):
    """Overall document validation result."""
    passed: bool = True
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    issues: List[str] = PydanticField(default_factory=list)
    details: List[ValidatorResult] = PydanticField(default_factory=list)


class DocumentMetadata(BaseModel):
    """Document-level metadata."""
    filename: str
    file_type: str  # pdf, image
    file_size_bytes: int = 0
    processed_at: datetime = PydanticField(default_factory=lambda: datetime.now(ZoneInfo("America/New_York")))
    processing_time_seconds: float = 0.0
    docvision_version: str = "0.1.0"


class Document(BaseModel):
    """Complete document extraction result."""
    id: str = PydanticField(default_factory=lambda: str(uuid.uuid4()))
    metadata: DocumentMetadata
    page_count: int
    pages: List[Page] = PydanticField(default_factory=list)
    tables: List[Table] = PydanticField(default_factory=list)
    fields: List[Field] = PydanticField(default_factory=list)
    validation: ValidationResult = PydanticField(default_factory=ValidationResult)
    
    def get_field(self, name: str) -> Optional[Field]:
        """Get field by name."""
        for field in self.fields:
            if field.name.lower() == name.lower():
                return field
        return None
    
    def get_fields_by_status(self, status: FieldStatus) -> List[Field]:
        """Get all fields with specific status."""
        return [f for f in self.fields if f.status == status]
    
    def get_confident_fields(self) -> List[Field]:
        """Get all confident fields."""
        return [f for f in self.fields if f.is_confident]
    
    def get_uncertain_fields(self) -> List[Field]:
        """Get all uncertain fields requiring review."""
        return [f for f in self.fields if f.status == FieldStatus.UNCERTAIN]


class ProcessingResult(BaseModel):
    """Result of document processing."""
    success: bool
    document: Optional[Document] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    warnings: List[str] = PydanticField(default_factory=list)


# Type aliases for convenience
BBox = BoundingBox
Poly = Polygon
