"""Tests for DocVision types and schemas."""

import pytest
from datetime import datetime


class TestBoundingBox:
    """Tests for BoundingBox type."""
    
    def test_create_valid(self):
        from docvision.types import BoundingBox
        
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 70
    
    def test_width_height_properties(self):
        from docvision.types import BoundingBox
        
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        assert bbox.width == 100
        assert bbox.height == 50
    
    def test_area_property(self):
        from docvision.types import BoundingBox
        
        bbox = BoundingBox(x1=0, y1=0, x2=10, y2=20)
        assert bbox.area == 200
    
    def test_center_property(self):
        from docvision.types import BoundingBox
        
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        cx, cy = bbox.center
        
        assert cx == 50
        assert cy == 25
    
    def test_to_list(self):
        from docvision.types import BoundingBox
        
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)
        coords = bbox.to_list()
        
        assert coords == [10, 20, 110, 70]
    
    def test_from_list(self):
        from docvision.types import BoundingBox
        
        bbox = BoundingBox.from_list([10, 20, 110, 70])
        
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 70


class TestPolygon:
    """Tests for Polygon type."""
    
    def test_create_valid(self):
        from docvision.types import Polygon
        
        points = [(0, 0), (100, 0), (100, 50), (0, 50)]
        poly = Polygon(points=points)
        
        assert len(poly.points) == 4
    
    def test_bounding_box_property(self):
        from docvision.types import Polygon
        
        points = [(10, 20), (110, 20), (110, 70), (10, 70)]
        poly = Polygon(points=points)
        bbox = poly.bounding_box
        
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.width == 100
        assert bbox.height == 50


class TestField:
    """Tests for Field type."""
    
    def test_create_field(self, sample_field):
        assert sample_field.name == "invoice_number"
        assert sample_field.value == "INV-001"
        assert 0 <= sample_field.confidence <= 1
    
    def test_field_status(self):
        from docvision.types import Field, FieldStatus
        
        # Field status must be set explicitly; it defaults to UNCERTAIN
        high_conf = Field(name="test", value="x", confidence=0.95, status=FieldStatus.CONFIDENT)
        assert high_conf.status == FieldStatus.CONFIDENT
        
        low_conf = Field(name="test", value="x", confidence=0.25)
        assert low_conf.status == FieldStatus.UNCERTAIN
    
    def test_field_with_candidates(self, sample_field):
        assert len(sample_field.candidates) == 2
        assert sample_field.candidates[0].value == "INV-001"


class TestTable:
    """Tests for Table type."""
    
    def test_create_table(self, sample_table):
        assert sample_table.rows == 2
        assert sample_table.cols == 3
        assert len(sample_table.cells) == 6
    
    def test_table_get_cell(self, sample_table):
        cell = sample_table.get_cell(0, 0)
        assert cell is not None
        assert cell.text == "Item"
        
        cell = sample_table.get_cell(1, 2)
        assert cell is not None
        assert cell.text == "$10.00"
    
    def test_table_to_list(self, sample_table):
        """Test converting table to 2D list using row helpers."""
        # Build a 2D list from the table using get_row_texts
        data = [sample_table.get_row_texts(i) for i in range(sample_table.rows)]
        
        assert len(data) == 2
        assert data[0] == ["Item", "Qty", "Price"]
        assert data[1] == ["Widget", "5", "$10.00"]


class TestDocument:
    """Tests for Document type."""
    
    def test_create_document(self, sample_field, sample_table):
        from docvision.types import Document, DocumentMetadata, Page, PageMetadata
        
        page = Page(
            number=1,
            metadata=PageMetadata(width=612, height=792, dpi=300),
            text_lines=[],
            tables=[sample_table],
            raw_text="Test document"
        )
        
        doc = Document(
            id="doc-001",
            metadata=DocumentMetadata(
                filename="test.pdf",
                file_type="pdf",
                file_size_bytes=1024
            ),
            page_count=1,
            pages=[page],
            fields=[sample_field],
            tables=[sample_table]
        )
        
        assert doc.id == "doc-001"
        assert doc.page_count == 1
        assert len(doc.fields) == 1
        assert len(doc.tables) == 1
    
    def test_document_serialization(self, sample_field, sample_table):
        from docvision.types import Document, DocumentMetadata, Page, PageMetadata
        
        page = Page(
            number=1,
            metadata=PageMetadata(width=612, height=792, dpi=300),
            text_lines=[],
            raw_text="Test"
        )
        
        doc = Document(
            id="doc-001",
            metadata=DocumentMetadata(
                filename="test.pdf",
                file_type="pdf",
                file_size_bytes=1024
            ),
            page_count=1,
            pages=[page],
            fields=[sample_field]
        )
        
        # Test JSON serialization
        json_dict = doc.model_dump(mode="json")
        
        assert json_dict["id"] == "doc-001"
        assert json_dict["page_count"] == 1
        assert len(json_dict["fields"]) == 1


class TestProcessingResult:
    """Tests for ProcessingResult type."""
    
    def test_success_result(self, sample_field):
        from docvision.types import ProcessingResult, Document, DocumentMetadata
        
        doc = Document(
            id="doc-001",
            metadata=DocumentMetadata(filename="test.pdf", file_type="pdf", file_size_bytes=1024),
            page_count=1,
            pages=[]
        )
        
        result = ProcessingResult(success=True, document=doc)
        
        assert result.success is True
        assert result.document is not None
        assert result.error is None
    
    def test_failure_result(self):
        from docvision.types import ProcessingResult
        
        result = ProcessingResult(
            success=False,
            error="File not found"
        )
        
        assert result.success is False
        assert result.document is None
        assert result.error == "File not found"
