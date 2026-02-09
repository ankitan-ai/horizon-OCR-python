"""Tests for Azure Document Intelligence provider."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np

from docvision.config import AzureConfig, ProcessingMode
from docvision.types import (
    BoundingBox,
    TextLine,
    Table,
    Cell,
    LayoutRegion,
    LayoutRegionType,
    SourceEngine,
    ContentType,
)
from docvision.azure.doc_intelligence import AzureDocIntelligenceProvider, _ROLE_MAP


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def azure_config():
    """AzureConfig with fake credentials (never hits the network)."""
    return AzureConfig(
        processing_mode=ProcessingMode.AZURE,
        doc_intelligence_endpoint="https://fake.cognitiveservices.azure.com/",
        doc_intelligence_key="fake-key-000",
        doc_intelligence_model="prebuilt-layout",
    )


@pytest.fixture
def provider(azure_config):
    """Provider instance with patched client (no real HTTP calls)."""
    return AzureDocIntelligenceProvider(azure_config)


@pytest.fixture
def dummy_image():
    """A small 100×200 BGR numpy image."""
    return np.zeros((100, 200, 3), dtype=np.uint8)


# ── Helper: mock Azure SDK objects ───────────────────────────────────────────

def _make_span(offset: int, length: int):
    s = MagicMock()
    s.offset = offset
    s.length = length
    return s


def _make_word(text: str, confidence: float, offset: int, polygon=None):
    w = MagicMock()
    w.content = text
    w.confidence = confidence
    w.span = _make_span(offset, len(text))
    w.polygon = polygon or [0, 0, 10, 0, 10, 10, 0, 10]
    return w


def _make_line(text: str, offset: int, polygon=None):
    ln = MagicMock()
    ln.content = text
    ln.spans = [_make_span(offset, len(text))]
    ln.polygon = polygon or [0, 0, 50, 0, 50, 12, 0, 12]
    return ln


def _make_bounding_region(page_number: int = 1, polygon=None):
    br = MagicMock()
    br.page_number = page_number
    br.polygon = polygon or [10, 10, 90, 10, 90, 30, 10, 30]
    return br


def _make_table_cell(row, col, text, kind="content", row_span=1, col_span=1):
    c = MagicMock()
    c.row_index = row
    c.column_index = col
    c.content = text
    c.kind = kind
    c.row_span = row_span
    c.column_span = col_span
    c.bounding_regions = [_make_bounding_region()]
    return c


def _make_table(rows, cols, cells, page_number=1):
    t = MagicMock()
    t.row_count = rows
    t.column_count = cols
    t.cells = cells
    t.bounding_regions = [_make_bounding_region(page_number)]
    return t


def _make_paragraph(text, role=None, page_number=1):
    p = MagicMock()
    p.content = text
    p.role = role
    p.bounding_regions = [_make_bounding_region(page_number)]
    return p


def _make_azure_page(lines=None, words=None, width=200, height=100):
    page = MagicMock()
    page.lines = lines or []
    page.words = words or []
    page.width = width
    page.height = height
    return page


# ── Tests: initialisation ───────────────────────────────────────────────────

class TestProviderInit:
    """Tests for provider construction and validation."""

    def test_init_success(self, azure_config):
        provider = AzureDocIntelligenceProvider(azure_config)
        assert provider._config is azure_config
        assert provider._client is None  # lazy

    def test_init_raises_without_credentials(self, monkeypatch):
        # Clear env vars so load_dotenv values are ignored
        monkeypatch.delenv("AZURE_DOC_INTELLIGENCE_KEY", raising=False)
        monkeypatch.delenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        # Prevent load_dotenv from re-populating env vars from .env file
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
        bad_config = AzureConfig()  # no endpoint / key
        with pytest.raises(ValueError, match="not configured"):
            AzureDocIntelligenceProvider(bad_config)

    def test_lazy_client_not_created_at_init(self, provider):
        assert provider._client is None


# ── Tests: image encoding ───────────────────────────────────────────────────

class TestImageEncoding:
    """Tests for _encode_image static method."""

    def test_encode_returns_bytes(self, dummy_image):
        result = AzureDocIntelligenceProvider._encode_image(dummy_image)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_png_header(self, dummy_image):
        result = AzureDocIntelligenceProvider._encode_image(dummy_image)
        # PNG magic bytes
        assert result[:4] == b"\x89PNG"


# ── Tests: text-line mapping ────────────────────────────────────────────────

class TestTextLineMapping:
    """Tests for _map_text_lines."""

    def test_empty_page_returns_empty(self, provider):
        result = provider._map_text_lines(None, 200, 100)
        assert result == []

    def test_single_line_no_words(self, provider):
        line = _make_line("Hello World", offset=0)
        page = _make_azure_page(lines=[line], words=[])

        result = provider._map_text_lines(page, 200, 100)

        assert len(result) == 1
        assert result[0].text == "Hello World"
        assert result[0].source == SourceEngine.AZURE_DOC_INTELLIGENCE
        assert result[0].content_type == ContentType.PRINTED

    def test_line_with_words(self, provider):
        w1 = _make_word("Hello", 0.98, offset=0)
        w2 = _make_word("World", 0.92, offset=6)
        line = _make_line("Hello World", offset=0)
        page = _make_azure_page(lines=[line], words=[w1, w2])

        result = provider._map_text_lines(page, 200, 100)

        assert len(result) == 1
        assert len(result[0].words) == 2
        assert result[0].words[0].text == "Hello"
        assert result[0].words[1].text == "World"
        # Confidence = average of word confidences
        assert abs(result[0].confidence - 0.95) < 0.01

    def test_multiple_lines(self, provider):
        lines = [
            _make_line("Line 1", offset=0),
            _make_line("Line 2", offset=7),
        ]
        page = _make_azure_page(lines=lines, words=[])

        result = provider._map_text_lines(page, 200, 100)
        assert len(result) == 2

    def test_confidence_clamped(self, provider):
        w = _make_word("test", 1.5, offset=0)  # over 1.0
        line = _make_line("test", offset=0)
        page = _make_azure_page(lines=[line], words=[w])

        result = provider._map_text_lines(page, 200, 100)
        assert result[0].confidence <= 1.0

    def test_polygon_set_on_line(self, provider):
        poly = [5, 10, 95, 10, 95, 25, 5, 25]
        line = _make_line("test", offset=0, polygon=poly)
        page = _make_azure_page(lines=[line], words=[])

        result = provider._map_text_lines(page, 200, 100)
        assert result[0].polygon is not None
        assert len(result[0].polygon.points) == 4
        assert result[0].bbox.x1 == 5
        assert result[0].bbox.y2 == 25


# ── Tests: table mapping ────────────────────────────────────────────────────

class TestTableMapping:
    """Tests for _map_tables."""

    def test_empty_tables(self, provider):
        assert provider._map_tables(None, 1, 200, 100) == []
        assert provider._map_tables([], 1, 200, 100) == []

    def test_simple_2x2_table(self, provider):
        cells = [
            _make_table_cell(0, 0, "A", kind="columnHeader"),
            _make_table_cell(0, 1, "B", kind="columnHeader"),
            _make_table_cell(1, 0, "1"),
            _make_table_cell(1, 1, "2"),
        ]
        table = _make_table(2, 2, cells)

        result = provider._map_tables([table], page_num=1, page_w=200, page_h=100)

        assert len(result) == 1
        t = result[0]
        assert t.rows == 2
        assert t.cols == 2
        assert t.page == 1
        assert len(t.cells) == 4
        # Headers flagged correctly
        assert t.cells[0].is_header is True
        assert t.cells[2].is_header is False

    def test_table_cell_span(self, provider):
        cells = [
            _make_table_cell(0, 0, "Merged", row_span=2, col_span=2),
        ]
        table = _make_table(2, 2, cells)

        result = provider._map_tables([table], 1, 200, 100)
        assert result[0].cells[0].row_span == 2
        assert result[0].cells[0].col_span == 2

    def test_source_engine_set(self, provider):
        cells = [_make_table_cell(0, 0, "X")]
        table = _make_table(1, 1, cells)

        result = provider._map_tables([table], 1, 200, 100)
        assert result[0].cells[0].source == SourceEngine.AZURE_DOC_INTELLIGENCE


# ── Tests: layout-region mapping ─────────────────────────────────────────────

class TestLayoutRegionMapping:
    """Tests for _map_layout_regions."""

    def test_empty_paragraphs(self, provider):
        assert provider._map_layout_regions(None, 200, 100) == []
        assert provider._map_layout_regions([], 200, 100) == []

    def test_plain_paragraph_maps_to_text(self, provider):
        para = _make_paragraph("Some body text", role=None)

        result = provider._map_layout_regions([para], 200, 100)

        assert len(result) == 1
        assert result[0].type == LayoutRegionType.TEXT
        assert result[0].text_lines[0].text == "Some body text"

    def test_title_role(self, provider):
        para = _make_paragraph("Document Title", role="title")
        result = provider._map_layout_regions([para], 200, 100)
        assert result[0].type == LayoutRegionType.TITLE

    def test_header_role(self, provider):
        para = _make_paragraph("Header text", role="pageHeader")
        result = provider._map_layout_regions([para], 200, 100)
        assert result[0].type == LayoutRegionType.HEADER

    def test_footer_role(self, provider):
        para = _make_paragraph("Page 1", role="pageFooter")
        result = provider._map_layout_regions([para], 200, 100)
        assert result[0].type == LayoutRegionType.FOOTER

    def test_page_number_role(self, provider):
        para = _make_paragraph("42", role="pageNumber")
        result = provider._map_layout_regions([para], 200, 100)
        assert result[0].type == LayoutRegionType.PAGE_NUMBER

    def test_section_heading_role(self, provider):
        para = _make_paragraph("Section 1", role="sectionHeading")
        result = provider._map_layout_regions([para], 200, 100)
        assert result[0].type == LayoutRegionType.TITLE

    def test_source_engine_set(self, provider):
        para = _make_paragraph("Test")
        result = provider._map_layout_regions([para], 200, 100)
        assert result[0].text_lines[0].source == SourceEngine.AZURE_DOC_INTELLIGENCE


# ── Tests: role map completeness ─────────────────────────────────────────────

class TestRoleMap:
    """Ensure all mapped roles produce valid LayoutRegionType values."""

    def test_all_roles_are_valid(self):
        for role, region_type in _ROLE_MAP.items():
            assert isinstance(region_type, LayoutRegionType), f"Bad type for role '{role}'"


# ── Tests: helper methods ───────────────────────────────────────────────────

class TestHelpers:
    """Tests for static helper methods."""

    def test_polygon_from_flat_valid(self):
        poly = AzureDocIntelligenceProvider._polygon_from_flat(
            [0, 0, 100, 0, 100, 50, 0, 50], 200, 100
        )
        assert poly is not None
        assert len(poly.points) == 4
        assert poly.bounding_box.x1 == 0
        assert poly.bounding_box.x2 == 100

    def test_polygon_from_flat_none(self):
        assert AzureDocIntelligenceProvider._polygon_from_flat(None, 200, 100) is None

    def test_polygon_from_flat_too_short(self):
        assert AzureDocIntelligenceProvider._polygon_from_flat([0, 0], 200, 100) is None

    def test_bbox_from_regions_valid(self):
        regions = [_make_bounding_region(polygon=[10, 20, 90, 20, 90, 80, 10, 80])]
        bbox = AzureDocIntelligenceProvider._bbox_from_regions(regions, 200, 100)
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 90
        assert bbox.y2 == 80

    def test_bbox_from_regions_none(self):
        bbox = AzureDocIntelligenceProvider._bbox_from_regions(None, 200, 100)
        # Fallback bbox
        assert bbox.x1 == 0 and bbox.y2 == 1

    def test_tables_for_page_filter(self):
        t1 = _make_table(1, 1, [], page_number=1)
        t2 = _make_table(1, 1, [], page_number=2)

        result = AzureDocIntelligenceProvider._tables_for_page([t1, t2], 1)
        assert len(result) == 1

    def test_paragraphs_for_page_filter(self):
        p1 = _make_paragraph("Page 1 text", page_number=1)
        p2 = _make_paragraph("Page 2 text", page_number=2)

        result = AzureDocIntelligenceProvider._paragraphs_for_page([p1, p2], 2)
        assert len(result) == 1
        assert result[0].content == "Page 2 text"


# ── Tests: full analyze (mocked SDK call) ───────────────────────────────────

class TestAnalyze:
    """End-to-end test of analyze() with a mocked Azure response."""

    def test_analyze_returns_all_keys(self, provider, dummy_image):
        # Build a mock Azure response
        w1 = _make_word("Invoice", 0.99, offset=0)
        w2 = _make_word("#12345", 0.97, offset=8)
        line = _make_line("Invoice #12345", offset=0)
        page = _make_azure_page(lines=[line], words=[w1, w2])

        cells = [
            _make_table_cell(0, 0, "Item"),
            _make_table_cell(0, 1, "Qty"),
            _make_table_cell(1, 0, "Widget"),
            _make_table_cell(1, 1, "5"),
        ]
        table = _make_table(2, 2, cells, page_number=1)
        para = _make_paragraph("Invoice #12345", role="title")

        mock_result = MagicMock()
        mock_result.pages = [page]
        mock_result.tables = [table]
        mock_result.paragraphs = [para]
        mock_result.content = "Invoice #12345\n\nItem\tQty\nWidget\t5"

        mock_poller = MagicMock()
        mock_poller.result.return_value = mock_result

        with patch.object(
            type(provider), "client", new_callable=PropertyMock
        ) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.begin_analyze_document.return_value = mock_poller
            mock_client_prop.return_value = mock_client

            result = provider.analyze(dummy_image, page_num=1)

        assert "text_lines" in result
        assert "tables" in result
        assert "layout_regions" in result
        assert "raw_text" in result

        assert len(result["text_lines"]) == 1
        assert result["text_lines"][0].text == "Invoice #12345"

        assert len(result["tables"]) == 1
        assert result["tables"][0].rows == 2

        assert len(result["layout_regions"]) == 1
        assert result["layout_regions"][0].type == LayoutRegionType.TITLE

        assert "Invoice" in result["raw_text"]
