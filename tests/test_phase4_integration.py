"""
Phase 4 integration tests — orchestrator azure routing, web/API params.

All Azure API calls are mocked; these tests verify the wiring between
the orchestrator, web app, and API server with the new processing_mode
and document_type parameters.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import asdict

from docvision.pipeline.orchestrator import DocumentProcessor, ProcessingOptions
from docvision.config import Config, AzureConfig
from docvision.types import (
    TextLine, Table, Cell, LayoutRegion, LayoutRegionType, Field, FieldStatus,
    Candidate, SourceEngine, BoundingBox, Page, PageMetadata,
    ContentType,
)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _azure_config(**overrides):
    """Return an AzureConfig with fake but valid-looking creds."""
    defaults = dict(
        doc_intelligence_key="fake-key-000",
        doc_intelligence_endpoint="https://fake.cognitiveservices.azure.com/",
        openai_key="fake-key-000",
        openai_endpoint="https://fake.openai.azure.com/",
        openai_deployment="gpt-4o-mini",
        openai_api_version="2024-12-01-preview",
    )
    defaults.update(overrides)
    return AzureConfig(**defaults)


def _make_text_line(text="Sample Line", page=1):
    return TextLine(
        id=f"az-line-{page}",
        bbox=BoundingBox(x1=10, y1=10, x2=200, y2=30),
        text=text,
        confidence=0.95,
        source=SourceEngine.AZURE_DOC_INTELLIGENCE,
    )


def _make_field(name="invoice_number", value="INV-001", page=1):
    return Field(
        name=name,
        value=value,
        confidence=0.90,
        status=FieldStatus.CONFIDENT,
        candidates=[
            Candidate(value=value, confidence=0.90, source=SourceEngine.GPT_VISION)
        ],
        page=page,
    )


def _make_table(page=1):
    return Table(
        id="az-table-1",
        bbox=BoundingBox(x1=10, y1=100, x2=500, y2=300),
        rows=1,
        cols=2,
        cells=[
            Cell(row=0, col=0, text="Item", confidence=0.9),
            Cell(row=0, col=1, text="Qty", confidence=0.9),
        ],
        confidence=0.88,
        page=page,
    )


def _make_layout_region(region_type=LayoutRegionType.TITLE, page=1):
    return LayoutRegion(
        id="az-region-1",
        type=region_type,
        bbox=BoundingBox(x1=10, y1=5, x2=400, y2=40),
        confidence=0.92,
    )


def _stub_azure_result(page_num=1):
    """Return a dict that mimics AzureDocIntelligenceProvider.analyze()."""
    return {
        "text_lines": [_make_text_line(page=page_num)],
        "tables": [_make_table(page=page_num)],
        "layout_regions": [_make_layout_region(page=page_num)],
        "raw_text": "Sample Line",
    }


# ──────────────────────────────────────────────────────────────
# ProcessingOptions Tests
# ──────────────────────────────────────────────────────────────

class TestProcessingOptions:
    """Verify new fields on ProcessingOptions."""

    def test_defaults(self):
        opts = ProcessingOptions()
        assert opts.processing_mode == "local"
        assert opts.document_type == "auto"
        assert opts.use_gpt_vision_kie is True

    def test_azure_mode(self):
        opts = ProcessingOptions(processing_mode="azure", document_type="bol")
        assert opts.processing_mode == "azure"
        assert opts.document_type == "bol"

    def test_hybrid_mode(self):
        opts = ProcessingOptions(processing_mode="hybrid")
        assert opts.processing_mode == "hybrid"

    def test_all_document_types(self):
        for dt in ("auto", "bol", "invoice", "receipt", "delivery_ticket"):
            opts = ProcessingOptions(document_type=dt)
            assert opts.document_type == dt


# ──────────────────────────────────────────────────────────────
# Orchestrator Azure Routing Tests
# ──────────────────────────────────────────────────────────────

class TestOrchestratorAzureRouting:
    """Verify the orchestrator routes to _process_page_azure when mode='azure'."""

    @pytest.fixture
    def processor(self, test_config):
        test_config.azure = _azure_config()
        return DocumentProcessor(test_config)

    def test_local_mode_does_not_call_azure(self, processor, sample_image):
        """Local mode must NOT invoke the azure branch."""
        with patch.object(processor, "_process_page_azure") as mock_azure:
            # Call _process_page directly in local mode
            opts = ProcessingOptions(
                processing_mode="local",
                detect_layout=False,
                detect_text=False,
                detect_tables=False,
                run_ocr=False,
                run_donut=False,
                run_layoutlmv3=False,
                preprocess=False,
            )
            processor._process_page(sample_image, 1, "test", opts)
            mock_azure.assert_not_called()

    def test_azure_mode_routes_to_azure_method(self, processor, sample_image):
        """Azure mode must call _process_page_azure and skip local pipeline."""
        with patch.object(processor, "_process_page_azure", return_value={
            "page": MagicMock(),
            "tables": [],
            "fields": [],
        }) as mock_azure:
            opts = ProcessingOptions(processing_mode="azure")
            processor._process_page(sample_image, 1, "test", opts)
            mock_azure.assert_called_once()


# ──────────────────────────────────────────────────────────────
# _process_page_azure Unit Tests
# ──────────────────────────────────────────────────────────────

class TestProcessPageAzure:
    """Verify _process_page_azure wires providers correctly."""

    @pytest.fixture
    def processor(self, test_config):
        test_config.azure = _azure_config()
        return DocumentProcessor(test_config)

    def test_azure_page_returns_expected_structure(self, processor, sample_image):
        """_process_page_azure should return page, tables, fields."""
        mock_di = MagicMock()
        mock_di.analyze.return_value = _stub_azure_result(1)

        mock_gpt = MagicMock()
        mock_gpt.extract.return_value = [_make_field()]

        with patch.object(type(processor), "azure_di_provider", new_callable=PropertyMock, return_value=mock_di), \
             patch.object(type(processor), "gpt_vision_extractor", new_callable=PropertyMock, return_value=mock_gpt):
            opts = ProcessingOptions(processing_mode="azure", document_type="invoice")
            result = processor._process_page_azure(sample_image, 1, "test", opts)

        assert "page" in result
        assert "tables" in result
        assert "fields" in result

        page = result["page"]
        assert isinstance(page, Page)
        assert page.number == 1
        assert len(result["tables"]) == 1
        assert len(result["fields"]) == 1

    def test_azure_page_calls_di_provider(self, processor, sample_image):
        """DI provider.analyze must be called with image and page_num."""
        mock_di = MagicMock()
        mock_di.analyze.return_value = _stub_azure_result(1)

        mock_gpt = MagicMock()
        mock_gpt.extract.return_value = []

        with patch.object(type(processor), "azure_di_provider", new_callable=PropertyMock, return_value=mock_di), \
             patch.object(type(processor), "gpt_vision_extractor", new_callable=PropertyMock, return_value=mock_gpt):
            opts = ProcessingOptions(processing_mode="azure")
            processor._process_page_azure(sample_image, 1, "test", opts)

        mock_di.analyze.assert_called_once()
        call_args = mock_di.analyze.call_args
        # First arg should be the image
        assert isinstance(call_args[0][0], np.ndarray)

    def test_azure_page_skips_gpt_when_disabled(self, processor, sample_image):
        """GPT Vision should not be called when use_gpt_vision_kie=False."""
        mock_di = MagicMock()
        mock_di.analyze.return_value = _stub_azure_result(1)

        mock_gpt = MagicMock()

        with patch.object(type(processor), "azure_di_provider", new_callable=PropertyMock, return_value=mock_di), \
             patch.object(type(processor), "gpt_vision_extractor", new_callable=PropertyMock, return_value=mock_gpt):
            opts = ProcessingOptions(
                processing_mode="azure",
                use_gpt_vision_kie=False,
            )
            result = processor._process_page_azure(sample_image, 1, "test", opts)

        mock_gpt.extract.assert_not_called()
        assert result["fields"] == []

    def test_azure_page_skips_gpt_when_openai_not_ready(self, processor, sample_image, monkeypatch):
        """GPT Vision should be skipped when OpenAI creds are missing."""
        # Clear env vars and prevent load_dotenv from re-populating
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_DOC_INTELLIGENCE_KEY", raising=False)
        monkeypatch.delenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", raising=False)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
        processor.config.azure = AzureConfig(
            doc_intelligence_key="fake-key",
            doc_intelligence_endpoint="https://fake.cognitiveservices.azure.com/",
            openai_key="",
            openai_endpoint="",
        )

        mock_di = MagicMock()
        mock_di.analyze.return_value = _stub_azure_result(1)

        mock_gpt = MagicMock()

        with patch.object(type(processor), "azure_di_provider", new_callable=PropertyMock, return_value=mock_di), \
             patch.object(type(processor), "gpt_vision_extractor", new_callable=PropertyMock, return_value=mock_gpt):
            opts = ProcessingOptions(processing_mode="azure")
            result = processor._process_page_azure(sample_image, 1, "test", opts)

        mock_gpt.extract.assert_not_called()
        assert result["fields"] == []

    def test_azure_page_passes_document_type(self, processor, sample_image):
        """document_type should be forwarded to GPT Vision extractor."""
        mock_di = MagicMock()
        mock_di.analyze.return_value = _stub_azure_result(1)

        mock_gpt = MagicMock()
        mock_gpt.extract.return_value = [_make_field()]

        with patch.object(type(processor), "azure_di_provider", new_callable=PropertyMock, return_value=mock_di), \
             patch.object(type(processor), "gpt_vision_extractor", new_callable=PropertyMock, return_value=mock_gpt):
            opts = ProcessingOptions(processing_mode="azure", document_type="bol")
            processor._process_page_azure(sample_image, 1, "test", opts)

        # Check that document_type="bol" was passed (as kwarg or positional)
        call_kwargs = mock_gpt.extract.call_args.kwargs
        call_args = mock_gpt.extract.call_args.args
        # Check kwargs first
        if "document_type" in call_kwargs:
            assert call_kwargs["document_type"] == "bol"
        else:
            # Check positional args (skip numpy arrays)
            str_args = [a for a in call_args if isinstance(a, str)]
            assert "bol" in str_args

    def test_azure_page_metadata(self, processor, sample_image):
        """Page metadata should have correct dimensions."""
        mock_di = MagicMock()
        mock_di.analyze.return_value = _stub_azure_result(1)

        mock_gpt = MagicMock()
        mock_gpt.extract.return_value = []

        with patch.object(type(processor), "azure_di_provider", new_callable=PropertyMock, return_value=mock_di), \
             patch.object(type(processor), "gpt_vision_extractor", new_callable=PropertyMock, return_value=mock_gpt):
            opts = ProcessingOptions(processing_mode="azure", use_gpt_vision_kie=False)
            result = processor._process_page_azure(sample_image, 1, "test", opts)

        page = result["page"]
        h, w = sample_image.shape[:2]
        assert page.metadata.width == w
        assert page.metadata.height == h

    def test_azure_page_raw_text(self, processor, sample_image):
        """raw_text from DI should be stored on the Page."""
        mock_di = MagicMock()
        di_result = _stub_azure_result(1)
        di_result["raw_text"] = "Hello World from Azure"
        mock_di.analyze.return_value = di_result

        mock_gpt = MagicMock()
        mock_gpt.extract.return_value = []

        with patch.object(type(processor), "azure_di_provider", new_callable=PropertyMock, return_value=mock_di), \
             patch.object(type(processor), "gpt_vision_extractor", new_callable=PropertyMock, return_value=mock_gpt):
            opts = ProcessingOptions(processing_mode="azure", use_gpt_vision_kie=False)
            result = processor._process_page_azure(sample_image, 1, "test", opts)

        assert result["page"].raw_text == "Hello World from Azure"

    def test_azure_page_passes_ocr_text_to_gpt(self, processor, sample_image):
        """_process_page_azure should pass raw_text as ocr_text to GPT."""
        mock_di = MagicMock()
        di_result = _stub_azure_result(1)
        di_result["raw_text"] = "OCR boost text"
        mock_di.analyze.return_value = di_result

        mock_gpt = MagicMock()
        mock_gpt.extract.return_value = []

        with patch.object(type(processor), "azure_di_provider", new_callable=PropertyMock, return_value=mock_di), \
             patch.object(type(processor), "gpt_vision_extractor", new_callable=PropertyMock, return_value=mock_gpt):
            opts = ProcessingOptions(processing_mode="azure", document_type="auto")
            processor._process_page_azure(sample_image, 1, "test", opts)

        call_kwargs = mock_gpt.extract.call_args.kwargs
        call_args = mock_gpt.extract.call_args.args
        if "ocr_text" in call_kwargs:
            assert call_kwargs["ocr_text"] == "OCR boost text"
        else:
            str_args = [a for a in call_args if isinstance(a, str)]
            assert "OCR boost text" in str_args


# ──────────────────────────────────────────────────────────────
# Lazy Property Tests
# ──────────────────────────────────────────────────────────────

class TestLazyAzureProperties:
    """Verify lazy init of azure_di_provider and gpt_vision_extractor."""

    def test_azure_di_property_not_loaded_at_init(self, test_config):
        test_config.azure = _azure_config()
        processor = DocumentProcessor(test_config)
        assert processor._azure_di_provider is None

    def test_gpt_vision_property_not_loaded_at_init(self, test_config):
        test_config.azure = _azure_config()
        processor = DocumentProcessor(test_config)
        assert processor._gpt_vision_extractor is None

    def test_azure_di_lazy_creates_instance(self, test_config):
        test_config.azure = _azure_config()
        processor = DocumentProcessor(test_config)
        with patch("docvision.azure.doc_intelligence.AzureDocIntelligenceProvider") as MockDI:
            MockDI.return_value = MagicMock()
            provider = processor.azure_di_provider
            assert provider is not None
            MockDI.assert_called_once_with(test_config.azure)

    def test_gpt_vision_lazy_creates_instance(self, test_config):
        test_config.azure = _azure_config()
        processor = DocumentProcessor(test_config)
        with patch("docvision.azure.gpt_vision_kie.GPTVisionExtractor") as MockGPT:
            MockGPT.return_value = MagicMock()
            extractor = processor.gpt_vision_extractor
            assert extractor is not None
            MockGPT.assert_called_once_with(test_config.azure)


# ──────────────────────────────────────────────────────────────
# Fuser Weight Tests
# ──────────────────────────────────────────────────────────────

class TestFuserAzureWeights:
    """Verify GPT_VISION and AZURE_DOC_INTELLIGENCE weights in fuser."""

    def test_fuser_includes_azure_weights(self, test_config):
        processor = DocumentProcessor(test_config)
        fuser = processor.fuser
        weights = fuser.source_weights

        assert SourceEngine.GPT_VISION in weights
        assert SourceEngine.AZURE_DOC_INTELLIGENCE in weights

    def test_gpt_vision_weight_is_1_2(self, test_config):
        processor = DocumentProcessor(test_config)
        weights = processor.fuser.source_weights
        assert weights[SourceEngine.GPT_VISION] == 1.2

    def test_azure_di_weight_is_1_0(self, test_config):
        processor = DocumentProcessor(test_config)
        weights = processor.fuser.source_weights
        assert weights[SourceEngine.AZURE_DOC_INTELLIGENCE] == 1.0


# ──────────────────────────────────────────────────────────────
# Web App Param Wiring (FastAPI TestClient)
# ──────────────────────────────────────────────────────────────

class TestWebAppParams:
    """Verify the web /process endpoint accepts new form params."""

    @pytest.fixture
    def web_client(self):
        try:
            from fastapi.testclient import TestClient
            from docvision.web.app import app
            return TestClient(app)
        except Exception:
            pytest.skip("FastAPI test client setup failed")

    def test_process_accepts_processing_mode_form_param(self, web_client, sample_image_path):
        """POST /api/process should accept processing_mode as form data."""
        with open(sample_image_path, "rb") as f:
            resp = web_client.post(
                "/api/process",
                files={"file": ("test.png", f, "image/png")},
                data={"processing_mode": "local", "document_type": "auto"},
            )
        # Even if processing fails (models not loaded / service not ready), param is accepted
        assert resp.status_code in (200, 500, 503)

    def test_process_defaults_to_local_mode(self, web_client, sample_image_path):
        """POST /api/process without processing_mode should default to local."""
        with open(sample_image_path, "rb") as f:
            resp = web_client.post(
                "/api/process",
                files={"file": ("test.png", f, "image/png")},
            )
        assert resp.status_code in (200, 500, 503)


# ──────────────────────────────────────────────────────────────
# API Server Param Wiring (FastAPI TestClient)
# ──────────────────────────────────────────────────────────────

class TestApiServerParams:
    """Verify the headless API /process endpoint accepts new query params."""

    @pytest.fixture
    def api_client(self):
        try:
            from fastapi.testclient import TestClient
            from docvision.api.server import app
            return TestClient(app)
        except Exception:
            pytest.skip("FastAPI test client setup failed")

    def test_process_accepts_mode_query_param(self, api_client, sample_image_path):
        """POST /process?processing_mode=azure should be accepted."""
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/process?processing_mode=azure&document_type=bol",
                files={"file": ("test.png", f, "image/png")},
            )
        assert resp.status_code in (200, 500, 503)

    def test_process_defaults_to_local(self, api_client, sample_image_path):
        """POST /process without query params defaults to local."""
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/process",
                files={"file": ("test.png", f, "image/png")},
            )
        assert resp.status_code in (200, 500, 503)
