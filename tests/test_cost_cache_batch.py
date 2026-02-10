"""
Tests for Azure cost tracking, response caching, and batch PDF processing.

All Azure API calls are mocked — no real network traffic.
"""

import json
import time
import threading

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from docvision.azure.cost_tracker import (
    CostTracker,
    APICallRecord,
    DI_COST_PER_PAGE,
    GPT_COST_PER_1K_INPUT,
    GPT_COST_PER_1K_OUTPUT,
)
from docvision.azure.response_cache import ResponseCache
from docvision.config import AzureConfig


# ═══════════════════════════════════════════════════════════════════════════
# Cost Tracker
# ═══════════════════════════════════════════════════════════════════════════


class TestCostTracker:
    """Unit tests for CostTracker."""

    @pytest.mark.unit
    def test_empty_tracker(self):
        tracker = CostTracker()
        assert tracker.total_calls == 0
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_di_calls == 0
        assert tracker.total_gpt_calls == 0
        assert tracker.total_tokens == 0
        assert tracker.cache_hit_count == 0

    @pytest.mark.unit
    def test_record_di_call(self):
        tracker = CostTracker()
        rec = tracker.record_di_call(pages=3, model="prebuilt-layout", latency=1.5)

        assert tracker.total_calls == 1
        assert tracker.total_di_calls == 1
        assert tracker.total_pages_analysed == 3
        assert rec.service == "doc_intelligence"
        assert rec.pages == 3
        expected = 3 * DI_COST_PER_PAGE["prebuilt-layout"]
        assert abs(rec.estimated_cost_usd - expected) < 1e-6
        assert tracker.total_cost_usd == pytest.approx(expected, abs=1e-6)

    @pytest.mark.unit
    def test_record_gpt_call(self):
        tracker = CostTracker()
        rec = tracker.record_gpt_call(
            prompt_tokens=1000,
            completion_tokens=200,
            deployment="gpt-4o-mini",
            latency=2.0,
        )

        assert tracker.total_calls == 1
        assert tracker.total_gpt_calls == 1
        assert tracker.total_tokens == 1200
        assert rec.service == "gpt_vision"
        assert rec.prompt_tokens == 1000
        assert rec.completion_tokens == 200

        expected = (
            (1000 / 1000) * GPT_COST_PER_1K_INPUT["gpt-4o-mini"]
            + (200 / 1000) * GPT_COST_PER_1K_OUTPUT["gpt-4o-mini"]
        )
        assert abs(rec.estimated_cost_usd - expected) < 1e-6

    @pytest.mark.unit
    def test_cached_call_zero_cost(self):
        tracker = CostTracker()
        rec = tracker.record_di_call(pages=5, model="prebuilt-layout", cached=True)

        assert rec.cached is True
        assert rec.estimated_cost_usd == 0.0
        assert tracker.total_cost_usd == 0.0
        assert tracker.cache_hit_count == 1

    @pytest.mark.unit
    def test_cost_saved_by_cache(self):
        tracker = CostTracker()
        tracker.record_di_call(pages=2, model="prebuilt-layout", cached=True)
        saved = tracker.cost_saved_by_cache
        expected = 2 * DI_COST_PER_PAGE["prebuilt-layout"]
        assert abs(saved - expected) < 1e-6

    @pytest.mark.unit
    def test_multiple_calls(self):
        tracker = CostTracker()
        tracker.record_di_call(pages=1)
        tracker.record_di_call(pages=2)
        tracker.record_gpt_call(prompt_tokens=500, completion_tokens=100)

        assert tracker.total_calls == 3
        assert tracker.total_di_calls == 2
        assert tracker.total_gpt_calls == 1
        assert tracker.total_pages_analysed == 3

    @pytest.mark.unit
    def test_reset(self):
        tracker = CostTracker()
        tracker.record_di_call(pages=5)
        tracker.record_gpt_call(prompt_tokens=1000, completion_tokens=200)
        tracker.reset()

        assert tracker.total_calls == 0
        assert tracker.total_cost_usd == 0.0

    @pytest.mark.unit
    def test_summary_string(self):
        tracker = CostTracker()
        tracker.record_di_call(pages=1)
        s = tracker.summary()
        assert "Azure API Cost Summary" in s
        assert "Total API calls" in s

    @pytest.mark.unit
    def test_to_dict(self):
        tracker = CostTracker()
        tracker.record_di_call(pages=1, model="prebuilt-layout", latency=1.0)
        d = tracker.to_dict()

        assert d["total_calls"] == 1
        assert d["total_di_calls"] == 1
        assert "records" in d
        assert len(d["records"]) == 1
        assert d["records"][0]["service"] == "doc_intelligence"

    @pytest.mark.unit
    def test_unknown_model_uses_default(self):
        tracker = CostTracker()
        rec = tracker.record_di_call(pages=1, model="custom-model")
        expected = DI_COST_PER_PAGE["default"]
        assert abs(rec.estimated_cost_usd - expected) < 1e-6

    @pytest.mark.unit
    def test_thread_safety(self):
        tracker = CostTracker()
        errors = []

        def add_calls():
            try:
                for _ in range(50):
                    tracker.record_di_call(pages=1)
                    tracker.record_gpt_call(prompt_tokens=10, completion_tokens=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_calls) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert tracker.total_calls == 400  # 4 threads × 50 × 2 calls


# ═══════════════════════════════════════════════════════════════════════════
# Response Cache
# ═══════════════════════════════════════════════════════════════════════════


class TestResponseCache:
    """Unit tests for ResponseCache."""

    @pytest.mark.unit
    def test_make_key_deterministic(self):
        k1 = ResponseCache.make_key(b"hello", service="di", model="prebuilt-layout")
        k2 = ResponseCache.make_key(b"hello", service="di", model="prebuilt-layout")
        assert k1 == k2

    @pytest.mark.unit
    def test_make_key_differs_by_service(self):
        k1 = ResponseCache.make_key(b"hello", service="di")
        k2 = ResponseCache.make_key(b"hello", service="gpt")
        assert k1 != k2

    @pytest.mark.unit
    def test_make_key_differs_by_content(self):
        k1 = ResponseCache.make_key(b"doc_a", service="di")
        k2 = ResponseCache.make_key(b"doc_b", service="di")
        assert k1 != k2

    @pytest.mark.unit
    def test_put_and_get(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        key = cache.make_key(b"test", service="di")
        cache.put(key, {"text_lines": [], "tables": []})

        result = cache.get(key)
        assert result is not None
        assert "text_lines" in result

    @pytest.mark.unit
    def test_miss_returns_none(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        result = cache.get("nonexistent-key")
        assert result is None

    @pytest.mark.unit
    def test_has(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        key = cache.make_key(b"test", service="di")

        assert cache.has(key) is False
        cache.put(key, {"data": 123})
        assert cache.has(key) is True

    @pytest.mark.unit
    def test_disabled_cache(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), enabled=False)
        key = cache.make_key(b"test", service="di")
        cache.put(key, {"data": 123})

        assert cache.get(key) is None
        assert cache.has(key) is False

    @pytest.mark.unit
    def test_stats(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        key = cache.make_key(b"test", service="di")

        cache.get("miss-1")
        cache.get("miss-2")
        cache.put(key, {"data": 1})
        cache.get(key)

        s = cache.stats()
        assert s["hits"] == 1
        assert s["misses"] == 2
        assert s["entries"] == 1
        assert s["enabled"] is True

    @pytest.mark.unit
    def test_clear(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        for i in range(5):
            key = cache.make_key(f"doc_{i}".encode(), service="di")
            cache.put(key, {"data": i})

        assert cache.size == 5
        count = cache.clear()
        assert count == 5
        assert cache.size == 0

    @pytest.mark.unit
    def test_persistence(self, tmp_path):
        """Cache survives object re-creation (file-based)."""
        cache_dir = str(tmp_path / "cache")
        cache = ResponseCache(cache_dir=cache_dir)
        key = cache.make_key(b"persist", service="gpt")
        cache.put(key, {"field": "value"})

        # Create a new instance pointing to the same dir
        cache2 = ResponseCache(cache_dir=cache_dir)
        result = cache2.get(key)
        assert result is not None
        assert result["field"] == "value"

    @pytest.mark.unit
    def test_eviction(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"), max_entries=3)
        keys = []
        for i in range(5):
            key = cache.make_key(f"doc_{i}".encode(), service="di")
            cache.put(key, {"data": i})
            keys.append(key)
            time.sleep(0.01)  # Ensure different mtime

        # Should have evicted 2 oldest
        assert cache.size <= 3

    @pytest.mark.unit
    def test_serialise_pydantic_model(self, tmp_path):
        """Cache can serialise Pydantic models."""
        from docvision.types import BoundingBox

        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        key = cache.make_key(b"pydantic", service="di")
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        cache.put(key, {"bbox": bbox})

        result = cache.get(key)
        assert result is not None
        assert result["bbox"]["x1"] == 0

    @pytest.mark.unit
    def test_hit_rate(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        key = cache.make_key(b"test", service="di")
        cache.put(key, {"data": 1})

        cache.get(key)   # hit
        cache.get(key)   # hit
        cache.get("bad") # miss

        assert cache.hit_rate == pytest.approx(2 / 3, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════
# Provider integration (cost tracker + cache wiring)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def azure_config():
    return AzureConfig(
        doc_intelligence_endpoint="https://fake.cognitiveservices.azure.com/",
        doc_intelligence_key="fake-key",
        openai_endpoint="https://fake.openai.azure.com/",
        openai_key="fake-key",
        openai_deployment="gpt-4o-mini",
    )


@pytest.fixture
def dummy_image():
    return np.zeros((100, 200, 3), dtype=np.uint8)


class TestDIProviderCostAndCache:
    """Integration: DI provider with tracker + cache."""

    @pytest.mark.unit
    def test_analyze_records_cost(self, azure_config, dummy_image, tmp_path):
        from docvision.azure.doc_intelligence import AzureDocIntelligenceProvider

        tracker = CostTracker()
        cache = ResponseCache(cache_dir=str(tmp_path / "c"), enabled=False)
        provider = AzureDocIntelligenceProvider(azure_config, tracker, cache)

        # Mock the SDK client
        mock_result = MagicMock()
        mock_result.pages = [MagicMock(lines=[], words=[])]
        mock_result.tables = []
        mock_result.paragraphs = []
        mock_result.content = "hello"

        mock_poller = MagicMock()
        mock_poller.result.return_value = mock_result

        with patch.object(
            type(provider), "client", new_callable=PropertyMock
        ) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.begin_analyze_document.return_value = mock_poller
            mock_client_prop.return_value = mock_client

            provider.analyze(dummy_image, page_num=1)

        assert tracker.total_di_calls == 1
        assert tracker.total_cost_usd > 0

    @pytest.mark.unit
    def test_analyze_cache_hit(self, azure_config, dummy_image, tmp_path):
        from docvision.azure.doc_intelligence import AzureDocIntelligenceProvider

        tracker = CostTracker()
        cache = ResponseCache(cache_dir=str(tmp_path / "c"))
        provider = AzureDocIntelligenceProvider(azure_config, tracker, cache)

        # Pre-populate cache
        image_bytes = provider._encode_image(dummy_image)
        key = cache.make_key(
            image_bytes, service="di",
            model=azure_config.doc_intelligence_model,
        )
        cache.put(key, {
            "text_lines": [], "tables": [],
            "layout_regions": [], "raw_text": "cached",
        })

        # Should NOT call the SDK
        result = provider.analyze(dummy_image, page_num=1)
        assert result["raw_text"] == "cached"
        assert tracker.cache_hit_count == 1
        assert tracker.total_cost_usd == 0.0


class TestGPTProviderCostAndCache:
    """Integration: GPT Vision provider with tracker + cache."""

    @pytest.mark.unit
    def test_extract_records_cost(self, azure_config, dummy_image, tmp_path):
        from docvision.azure.gpt_vision_kie import GPTVisionExtractor

        tracker = CostTracker()
        cache = ResponseCache(cache_dir=str(tmp_path / "c"), enabled=False)
        extractor = GPTVisionExtractor(azure_config, tracker, cache)

        # Mock OpenAI client
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 500
        mock_usage.completion_tokens = 100

        mock_choice = MagicMock()
        mock_choice.message.content = '{"invoice_number": "INV-001"}'

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch.object(
            type(extractor), "client", new_callable=PropertyMock
        ) as mock_prop:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_prop.return_value = mock_client

            fields = extractor.extract(dummy_image, page_num=1)

        assert tracker.total_gpt_calls == 1
        assert tracker.total_tokens == 600
        assert len(fields) == 1

    @pytest.mark.unit
    def test_extract_cache_hit(self, azure_config, dummy_image, tmp_path):
        from docvision.azure.gpt_vision_kie import GPTVisionExtractor

        tracker = CostTracker()
        cache = ResponseCache(cache_dir=str(tmp_path / "c"))
        extractor = GPTVisionExtractor(azure_config, tracker, cache)

        # Pre-populate cache
        image_b64 = extractor._encode_image_b64(dummy_image).encode()
        key = cache.make_key(
            image_b64,
            service="gpt",
            model=azure_config.openai_deployment,
            extra="auto",
        )
        cache.put(key, {"invoice_number": "CACHED-001"})

        fields = extractor.extract(dummy_image, page_num=1)
        assert tracker.cache_hit_count == 1
        assert any(f.value == "CACHED-001" for f in fields)


# ═══════════════════════════════════════════════════════════════════════════
# Batch PDF processing
# ═══════════════════════════════════════════════════════════════════════════


class TestBatchPDFProcessing:
    """Tests for Azure batch PDF optimisation in the orchestrator."""

    def _make_azure_config(self):
        return AzureConfig(
            doc_intelligence_endpoint="https://fake.cognitiveservices.azure.com/",
            doc_intelligence_key="fake-key",
            openai_endpoint="https://fake.openai.azure.com/",
            openai_key="fake-key",
            openai_deployment="gpt-4o-mini",
        )

    @pytest.mark.unit
    def test_batch_route_taken_for_multi_page_pdf(self, tmp_path):
        """Multi-page PDF in azure mode should call _process_pdf_azure_batch."""
        from docvision.pipeline.orchestrator import DocumentProcessor, ProcessingOptions
        from docvision.config import Config

        config = Config(azure=self._make_azure_config())
        config.artifacts.enable = False
        processor = DocumentProcessor(config)

        # Create a minimal fake PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

        # Mock the PDF loader to return 3 pages
        fake_images = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]

        opts = ProcessingOptions(
            processing_mode="azure",
            save_artifacts=False,
            save_json=False,
            run_validators=False,
        )

        with patch.object(
            type(processor), "pdf_loader", new_callable=PropertyMock
        ) as mock_loader, \
             patch.object(processor, "_process_pdf_azure_batch") as mock_batch:

            mock_loader_inst = MagicMock()
            mock_loader_inst.load.return_value = fake_images
            mock_loader.return_value = mock_loader_inst

            mock_batch.return_value = MagicMock(success=True)

            processor.process(str(pdf_path), opts)

            # Should have called batch, not page-by-page
            mock_batch.assert_called_once()

    @pytest.mark.unit
    def test_single_page_does_not_batch(self, tmp_path):
        """Single-page image should NOT route to batch."""
        from docvision.pipeline.orchestrator import DocumentProcessor, ProcessingOptions
        from docvision.config import Config

        config = Config(azure=self._make_azure_config())
        config.artifacts.enable = False
        processor = DocumentProcessor(config)

        img_path = tmp_path / "test.png"
        # Create a valid PNG
        import cv2
        cv2.imwrite(str(img_path), np.zeros((100, 200, 3), dtype=np.uint8))

        opts = ProcessingOptions(
            processing_mode="azure",
            save_artifacts=False,
            save_json=False,
            run_validators=False,
        )

        with patch.object(processor, "_process_pdf_azure_batch") as mock_batch, \
             patch.object(processor, "_process_page_azure") as mock_page_azure:

            mock_page_azure.return_value = {
                "page": MagicMock(),
                "tables": [],
                "fields": [],
            }

            processor.process(str(img_path), opts)

            mock_batch.assert_not_called()
            mock_page_azure.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator cost stats API
# ═══════════════════════════════════════════════════════════════════════════


class TestOrchestratorCostStats:
    """Tests for DocumentProcessor.get_cost_stats()."""

    @pytest.mark.unit
    def test_get_cost_stats_no_calls(self):
        from docvision.pipeline.orchestrator import DocumentProcessor
        from docvision.config import Config

        processor = DocumentProcessor(Config())
        stats = processor.get_cost_stats()

        # No tracker initialized yet → default zeros
        assert stats["costs"]["total_calls"] == 0

    @pytest.mark.unit
    def test_get_cost_stats_after_recording(self):
        from docvision.pipeline.orchestrator import DocumentProcessor
        from docvision.config import Config

        processor = DocumentProcessor(Config())
        processor.cost_tracker.record_di_call(pages=2)
        processor.cost_tracker.record_gpt_call(prompt_tokens=100, completion_tokens=50)

        stats = processor.get_cost_stats()
        assert stats["costs"]["total_calls"] == 2
        assert stats["costs"]["total_di_calls"] == 1
        assert stats["costs"]["total_gpt_calls"] == 1
        assert stats["costs"]["estimated_cost_usd"] > 0

    @pytest.mark.unit
    def test_print_cost_summary(self, capsys):
        from docvision.pipeline.orchestrator import DocumentProcessor
        from docvision.config import Config

        processor = DocumentProcessor(Config())
        processor.cost_tracker.record_di_call(pages=1)
        processor.print_cost_summary()  # should not raise


# ═══════════════════════════════════════════════════════════════════════════
# Web app cost/cache endpoints
# ═══════════════════════════════════════════════════════════════════════════


class TestWebCostEndpoints:
    """Tests for /api/costs and /api/cache endpoints."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        """FastAPI TestClient with mocked processor."""
        monkeypatch.delenv("AZURE_DOC_INTELLIGENCE_KEY", raising=False)
        monkeypatch.delenv("AZURE_DOC_INTELLIGENCE_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)

        from docvision.web.app import app
        import docvision.web.app as web_module
        from docvision.pipeline.orchestrator import DocumentProcessor
        from docvision.config import Config
        from fastapi.testclient import TestClient

        processor = DocumentProcessor(Config())
        monkeypatch.setattr(web_module, "_processor", processor)

        return TestClient(app)

    @pytest.mark.unit
    def test_get_costs(self, client):
        resp = client.get("/api/costs")
        assert resp.status_code == 200
        data = resp.json()
        assert "costs" in data
        assert "cache" in data

    @pytest.mark.unit
    def test_reset_costs(self, client):
        resp = client.post("/api/costs/reset")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    @pytest.mark.unit
    def test_clear_cache(self, client):
        resp = client.post("/api/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
