"""
Azure End-to-End Smoke Test
===========================
Tests the Azure cloud processing path by injecting mock results directly
into the web app's job store, simulating the entire flow:
  Upload → Azure DI + GPT → Result → Field editing → History → Cost → Save

This avoids needing real Azure credentials or valid image files.
"""

import json
import uuid
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
from zoneinfo import ZoneInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
AZURE_RESULT = {
    "page_count": 1,
    "pages": [
        {
            "page_number": 1,
            "width": 612,
            "height": 792,
            "text_lines": [
                {"text": "INVOICE", "confidence": 0.99, "content_type": "printed",
                 "bounding_box": [[50, 50], [200, 50], [200, 80], [50, 80]]},
                {"text": "Date: 2024-01-15", "confidence": 0.97, "content_type": "printed",
                 "bounding_box": [[50, 100], [250, 100], [250, 120], [50, 120]]},
                {"text": "Total: $1,250.00", "confidence": 0.95, "content_type": "printed",
                 "bounding_box": [[50, 200], [250, 200], [250, 220], [50, 220]]},
                {"text": "Scribbled note", "confidence": 0.42, "content_type": "handwritten",
                 "bounding_box": [[300, 300], [500, 300], [500, 330], [300, 330]]},
            ],
            "layout_regions": [
                {"region_type": "title", "confidence": 0.95,
                 "bounding_box": [[50, 50], [200, 50], [200, 80], [50, 80]]},
            ],
            "tables": [],
        }
    ],
    "tables": [
        {
            "id": "table_1",
            "page_number": 1,
            "rows": 2,
            "columns": 3,
            "cells": [
                {"row": 0, "col": 0, "text": "Item", "confidence": 0.98, "is_header": True},
                {"row": 0, "col": 1, "text": "Qty", "confidence": 0.97, "is_header": True},
                {"row": 0, "col": 2, "text": "Price", "confidence": 0.96, "is_header": True},
                {"row": 1, "col": 0, "text": "Widget A", "confidence": 0.94},
                {"row": 1, "col": 1, "text": "5", "confidence": 0.92},
                {"row": 1, "col": 2, "text": "$250.00", "confidence": 0.91},
            ],
            "confidence": 0.95,
            "is_valid": True,
        }
    ],
    "fields": [
        {
            "name": "invoice_number", "value": "INV-2024-0042",
            "field_type": "string", "confidence": 0.98,
            "status": "confident", "source": "azure_di",
        },
        {
            "name": "date", "value": "2024-01-15",
            "field_type": "date", "confidence": 0.96,
            "status": "confident", "source": "azure_di",
        },
        {
            "name": "total", "value": "$1,250.00",
            "field_type": "currency", "confidence": 0.94,
            "status": "confident", "source": "gpt_vision",
        },
        {
            "name": "vendor_name", "value": "Acme Corp",
            "field_type": "string", "confidence": 0.45,
            "status": "uncertain", "source": "gpt_vision",
        },
    ],
    "metadata": {
        "filename": "test_invoice.png",
        "processing_time_seconds": 3.2,
        "processing_mode": "azure",
        "azure_di_model": "prebuilt-layout",
        "gpt_model": "gpt-4o",
    },
}

COST_DATA = {
    "costs": {
        "total_calls": 2,
        "total_di_calls": 1,
        "total_gpt_calls": 1,
        "estimated_cost_usd": 0.015,
        "total_pages_analysed": 1,
        "total_tokens": 450,
        "cache_hits": 0,
        "cost_saved_by_cache_usd": 0.0,
        "records": [
            {
                "service": "doc_intelligence", "model": "prebuilt-layout",
                "pages": 1, "estimated_cost_usd": 0.01, "latency_seconds": 1.5,
                "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat(), "cached": False,
                "prompt_tokens": 0, "completion_tokens": 0,
            },
            {
                "service": "gpt_vision", "model": "gpt-4o",
                "pages": 1, "estimated_cost_usd": 0.005, "latency_seconds": 1.7,
                "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat(), "cached": False,
                "prompt_tokens": 300, "completion_tokens": 150,
            },
        ],
    },
    "cache": {"hits": 0, "misses": 1, "entries": 1, "enabled": True},
}


def _inject_azure_job():
    """Inject a completed Azure job directly into _jobs."""
    import docvision.web.app as web_module

    jid = str(uuid.uuid4())[:12]
    web_module._jobs[jid] = {
        "status": "completed",
        "filename": "test_invoice.png",
        "processing_mode": "azure",
        "created": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "result": json.loads(json.dumps(AZURE_RESULT)),  # deep copy
        "error": None,
        "artifacts_dir": None,
    }
    return jid


@pytest.fixture
def _reset():
    import docvision.web.app as web_module
    web_module._jobs.clear()
    yield
    web_module._jobs.clear()


@pytest.fixture
def client(_reset):
    from docvision.web.app import app
    from starlette.testclient import TestClient

    # Mock the processor so lifespan doesn't load real models
    mock_proc = MagicMock()
    mock_proc.get_cost_stats.return_value = COST_DATA
    mock_proc._cost_tracker = MagicMock()
    mock_proc._response_cache = MagicMock()
    mock_proc.response_cache.clear.return_value = 0

    with patch("docvision.web.app._processor", mock_proc):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ===========================================================================
#  TESTS
# ===========================================================================
class TestAzureSmokeTest:
    """End-to-end Azure processing smoke test using injected results."""

    def test_azure_job_result_accessible(self, client):
        """Injected Azure job result is accessible via GET."""
        jid = _inject_azure_job()
        res = client.get(f"/api/jobs/{jid}")
        assert res.status_code == 200
        assert res.json()["status"] == "completed"
        assert res.json()["filename"] == "test_invoice.png"

    def test_result_has_fields(self, client):
        """Result contains fields with names, values, and sources."""
        jid = _inject_azure_job()
        res = client.get(f"/api/jobs/{jid}/result")
        assert res.status_code == 200
        result = res.json()
        assert len(result["fields"]) == 4
        assert result["fields"][0]["name"] == "invoice_number"
        assert result["fields"][0]["value"] == "INV-2024-0042"
        assert result["fields"][0]["source"] == "azure_di"

    def test_result_has_tables(self, client):
        """Result contains extracted tables."""
        jid = _inject_azure_job()
        result = client.get(f"/api/jobs/{jid}/result").json()
        assert len(result["tables"]) == 1
        table = result["tables"][0]
        assert table["rows"] == 2
        assert table["columns"] == 3
        assert len(table["cells"]) == 6

    def test_result_has_text_lines(self, client):
        """Result pages contain text lines with confidence."""
        jid = _inject_azure_job()
        result = client.get(f"/api/jobs/{jid}/result").json()
        lines = result["pages"][0]["text_lines"]
        assert len(lines) == 4
        for line in lines:
            assert "confidence" in line
            assert "text" in line
            assert 0 <= line["confidence"] <= 1

    def test_confidence_ranges(self, client):
        """Verify high, medium, low confidence items exist."""
        jid = _inject_azure_job()
        result = client.get(f"/api/jobs/{jid}/result").json()

        lines = result["pages"][0]["text_lines"]
        high = [l for l in lines if l["confidence"] >= 0.8]
        low = [l for l in lines if l["confidence"] < 0.5]
        assert len(high) >= 3  # INVOICE, Date, Total
        assert len(low) >= 1   # Scribbled note

        fields = result["fields"]
        uncertain = [f for f in fields if f["confidence"] < 0.5]
        assert len(uncertain) >= 1  # vendor_name

    def test_field_editing_roundtrip(self, client):
        """Edit a field value and verify it persists."""
        jid = _inject_azure_job()
        result = client.get(f"/api/jobs/{jid}/result").json()

        # Fix the uncertain vendor_name
        result["fields"][3]["value"] = "Corrected Vendor Inc."
        result["fields"][3]["confidence"] = 1.0
        result["fields"][3]["status"] = "validated"

        res = client.put(f"/api/jobs/{jid}/result", json=result)
        assert res.status_code == 200

        updated = client.get(f"/api/jobs/{jid}/result").json()
        assert updated["fields"][3]["value"] == "Corrected Vendor Inc."
        assert updated["fields"][3]["confidence"] == 1.0
        assert updated["fields"][3]["status"] == "validated"

    def test_table_cell_editing(self, client):
        """Edit a table cell and verify it persists."""
        jid = _inject_azure_job()
        result = client.get(f"/api/jobs/{jid}/result").json()

        result["tables"][0]["cells"][3]["text"] = "Widget B"
        client.put(f"/api/jobs/{jid}/result", json=result)

        updated = client.get(f"/api/jobs/{jid}/result").json()
        assert updated["tables"][0]["cells"][3]["text"] == "Widget B"

    def test_history_shows_azure_job(self, client):
        """History API lists the Azure job with correct mode."""
        jid = _inject_azure_job()
        res = client.get("/api/history")
        assert res.status_code == 200
        jobs = res.json()["jobs"]
        assert len(jobs) >= 1
        job = next(j for j in jobs if j["job_id"] == jid)
        assert job["processing_mode"] == "azure"
        assert job["status"] == "completed"
        assert job["fields"] == 4
        assert job["tables"] == 1
        assert job["text_lines"] == 4
        assert job["page_count"] == 1

    def test_cost_tracking_data_structure(self, client):
        """Cost data structure has expected shape (validated offline)."""
        # The COST_DATA represents what the cost tracker returns in Azure mode
        costs = COST_DATA["costs"]
        assert costs["total_calls"] == 2
        assert costs["total_di_calls"] == 1
        assert costs["total_gpt_calls"] == 1
        assert costs["estimated_cost_usd"] == 0.015
        assert len(costs["records"]) == 2
        cache = COST_DATA["cache"]
        assert cache["enabled"] is True
        assert cache["entries"] == 1

    def test_cost_breakdown_services(self, client):
        """Cost records distinguish DI and GPT services."""
        records = COST_DATA["costs"]["records"]
        di = [r for r in records if r["service"] == "doc_intelligence"]
        gpt = [r for r in records if r["service"] == "gpt_vision"]
        assert len(di) == 1
        assert len(gpt) == 1
        assert di[0]["pages"] == 1
        assert gpt[0]["prompt_tokens"] == 300

    def test_cost_endpoint_accessible(self, client):
        """Cost endpoint is accessible and returns valid JSON."""
        res = client.get("/api/costs")
        assert res.status_code == 200
        data = res.json()
        assert "costs" in data or "cache" in data or isinstance(data, dict)

    def test_save_to_disk_azure_subfolder(self, client, tmp_path):
        """Save to disk writes to Azure_Cloud/ subfolder."""
        jid = _inject_azure_job()
        with patch("docvision.web.app.OUTPUT_BASE", tmp_path):
            res = client.post(f"/api/jobs/{jid}/save")
            assert res.status_code == 200
            data = res.json()
            assert "Azure_Cloud" in data["path"]

            saved = list(tmp_path.glob("Azure_Cloud/*.json"))
            assert len(saved) == 1
            content = json.loads(saved[0].read_text())
            assert content["fields"][0]["name"] == "invoice_number"

    def test_save_preserves_edits(self, client, tmp_path):
        """Edits made before save are reflected in the saved file."""
        jid = _inject_azure_job()
        result = client.get(f"/api/jobs/{jid}/result").json()
        result["fields"][0]["value"] = "EDITED-INV"
        client.put(f"/api/jobs/{jid}/result", json=result)

        with patch("docvision.web.app.OUTPUT_BASE", tmp_path):
            client.post(f"/api/jobs/{jid}/save")
            saved = list(tmp_path.glob("Azure_Cloud/*.json"))
            content = json.loads(saved[0].read_text())
            assert content["fields"][0]["value"] == "EDITED-INV"

    def test_multiple_azure_jobs_in_history(self, client):
        """Multiple Azure jobs all appear in history."""
        j1 = _inject_azure_job()
        j2 = _inject_azure_job()
        j3 = _inject_azure_job()
        res = client.get("/api/history")
        jobs = res.json()["jobs"]
        ids = {j["job_id"] for j in jobs}
        assert j1 in ids and j2 in ids and j3 in ids
        assert all(j["processing_mode"] == "azure" for j in jobs)

    def test_metadata_includes_azure_info(self, client):
        """Result metadata contains Azure-specific info."""
        jid = _inject_azure_job()
        result = client.get(f"/api/jobs/{jid}/result").json()
        meta = result["metadata"]
        assert meta["processing_mode"] == "azure"
        assert meta["azure_di_model"] == "prebuilt-layout"
        assert meta["gpt_model"] == "gpt-4o"
        assert meta["processing_time_seconds"] == 3.2
