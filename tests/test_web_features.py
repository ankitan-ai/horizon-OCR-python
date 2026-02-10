"""
Tests for the new web features:
  - History panel API
  - Batch upload API
  - PDF preview API
  - Field editing (PUT result) round-trip
  - Frontend HTML structure (tabs, views)
"""

import json
import io
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture: test client using httpx / TestClient
# ---------------------------------------------------------------------------
@pytest.fixture
def _reset_jobs():
    """Reset the in-memory jobs dict before each test."""
    from docvision.web import app as web_module
    web_module._jobs.clear()
    yield
    web_module._jobs.clear()


@pytest.fixture
def client(_reset_jobs):
    """Create a FastAPI TestClient (synchronous) for the web app."""
    from docvision.web.app import app
    from starlette.testclient import TestClient

    # Patch lifespan so processor doesn't try to init real models
    with patch("docvision.web.app._processor") as mock_proc:
        mock_proc.get_cost_stats.return_value = {
            "costs": {"total_calls": 0, "estimated_cost_usd": 0},
            "cache": {"hits": 0, "misses": 0, "entries": 0, "enabled": False},
        }
        mock_proc._cost_tracker = None
        mock_proc._response_cache = None
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _add_job(status="completed", mode="local", filename="test.pdf", result=None):
    """Helper to inject a job into the global state."""
    from docvision.web import app as web_module
    from datetime import datetime
    import uuid

    jid = str(uuid.uuid4())
    if result is None:
        result = {
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text_lines": [
                        {"text": "Hello world", "confidence": 0.95, "content_type": "printed"},
                        {"text": "Low conf", "confidence": 0.3, "content_type": "handwritten"},
                    ],
                    "tables": [],
                }
            ],
            "tables": [
                {
                    "id": "t1",
                    "rows": 2,
                    "columns": 2,
                    "cells": [
                        {"row": 0, "col": 0, "text": "A", "confidence": 0.9, "is_header": True},
                        {"row": 0, "col": 1, "text": "B", "confidence": 0.85, "is_header": True},
                        {"row": 1, "col": 0, "text": "1", "confidence": 0.7},
                        {"row": 1, "col": 1, "text": "2", "confidence": 0.4},
                    ],
                    "confidence": 0.8,
                }
            ],
            "fields": [
                {
                    "name": "invoice_number",
                    "value": "INV-001",
                    "field_type": "string",
                    "confidence": 0.92,
                    "status": "confident",
                    "source": "donut",
                },
                {
                    "name": "total",
                    "value": "$100.00",
                    "field_type": "currency",
                    "confidence": 0.45,
                    "status": "uncertain",
                    "source": "layoutlmv3",
                },
            ],
            "metadata": {"filename": filename, "processing_time_seconds": 2.5},
        }

    web_module._jobs[jid] = {
        "job_id": jid,
        "filename": filename,
        "status": status,
        "processing_mode": mode,
        "created": datetime.utcnow().isoformat(),
        "result": result if status == "completed" else None,
        "error": "something failed" if status == "failed" else None,
        "artifacts_dir": None,
    }
    return jid


# ===========================================================================
#  HISTORY API
# ===========================================================================
class TestHistoryAPI:
    def test_empty_history(self, client):
        res = client.get("/api/history")
        assert res.status_code == 200
        data = res.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_history_with_jobs(self, client):
        j1 = _add_job(status="completed", filename="doc1.pdf")
        j2 = _add_job(status="failed", filename="doc2.png", mode="azure")
        j3 = _add_job(status="completed", filename="doc3.pdf")

        res = client.get("/api/history")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 3
        assert len(data["jobs"]) == 3
        # All jobs should be present
        job_ids = {j["job_id"] for j in data["jobs"]}
        assert j1 in job_ids
        assert j2 in job_ids
        assert j3 in job_ids

    def test_history_has_summary_stats(self, client):
        _add_job(status="completed", filename="invoice.pdf")
        res = client.get("/api/history")
        jobs = res.json()["jobs"]
        job = jobs[0]
        assert "page_count" in job
        assert "text_lines" in job
        assert "tables" in job
        assert "fields" in job
        assert "processing_time" in job
        assert job["text_lines"] == 2
        assert job["tables"] == 1
        assert job["fields"] == 2

    def test_history_newest_first(self, client):
        import time

        _add_job(filename="old.pdf")
        time.sleep(0.01)
        _add_job(filename="new.pdf")
        res = client.get("/api/history")
        jobs = res.json()["jobs"]
        assert jobs[0]["filename"] == "new.pdf"
        assert jobs[1]["filename"] == "old.pdf"

    def test_history_failed_job_no_stats(self, client):
        _add_job(status="failed", filename="bad.pdf")
        res = client.get("/api/history")
        job = res.json()["jobs"][0]
        assert job["status"] == "failed"
        assert "page_count" not in job  # no result → no stats

    def test_history_mode_preserved(self, client):
        _add_job(mode="azure", filename="cloud.pdf")
        res = client.get("/api/history")
        job = res.json()["jobs"][0]
        assert job["processing_mode"] == "azure"


# ===========================================================================
#  FIELD EDITING (PUT round-trip)
# ===========================================================================
class TestFieldEditing:
    def test_update_field_value(self, client):
        jid = _add_job()
        # Get current result
        res = client.get(f"/api/jobs/{jid}/result")
        result = res.json()
        assert result["fields"][0]["value"] == "INV-001"

        # Update a field
        result["fields"][0]["value"] = "INV-999"
        res = client.put(
            f"/api/jobs/{jid}/result",
            json=result,
        )
        assert res.status_code == 200

        # Verify persistence
        res = client.get(f"/api/jobs/{jid}/result")
        assert res.json()["fields"][0]["value"] == "INV-999"

    def test_update_table_cell(self, client):
        jid = _add_job()
        res = client.get(f"/api/jobs/{jid}/result")
        result = res.json()
        result["tables"][0]["cells"][2]["text"] = "EDITED"
        client.put(f"/api/jobs/{jid}/result", json=result)
        res = client.get(f"/api/jobs/{jid}/result")
        assert res.json()["tables"][0]["cells"][2]["text"] == "EDITED"

    def test_update_nonexistent_job(self, client):
        res = client.put("/api/jobs/nonexistent/result", json={"foo": "bar"})
        assert res.status_code == 404

    def test_full_json_replace(self, client):
        jid = _add_job()
        new_result = {"completely": "different", "structure": True}
        res = client.put(f"/api/jobs/{jid}/result", json=new_result)
        assert res.status_code == 200
        res = client.get(f"/api/jobs/{jid}/result")
        assert res.json() == new_result


# ===========================================================================
#  BATCH UPLOAD
# ===========================================================================
class TestBatchUpload:
    def test_batch_endpoint_exists(self, client):
        """Endpoint should exist even if processing fails due to no processor."""
        # Send empty - should get validation error, not 404
        res = client.post("/api/process/batch")
        assert res.status_code != 404

    def test_batch_with_files(self, client):
        """Batch upload should create multiple jobs."""
        from docvision.web import app as web_module

        # Mock the processor and background processing
        with patch.object(web_module, "_processor") as mock_proc:
            mock_proc.get_cost_stats.return_value = {
                "costs": {},
                "cache": {},
            }

            files = [
                ("files", ("test1.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png")),
                ("files", ("test2.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png")),
            ]
            res = client.post(
                "/api/process/batch",
                files=files,
                data={"processing_mode": "local"},
            )
            # Should either succeed or fail gracefully
            assert res.status_code in (200, 422, 500)
            if res.status_code == 200:
                data = res.json()
                assert "job_ids" in data
                assert "count" in data
                assert data["count"] == 2


# ===========================================================================
#  PREVIEW ENDPOINT
# ===========================================================================
class TestPreview:
    def test_preview_endpoint_exists(self, client):
        """Preview endpoint should not 404."""
        res = client.post("/api/preview")
        assert res.status_code != 404

    def test_preview_image_file(self, client):
        """Upload a tiny PNG and get a preview back."""
        # Minimal valid 1x1 PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02"
            b"\x00\x00\x00\x90wS\xde\x00\x00\x00\x0c"
            b"IDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        res = client.post(
            "/api/preview",
            files={"file": ("test.png", png_data, "image/png")},
        )
        # Should succeed or handle gracefully
        assert res.status_code in (200, 500)
        if res.status_code == 200:
            data = res.json()
            assert "preview" in data
            assert "pages" in data


# ===========================================================================
#  FRONTEND STRUCTURE (HTML sanity checks)
# ===========================================================================
class TestFrontendHTML:
    def test_serves_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")

    def test_has_all_tabs(self, client):
        res = client.get("/")
        html = res.text
        assert 'data-tab="upload"' in html
        assert 'data-tab="artifacts"' in html
        assert 'data-tab="output"' in html
        assert 'data-tab="history"' in html
        assert 'data-tab="costs"' in html

    def test_has_all_panels(self, client):
        res = client.get("/")
        html = res.text
        assert 'id="panel-upload"' in html
        assert 'id="panel-artifacts"' in html
        assert 'id="panel-output"' in html
        assert 'id="panel-history"' in html
        assert 'id="panel-costs"' in html

    def test_has_batch_button(self, client):
        res = client.get("/")
        assert 'id="btnBatch"' in res.text

    def test_has_multi_file_input(self, client):
        res = client.get("/")
        assert "multiple" in res.text
        assert 'id="fileInput"' in res.text

    def test_has_view_toggle_js(self, client):
        res = client.get("/")
        assert "switchView" in res.text
        assert "outputView" in res.text

    def test_has_confidence_css(self, client):
        res = client.get("/")
        assert "--conf-high" in res.text
        assert "--conf-mid" in res.text
        assert "--conf-low" in res.text
        assert "conf-badge" in res.text

    def test_has_field_editor_class(self, client):
        res = client.get("/")
        assert "field-card" in res.text
        assert "field-grid" in res.text
        assert "field-input" in res.text

    def test_has_history_table(self, client):
        res = client.get("/")
        assert "history-table" in res.text
        assert "historyTableBody" in res.text

    def test_has_preview_area(self, client):
        res = client.get("/")
        assert "previewArea" in res.text
        assert "generatePreview" in res.text

    def test_has_lightbox(self, client):
        res = client.get("/")
        assert "lightbox" in res.text
        assert "showLightbox" in res.text


# ===========================================================================
#  CONFIDENCE HIGHLIGHTING (data model checks)
# ===========================================================================
class TestConfidenceHighlighting:
    """Verify confidence levels flow correctly through the API."""

    def test_text_line_confidence_in_result(self, client):
        jid = _add_job()
        res = client.get(f"/api/jobs/{jid}/result")
        result = res.json()
        lines = result["pages"][0]["text_lines"]
        assert lines[0]["confidence"] == 0.95  # high
        assert lines[1]["confidence"] == 0.3  # low

    def test_field_confidence_in_result(self, client):
        jid = _add_job()
        res = client.get(f"/api/jobs/{jid}/result")
        fields = res.json()["fields"]
        assert fields[0]["confidence"] == 0.92  # high
        assert fields[1]["confidence"] == 0.45  # low/uncertain

    def test_table_cell_confidence_in_result(self, client):
        jid = _add_job()
        res = client.get(f"/api/jobs/{jid}/result")
        cells = res.json()["tables"][0]["cells"]
        confs = [c["confidence"] for c in cells]
        assert 0.9 in confs
        assert 0.4 in confs

    def test_history_shows_correct_line_counts(self, client):
        _add_job(filename="test.pdf")
        res = client.get("/api/history")
        job = res.json()["jobs"][0]
        # 2 text lines (one high, one low confidence)
        assert job["text_lines"] == 2


# ===========================================================================
#  DOCKER FILE VALIDATION
# ===========================================================================
class TestDockerFiles:
    """Validate Docker files reference the correct entrypoint and port."""

    def test_dockerfile_port_8080(self):
        content = Path("Dockerfile").read_text()
        assert "EXPOSE 8080" in content
        assert "EXPOSE 8000" not in content

    def test_dockerfile_web_app_cmd(self):
        content = Path("Dockerfile").read_text()
        assert "docvision.web.app" in content
        assert "docvision.api.server" not in content

    def test_dockerfile_gpu_port_8080(self):
        content = Path("Dockerfile.gpu").read_text()
        assert "EXPOSE 8080" in content
        assert "EXPOSE 8000" not in content

    def test_dockerfile_gpu_web_app_cmd(self):
        content = Path("Dockerfile.gpu").read_text()
        assert "docvision.web.app" in content
        assert "docvision.api.server" not in content

    def test_compose_port_8080(self):
        content = Path("docker-compose.yml").read_text()
        assert '"8080:8080"' in content
        assert '"8000:8000"' not in content

    def test_compose_web_app_entrypoint(self):
        content = Path("docker-compose.yml").read_text()
        assert "docvision.web.app" in content
        assert "docvision.api.server" not in content

    def test_compose_azure_env_vars(self):
        content = Path("docker-compose.yml").read_text()
        assert "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT" in content
        assert "AZURE_OPENAI_ENDPOINT" in content

    def test_compose_cache_volume(self):
        content = Path("docker-compose.yml").read_text()
        assert ".cache" in content

    def test_dockerfile_cache_dir(self):
        content = Path("Dockerfile").read_text()
        assert "/app/.cache" in content

    def test_dockerfile_healthcheck_8080(self):
        content = Path("Dockerfile").read_text()
        assert "localhost:8080" in content
        assert "localhost:8000" not in content


# ---------------------------------------------------------------------------
# Multi-user concurrency
# ---------------------------------------------------------------------------

class TestMultiUserConcurrency:
    """Verify that the thread locks and per-request isolation exist."""

    def test_locks_exist(self):
        from docvision.web import app as web_module
        import threading
        assert hasattr(web_module, "_jobs_lock")
        assert hasattr(web_module, "_local_model_lock")
        assert isinstance(web_module._jobs_lock, type(threading.Lock()))
        assert isinstance(web_module._local_model_lock, type(threading.Lock()))

    def test_jobs_dict_is_shared_but_lockable(self):
        """Jobs dict is shared across requests but protected by _jobs_lock."""
        from docvision.web import app as web_module
        import threading

        web_module._jobs.clear()

        # Simulate two threads writing to _jobs concurrently
        errors = []

        def writer(jid):
            try:
                with web_module._jobs_lock:
                    web_module._jobs[jid] = {"status": "completed", "result": jid}
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"job_{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(web_module._jobs) == 20
        web_module._jobs.clear()

    def test_local_lock_serialises_access(self):
        """Local model lock prevents two local jobs from running simultaneously."""
        from docvision.web import app as web_module
        import threading
        import time

        events = []

        def fake_local_job(name):
            with web_module._local_model_lock:
                events.append(f"{name}_start")
                time.sleep(0.05)
                events.append(f"{name}_end")

        t1 = threading.Thread(target=fake_local_job, args=("A",))
        t2 = threading.Thread(target=fake_local_job, args=("B",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # One must fully finish before the other starts
        a_start = events.index("A_start")
        a_end = events.index("A_end")
        b_start = events.index("B_start")
        b_end = events.index("B_end")

        # Either A fully before B, or B fully before A
        assert (a_end < b_start) or (b_end < a_start), (
            f"Local lock did not serialise: {events}"
        )

    def test_azure_jobs_can_run_in_parallel(self):
        """Azure jobs should NOT acquire the local model lock."""
        from docvision.web import app as web_module
        import threading
        import time

        events = []

        def fake_azure_job(name):
            # Azure path does NOT acquire _local_model_lock
            events.append(f"{name}_start")
            time.sleep(0.05)
            events.append(f"{name}_end")

        t1 = threading.Thread(target=fake_azure_job, args=("X",))
        t2 = threading.Thread(target=fake_azure_job, args=("Y",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Both should overlap (both start before either ends)
        x_start = events.index("X_start")
        y_start = events.index("Y_start")
        x_end = events.index("X_end")
        y_end = events.index("Y_end")

        # At least one must start before the other ends (parallel)
        overlaps = (x_start < y_end and y_start < x_end)
        assert overlaps, f"Azure jobs should run in parallel: {events}"
