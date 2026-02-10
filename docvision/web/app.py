"""
DocVision Web UI — FastAPI application with a browser-based interface
for uploading documents, viewing intermediate artifacts, and editing
the final JSON output.

Run with:
    python -m docvision.web.app
"""

import os
import uuid
import json
import time
import base64
import hashlib
import tempfile
import asyncio
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from docvision import __version__
from docvision.config import load_config, Config
from docvision.pipeline import DocumentProcessor, ProcessingOptions


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_processor: Optional[DocumentProcessor] = None
_jobs: Dict[str, Dict[str, Any]] = {}

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "docvision_uploads"
ARTIFACTS_BASE = Path("artifacts")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor
    logger.info("Starting DocVision Web UI …")
    config_path = os.environ.get("DOCVISION_CONFIG")
    config = load_config(config_path) if config_path and Path(config_path).exists() else Config()
    # Enable artifacts so the viewer has images to show
    config.artifacts.enable = True
    _processor = DocumentProcessor(config)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_BASE.mkdir(parents=True, exist_ok=True)
    logger.info("DocVision Web UI ready")
    yield
    logger.info("Shutting down DocVision Web UI")
    _processor = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="DocVision Web UI",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Serve static assets (CSS / JS)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page UI."""
    html_path = WEB_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# API: upload + process
# ---------------------------------------------------------------------------
@app.post("/api/process")
async def process_document(
    file: UploadFile = File(...),
    processing_mode: str = Form("local"),
    document_type: str = Form("auto"),
    preprocess: bool = Form(True),
    detect_layout: bool = Form(True),
    detect_text: bool = Form(True),
    detect_tables: bool = Form(True),
    run_ocr: bool = Form(True),
    run_donut: bool = Form(False),
    run_layoutlmv3: bool = Form(False),
    run_validators: bool = Form(True),
):
    """Upload a document, process it, return job_id."""
    if _processor is None:
        raise HTTPException(503, "Service not initialised")

    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    job_id = str(uuid.uuid4())[:12]
    content = await file.read()

    # --- Deduplicate temp uploads using content hash ---
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    upload_path = UPLOAD_DIR / f"{content_hash}{suffix}"
    if not upload_path.exists():
        upload_path.write_bytes(content)
        logger.info(f"Stored new upload: {upload_path.name}")
    else:
        logger.info(f"Upload deduplicated — reusing {upload_path.name}")

    # Determine mode subfolder for artifacts
    mode_subfolder = "Azure_Cloud" if processing_mode == "azure" else "Local"

    _jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "processing_mode": processing_mode,
        "created": datetime.utcnow().isoformat(),
        "result": None,
        "error": None,
        "artifacts_dir": None,
    }

    # Run synchronously in a thread so the event loop stays responsive
    import concurrent.futures
    loop = asyncio.get_event_loop()

    async def _run():
        try:
            opts = ProcessingOptions(
                processing_mode=processing_mode,
                document_type=document_type,
                preprocess=preprocess,
                detect_layout=detect_layout,
                detect_text=detect_text,
                detect_tables=detect_tables,
                run_ocr=run_ocr,
                run_donut=run_donut,
                run_layoutlmv3=run_layoutlmv3,
                run_validators=run_validators,
                save_artifacts=True,
                save_json=False,
            )
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(
                    pool,
                    lambda: _processor.process(str(upload_path), opts),
                )
            if result.success:
                doc_dict = result.document.model_dump(mode="json")
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["result"] = doc_dict
                _jobs[job_id]["artifacts_dir"] = str(
                    ARTIFACTS_BASE / mode_subfolder / result.document.id
                )
                # Log cost summary after Azure processing
                if processing_mode == "azure":
                    _processor.print_cost_summary()
            else:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = result.error
        except Exception as exc:
            logger.exception("Processing failed")
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)

    asyncio.ensure_future(_run())
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# API: job status / result
# ---------------------------------------------------------------------------
@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "created": job["created"],
        "error": job.get("error"),
        "has_result": job["result"] is not None,
    }


@app.get("/api/jobs/{job_id}/result")
async def get_result(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    if job["result"] is None:
        raise HTTPException(404, "No result yet")
    return job["result"]


@app.put("/api/jobs/{job_id}/result")
async def update_result(job_id: str, body: dict):
    """Save edits the user made to the JSON output."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    _jobs[job_id]["result"] = body
    return {"ok": True}


@app.get("/api/jobs/{job_id}/download")
async def download_json(job_id: str):
    if job_id not in _jobs or _jobs[job_id]["result"] is None:
        raise HTTPException(404, "No result")
    job = _jobs[job_id]
    filename = Path(job["filename"]).stem + "_result.json"
    content = json.dumps(job["result"], indent=2, ensure_ascii=False)
    return JSONResponse(
        content=job["result"],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# API: save result to disk
# ---------------------------------------------------------------------------
OUTPUT_BASE = Path("output")


@app.post("/api/jobs/{job_id}/save")
async def save_to_disk(job_id: str):
    """Save the current JSON result to the output/ folder on disk."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    if job["result"] is None:
        raise HTTPException(400, "No result to save")

    mode = job.get("processing_mode", "local")
    subfolder = "Azure_Cloud" if mode == "azure" else "Local"
    out_dir = OUTPUT_BASE / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(job["filename"]).stem
    filename = f"{stem}_{job_id}.json"
    filepath = out_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(job["result"], f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Result saved to disk: {filepath}")
    return {"ok": True, "path": str(filepath)}


# ---------------------------------------------------------------------------
# API: Azure cost tracking & cache stats
# ---------------------------------------------------------------------------
@app.get("/api/costs")
async def get_costs():
    """Return Azure API cost tracking and response cache statistics."""
    if _processor is None:
        raise HTTPException(503, "Service not initialised")
    return _processor.get_cost_stats()


@app.post("/api/costs/reset")
async def reset_costs():
    """Reset cost tracking counters."""
    if _processor is None:
        raise HTTPException(503, "Service not initialised")
    if _processor._cost_tracker is not None:
        _processor.cost_tracker.reset()
    return {"ok": True}


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear the Azure response cache."""
    if _processor is None:
        raise HTTPException(503, "Service not initialised")
    count = 0
    if _processor._response_cache is not None:
        count = _processor.response_cache.clear()
    return {"ok": True, "entries_cleared": count}


# ---------------------------------------------------------------------------
# API: artifacts
# ---------------------------------------------------------------------------
@app.get("/api/jobs/{job_id}/artifacts")
async def list_artifacts(job_id: str):
    """Return list of artifact image paths for the job."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    artifacts_dir = job.get("artifacts_dir")
    if not artifacts_dir or not Path(artifacts_dir).is_dir():
        return {"artifacts": []}

    files = sorted(Path(artifacts_dir).glob("*.png"))
    return {
        "artifacts": [
            {
                "name": f.stem,
                "filename": f.name,
                "url": f"/api/artifacts/{job_id}/{f.name}",
                "size_kb": round(f.stat().st_size / 1024, 1),
            }
            for f in files
        ]
    }


@app.get("/api/artifacts/{job_id}/{filename}")
async def serve_artifact(job_id: str, filename: str):
    """Serve a single artifact image."""
    if job_id not in _jobs:
        raise HTTPException(404)
    job = _jobs[job_id]
    artifacts_dir = job.get("artifacts_dir")
    if not artifacts_dir:
        raise HTTPException(404)
    fpath = Path(artifacts_dir) / filename
    if not fpath.is_file():
        raise HTTPException(404)
    return FileResponse(fpath, media_type="image/png")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    print(f"\n  DocVision Web UI → http://localhost:{port}\n")
    uvicorn.run(
        "docvision.web.app:app",
        host="127.0.0.1",
        port=port,
        reload=False,
    )
