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
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from docvision import __version__
from docvision.io.markdown import generate_markdown, save_markdown
from docvision.config import load_config, Config
from docvision.pipeline import DocumentProcessor, ProcessingOptions


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_processor: Optional[DocumentProcessor] = None
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()           # protects _jobs dict mutations
_local_model_lock = threading.Lock()    # serialises local-model inference

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
        "created": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "result": None,
        "error": None,
        "artifacts_dir": None,
    }

    # Run synchronously in a thread so the event loop stays responsive
    import concurrent.futures
    loop = asyncio.get_event_loop()

    def _update_job(jid, **kwargs):
        with _jobs_lock:
            _jobs[jid].update(kwargs)

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

            def _do_process():
                # Local models share weights — serialise access.
                # Azure requests are pure HTTP and can run in parallel.
                uses_local = processing_mode in ("local", "hybrid")
                if uses_local:
                    with _local_model_lock:
                        return _processor.process(str(upload_path), opts)
                return _processor.process(str(upload_path), opts)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, _do_process)

            if result.success:
                doc_dict = result.document.model_dump(mode="json")
                _update_job(
                    job_id,
                    status="completed",
                    result=doc_dict,
                    artifacts_dir=str(
                        ARTIFACTS_BASE / mode_subfolder / Path(result.document.metadata.filename).stem
                    ),
                )
                # Log cost summary after Azure processing
                if processing_mode == "azure":
                    _processor.print_cost_summary()
            else:
                _update_job(job_id, status="failed", error=result.error)
        except Exception as exc:
            logger.exception("Processing failed")
            _update_job(job_id, status="failed", error=str(exc))

    asyncio.ensure_future(_run())
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# API: batch upload + process
# ---------------------------------------------------------------------------
from fastapi import Body


@app.post("/api/process/batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    processing_mode: str = Form("local"),
    document_type: str = Form("auto"),
):
    """Upload multiple documents, process each, return list of job_ids."""
    if _processor is None:
        raise HTTPException(503, "Service not initialised")

    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    job_ids = []

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in allowed:
            continue  # skip unsupported

        job_id = str(uuid.uuid4())[:12]
        content = await file.read()

        content_hash = hashlib.sha256(content).hexdigest()[:16]
        upload_path = UPLOAD_DIR / f"{content_hash}{suffix}"
        if not upload_path.exists():
            upload_path.write_bytes(content)

        mode_subfolder = "Azure_Cloud" if processing_mode == "azure" else "Local"

        _jobs[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "processing_mode": processing_mode,
"created": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "result": None,
            "error": None,
            "artifacts_dir": None,
        }
        job_ids.append(job_id)

        # Schedule processing
        import concurrent.futures
        loop = asyncio.get_event_loop()
        _upload_path = str(upload_path)
        _mode = processing_mode
        _dtype = document_type
        _jid = job_id
        _msub = mode_subfolder

        def _update_job_batch(jid, **kwargs):
            with _jobs_lock:
                _jobs[jid].update(kwargs)

        async def _run_batch(jid=_jid, upath=_upload_path, mode=_mode, dtype=_dtype, msub=_msub):
            _update_job_batch(jid, status="processing")
            try:
                opts = ProcessingOptions(
                    processing_mode=mode,
                    document_type=dtype,
                    save_artifacts=True,
                    save_json=False,
                )

                def _do_process():
                    uses_local = mode in ("local", "hybrid")
                    if uses_local:
                        with _local_model_lock:
                            return _processor.process(upath, opts)
                    return _processor.process(upath, opts)

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = await loop.run_in_executor(pool, _do_process)

                if result.success:
                    doc_dict = result.document.model_dump(mode="json")
                    _update_job_batch(
                        jid,
                        status="completed",
                        result=doc_dict,
                        artifacts_dir=str(
                            ARTIFACTS_BASE / msub / Path(result.document.metadata.filename).stem
                        ),
                    )
                else:
                    _update_job_batch(jid, status="failed", error=result.error)
            except Exception as exc:
                logger.exception(f"Batch processing failed for {jid}")
                _update_job_batch(jid, status="failed", error=str(exc))

        asyncio.ensure_future(_run_batch())

    return {"job_ids": job_ids, "count": len(job_ids)}


# ---------------------------------------------------------------------------
# API: PDF thumbnail preview
# ---------------------------------------------------------------------------
@app.post("/api/preview")
async def preview_file(file: UploadFile = File(...)):
    """Return a base64 thumbnail of the first page of an uploaded file."""
    allowed = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    content = await file.read()

    try:
        from PIL import Image
        import io

        if suffix == ".pdf":
            # Try pdf2image first, fall back to fitz
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(content, first_page=1, last_page=1, dpi=120)
                img = images[0]
                page_count = len(convert_from_bytes(content, dpi=30))
            except ImportError:
                try:
                    import fitz
                    doc = fitz.open(stream=content, filetype="pdf")
                    page_count = len(doc)
                    pix = doc[0].get_pixmap(dpi=120)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    doc.close()
                except ImportError:
                    return {"preview": None, "pages": 0, "error": "No PDF renderer available"}
        else:
            img = Image.open(io.BytesIO(content))
            page_count = 1

        # Create thumbnail
        img.thumbnail((400, 500), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "preview": f"data:image/png;base64,{b64}",
            "pages": page_count,
            "width": img.width,
            "height": img.height,
        }
    except Exception as e:
        logger.warning(f"Preview generation failed: {e}")
        return {"preview": None, "pages": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# API: history (list all jobs)
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def list_history():
    """Return a list of all jobs for the history panel."""
    jobs_list = []
    for jid, job in _jobs.items():
        entry = {
            "job_id": jid,
            "filename": job["filename"],
            "processing_mode": job.get("processing_mode", "local"),
            "status": job["status"],
            "created": job["created"],
            "has_result": job["result"] is not None,
        }
        # Add summary stats if result exists
        if job["result"]:
            r = job["result"]
            entry["page_count"] = r.get("page_count", 0)
            entry["text_lines"] = sum(
                len(p.get("text_lines", [])) for p in r.get("pages", [])
            )
            entry["tables"] = len(r.get("tables", []))
            entry["fields"] = len(r.get("fields", []))
            entry["processing_time"] = r.get("metadata", {}).get(
                "processing_time_seconds", 0
            )
        jobs_list.append(entry)

    # Newest first
    jobs_list.sort(key=lambda x: x["created"], reverse=True)
    return {"jobs": jobs_list, "total": len(jobs_list)}


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
    filename = f"{stem}.json"
    filepath = out_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(job["result"], f, indent=2, ensure_ascii=False, default=str)

    # Also generate Markdown report
    md_path = save_markdown(
        data=job["result"],
        output_dir="markdown",
        processing_mode=mode,
        filename_stem=stem,
    )

    logger.info(f"Result saved to disk: {filepath}")
    logger.info(f"Markdown report saved: {md_path}")
    return {"ok": True, "path": str(filepath), "markdown_path": md_path}


@app.get("/api/jobs/{job_id}/download/markdown")
async def download_markdown(job_id: str):
    """Download the OCR result as a Markdown report."""
    if job_id not in _jobs or _jobs[job_id]["result"] is None:
        raise HTTPException(404, "No result")
    job = _jobs[job_id]
    md_content = generate_markdown(job["result"])
    filename = Path(job["filename"]).stem + "_report.md"
    from fastapi.responses import Response
    return Response(
        content=md_content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
