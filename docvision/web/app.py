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
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timezone
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

# Max upload size: 100 MB
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_processor: Optional[DocumentProcessor] = None
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()           # protects _jobs dict mutations
_local_model_lock = threading.Lock()    # serialises local-model inference
_background_tasks: Set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks
_shared_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None

# Job eviction settings
_JOB_MAX_AGE_SECS = 3600      # evict completed/failed jobs older than 1 hour
_JOB_MAX_COUNT = 200           # hard cap on total in-memory jobs


def _evict_old_jobs() -> None:
    """Remove stale completed/failed jobs to prevent unbounded memory growth."""
    now = time.time()
    with _jobs_lock:
        # Phase 1: remove completed/failed jobs older than max age
        stale = [
            jid for jid, j in _jobs.items()
            if j["status"] in ("completed", "failed")
            and (now - j.get("_created_ts", now)) > _JOB_MAX_AGE_SECS
        ]
        for jid in stale:
            del _jobs[jid]
        if stale:
            logger.debug(f"Evicted {len(stale)} stale jobs")

        # Phase 2: if still over cap, drop oldest completed jobs
        if len(_jobs) > _JOB_MAX_COUNT:
            completed = sorted(
                [(jid, j) for jid, j in _jobs.items() if j["status"] == "completed"],
                key=lambda x: x[1].get("_created_ts", 0),
            )
            to_drop = len(_jobs) - _JOB_MAX_COUNT
            for jid, _ in completed[:to_drop]:
                del _jobs[jid]
            if to_drop > 0:
                logger.debug(f"Evicted {min(to_drop, len(completed))} jobs (over cap)")

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "docvision_uploads"
ARTIFACTS_BASE = Path("artifacts")
OUTPUT_BASE = Path("output")


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

    # Pre-resolve Azure hostnames via DNS-over-HTTPS (bypasses VPN private link issues)
    if config.azure.processing_mode in ("azure", "hybrid") or config.azure.is_azure_ready:
        try:
            from docvision.dns_config import configure_doh_for_azure
            configure_doh_for_azure(
                di_endpoint=config.azure.doc_intelligence_endpoint,
                openai_endpoint=config.azure.openai_endpoint,
            )
        except Exception as exc:
            logger.warning(f"DoH DNS setup skipped: {exc}")

    _processor = DocumentProcessor(config)
    global _shared_pool
    _shared_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_BASE.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    logger.info("DocVision Web UI ready")
    yield
    logger.info("Shutting down DocVision Web UI")
    # Cancel all in-flight background tasks
    for task in _background_tasks:
        task.cancel()
    _shared_pool.shutdown(wait=False)
    _shared_pool = None
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
    cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Serve static assets (CSS / JS)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/load balancer."""
    return {"status": "healthy", "version": __version__}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page UI."""
    html_path = WEB_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Auto-save helper
# ---------------------------------------------------------------------------
def _auto_save_result(doc_dict: dict, original_filename: str, processing_mode: str) -> Optional[str]:
    """
    Automatically save the processing result to the output folder.
    
    Args:
        doc_dict: The document dictionary to save
        original_filename: The original filename from upload (not the hash-based temp name)
        processing_mode: 'local', 'azure', or 'hybrid'
        
    Returns:
        Path to the saved file, or None if save failed
    """
    try:
        subfolder = "Azure_Cloud" if processing_mode == "azure" else "Local"
        out_dir = OUTPUT_BASE / subfolder
        out_dir.mkdir(parents=True, exist_ok=True)
        
        stem = Path(original_filename).stem
        filename = f"{stem}.json"
        filepath = out_dir / filename
        
        # Add reconstruction data
        from docvision.io.reconstruction import add_reconstruction_to_document
        data = add_reconstruction_to_document(doc_dict)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Auto-saved result to: {filepath}")
        
        # Also generate Markdown report
        md_path = save_markdown(
            data=data,
            output_dir="markdown",
            processing_mode=processing_mode,
            filename_stem=stem,
        )
        logger.info(f"Auto-saved Markdown report: {md_path}")
        
        return str(filepath)
    except Exception as exc:
        logger.error(f"Auto-save failed: {exc}")
        return None


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

    # Enforce upload size limit
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large ({len(content) / 1024 / 1024:.0f} MB). Max is {_MAX_UPLOAD_BYTES // 1024 // 1024} MB.")

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

    _evict_old_jobs()

    _jobs[job_id] = {
        "status": "processing",
        "filename": file.filename,
        "processing_mode": processing_mode,
        "created": datetime.now(timezone.utc).isoformat(),
        "_created_ts": time.time(),
        "result": None,
        "error": None,
        "artifacts_dir": None,
        "progress": {"stage": "Uploading", "percent": 0},
    }

    # Run synchronously in a thread so the event loop stays responsive
    loop = asyncio.get_running_loop()

    def _update_job(jid, **kwargs):
        with _jobs_lock:
            _jobs[jid].update(kwargs)

    def _progress_cb(stage: str, percent: int):
        _update_job(job_id, progress={"stage": stage, "percent": percent})

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
                progress_callback=_progress_cb,
            )

            def _do_process():
                # Local models share weights — serialise access.
                # Azure requests are pure HTTP and can run in parallel.
                uses_local = processing_mode in ("local", "hybrid")
                if uses_local:
                    with _local_model_lock:
                        return _processor.process(str(upload_path), opts)
                return _processor.process(str(upload_path), opts)

            result = await loop.run_in_executor(_shared_pool, _do_process)

            # Check if cancelled while processing
            if _jobs.get(job_id, {}).get("status") == "cancelled":
                logger.info(f"Job {job_id} was cancelled — discarding result")
                return

            if result.success:
                doc_dict = result.document.model_dump(mode="json")
                artifacts_dir_path = ARTIFACTS_BASE / mode_subfolder / Path(result.document.metadata.filename).stem
                _update_job(
                    job_id,
                    status="completed",
                    result=doc_dict,
                    artifacts_dir=str(artifacts_dir_path),
                )
                # Auto-save JSON to output folder
                _auto_save_result(
                    doc_dict=doc_dict,
                    original_filename=file.filename,
                    processing_mode=processing_mode,
                )
                # Log cost summary after Azure processing
                if processing_mode == "azure":
                    _processor.print_cost_summary()
            else:
                _update_job(job_id, status="failed", error=result.error)
        except Exception as exc:
            logger.exception("Processing failed")
            _update_job(job_id, status="failed", error=str(exc))

    task = asyncio.create_task(_run())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return {"job_id": job_id}


# ---------------------------------------------------------------------------
# API: batch upload + process
# ---------------------------------------------------------------------------


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

        # Enforce upload size limit
        if len(content) > _MAX_UPLOAD_BYTES:
            continue  # skip oversized files

        content_hash = hashlib.sha256(content).hexdigest()[:16]
        upload_path = UPLOAD_DIR / f"{content_hash}{suffix}"
        if not upload_path.exists():
            upload_path.write_bytes(content)

        mode_subfolder = "Azure_Cloud" if processing_mode == "azure" else "Local"

        _evict_old_jobs()

        _jobs[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "processing_mode": processing_mode,
            "created": datetime.now(timezone.utc).isoformat(),
            "_created_ts": time.time(),
            "result": None,
            "error": None,
            "artifacts_dir": None,
            "progress": {"stage": "Queued", "percent": 0},
        }
        job_ids.append(job_id)

        # Schedule processing
        loop = asyncio.get_running_loop()
        _upload_path = str(upload_path)
        _mode = processing_mode
        _dtype = document_type
        _jid = job_id
        _msub = mode_subfolder

        def _update_job_batch(jid, **kwargs):
            with _jobs_lock:
                _jobs[jid].update(kwargs)

        def _make_batch_progress_cb(jid):
            def _cb(stage, percent):
                _update_job_batch(jid, progress={"stage": stage, "percent": percent})
            return _cb

        async def _run_batch(jid=_jid, upath=_upload_path, mode=_mode, dtype=_dtype, msub=_msub):
            _update_job_batch(jid, status="processing")
            try:
                opts = ProcessingOptions(
                    processing_mode=mode,
                    document_type=dtype,
                    save_artifacts=True,
                    save_json=False,
                    progress_callback=_make_batch_progress_cb(jid),
                )

                def _do_process():
                    uses_local = mode in ("local", "hybrid")
                    if uses_local:
                        with _local_model_lock:
                            return _processor.process(upath, opts)
                    return _processor.process(upath, opts)

                result = await loop.run_in_executor(_shared_pool, _do_process)

                # Check if cancelled while processing
                if _jobs.get(jid, {}).get("status") == "cancelled":
                    logger.info(f"Batch job {jid} was cancelled — discarding result")
                    return

                if result.success:
                    doc_dict = result.document.model_dump(mode="json")
                    artifacts_dir_path = ARTIFACTS_BASE / msub / Path(result.document.metadata.filename).stem
                    _update_job_batch(
                        jid,
                        status="completed",
                        result=doc_dict,
                        artifacts_dir=str(artifacts_dir_path),
                    )
                    # Auto-save JSON to output folder
                    _auto_save_result(
                        doc_dict=doc_dict,
                        original_filename=_jobs[jid]["filename"],
                        processing_mode=mode,
                    )
                else:
                    _update_job_batch(jid, status="failed", error=result.error)
            except Exception as exc:
                logger.exception(f"Batch processing failed for {jid}")
                _update_job_batch(jid, status="failed", error=str(exc))

        task = asyncio.create_task(_run_batch())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

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
        "progress": job.get("progress"),
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Mark a running job as cancelled."""
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = _jobs[job_id]
    if job["status"] in ("processing", "queued"):
        with _jobs_lock:
            _jobs[job_id]["status"] = "cancelled"
            _jobs[job_id]["error"] = "Cancelled by user"
        logger.info(f"Job {job_id} cancelled by user")
        return {"ok": True}
    return {"ok": False, "reason": f"Job is already {job['status']}"}


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
# API: save result to disk (manual save, used by Save button)
# ---------------------------------------------------------------------------


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
    if _processor.has_cost_tracker:
        _processor.cost_tracker.reset()
    return {"ok": True}


@app.post("/api/cache/clear")
async def clear_cache():
    """Clear the Azure response cache."""
    if _processor is None:
        raise HTTPException(503, "Service not initialised")
    count = 0
    if _processor.has_response_cache:
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
    # Prevent path traversal — resolve and verify path is under artifacts_dir
    base = Path(artifacts_dir).resolve()
    fpath = (base / filename).resolve()
    if not str(fpath).startswith(str(base)):
        raise HTTPException(403, "Forbidden")
    if not fpath.is_file():
        raise HTTPException(404)
    return FileResponse(fpath, media_type="image/png")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    print(f"\n  DocVision Web UI -> http://localhost:{port}\n")
    uvicorn.run(
        "docvision.web.app:app",
        host="127.0.0.1",
        port=port,
        reload=False,
    )
