"""
FastAPI server for DocVision document processing.

Provides REST API endpoints for:
- Document processing (POST /process)
- Health check (GET /health)
- Version info (GET /version)
- Async job processing (POST /process/async, GET /jobs/{job_id})
"""

import os
import uuid
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from docvision import __version__
from docvision.config import load_config, Config
from docvision.pipeline import DocumentProcessor, ProcessingOptions


# Global state for processor and jobs
_processor: Optional[DocumentProcessor] = None
_jobs: Dict[str, Dict[str, Any]] = {}


class ProcessRequest(BaseModel):
    """Request body for process endpoint."""
    preprocess: bool = Field(default=True, description="Run preprocessing")
    detect_layout: bool = Field(default=True, description="Detect layout regions")
    detect_text: bool = Field(default=True, description="Detect text regions")
    detect_tables: bool = Field(default=True, description="Detect tables")
    run_ocr: bool = Field(default=True, description="Run OCR recognition")
    run_donut: bool = Field(default=True, description="Run Donut KIE")
    run_layoutlmv3: bool = Field(default=True, description="Run LayoutLMv3 KIE")
    run_validators: bool = Field(default=True, description="Run field validators")
    save_artifacts: bool = Field(default=True, description="Save debug artifacts")


class JobStatus(BaseModel):
    """Job status response."""
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_loaded: Dict[str, bool]
    device: str


class VersionResponse(BaseModel):
    """Version info response."""
    version: str
    api_version: str
    python_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _processor
    
    # Startup
    logger.info("Starting DocVision API server...")
    
    # Load config
    config_path = os.environ.get("DOCVISION_CONFIG")
    if config_path and Path(config_path).exists():
        config = load_config(config_path)
    else:
        config = Config()
    
    # Initialize processor
    _processor = DocumentProcessor(config)
    logger.info("DocumentProcessor initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down DocVision API server...")
    _processor = None


def create_app(config: Optional[Config] = None) -> FastAPI:
    """Create FastAPI application with optional config."""
    global _processor
    
    app = FastAPI(
        title="DocVision API",
        description="Document AI processing service for extracting structured data from documents",
        version=__version__,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


# Create default app
app = create_app()


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    global _processor
    
    if _processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Check which models are loaded
    models_loaded = {
        "layout_detector": _processor._layout_detector is not None,
        "text_detector": _processor._text_detector is not None,
        "table_detector": _processor._table_detector is not None,
        "trocr": _processor._trocr is not None,
        "tesseract": _processor._tesseract is not None,
        "donut": _processor._donut is not None,
        "layoutlmv3": _processor._layoutlmv3 is not None,
    }
    
    return HealthResponse(
        status="healthy",
        version=__version__,
        models_loaded=models_loaded,
        device=str(_processor.device)
    )


@app.get("/version", response_model=VersionResponse, tags=["System"])
async def version_info():
    """Get version information."""
    import sys
    
    return VersionResponse(
        version=__version__,
        api_version="1.0.0",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


@app.post("/process", tags=["Processing"])
async def process_document(
    file: UploadFile = File(..., description="PDF or image file to process"),
    processing_mode: str = Query("local", description="Processing mode: local, azure, or hybrid"),
    document_type: str = Query("auto", description="Document type: auto, bol, invoice, receipt, delivery_ticket"),
    preprocess: bool = Query(True, description="Run preprocessing"),
    detect_layout: bool = Query(True, description="Detect layout regions"),
    detect_text: bool = Query(True, description="Detect text regions"),
    detect_tables: bool = Query(True, description="Detect tables"),
    run_ocr: bool = Query(True, description="Run OCR recognition"),
    run_donut: bool = Query(True, description="Run Donut KIE"),
    run_layoutlmv3: bool = Query(True, description="Run LayoutLMv3 KIE"),
    run_validators: bool = Query(True, description="Run field validators"),
    save_artifacts: bool = Query(False, description="Save debug artifacts"),
):
    """
    Process a document and extract structured information.
    
    Upload a PDF or image file and get back structured JSON with:
    - Extracted fields with confidence scores
    - Tables with cell contents
    - Text lines with bounding boxes
    - Layout regions
    - Validation results
    """
    global _processor
    
    if _processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate file type
    allowed_types = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]
    suffix = Path(file.filename).suffix.lower()
    
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_types}"
        )
    
    # Save to temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Create processing options
        options = ProcessingOptions(
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
            save_artifacts=save_artifacts,
            save_json=False  # Don't save to disk, return response
        )
        
        # Process document
        result = _processor.process(tmp_path, options)
        
        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)
        
        # Return document as JSON
        return JSONResponse(
            content=result.document.model_dump(mode="json"),
            media_type="application/json"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/process/async", response_model=JobStatus, tags=["Processing"])
async def process_document_async(
    file: UploadFile = File(..., description="PDF or image file to process"),
    background_tasks: BackgroundTasks = None,
):
    """
    Start async document processing job.
    
    Returns a job ID that can be used to check status and get results.
    """
    global _processor, _jobs
    
    if _processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate file type
    allowed_types = [".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"]
    suffix = Path(file.filename).suffix.lower()
    
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_types}"
        )
    
    # Create job
    job_id = str(uuid.uuid4())
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    # Initialize job status
    _jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "result": None,
        "error": None,
        "tmp_path": tmp_path
    }
    
    # Start background processing
    background_tasks.add_task(_process_job, job_id, tmp_path)
    
    return JobStatus(
        job_id=job_id,
        status="pending",
        created_at=_jobs[job_id]["created_at"]
    )


async def _process_job(job_id: str, file_path: str):
    """Background job processor."""
    global _processor, _jobs
    
    _jobs[job_id]["status"] = "processing"
    
    try:
        # Process in thread pool to avoid blocking
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as pool:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                pool,
                lambda: _processor.process(file_path, ProcessingOptions(save_json=False))
            )
        
        if result.success:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result.document.model_dump(mode="json")
        else:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = result.error
        
        _jobs[job_id]["completed_at"] = datetime.utcnow()
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        _jobs[job_id]["completed_at"] = datetime.utcnow()
    
    finally:
        # Clean up temp file
        try:
            os.unlink(file_path)
        except Exception:
            pass


@app.get("/jobs/{job_id}", response_model=JobStatus, tags=["Processing"])
async def get_job_status(job_id: str):
    """Get status of an async processing job."""
    global _jobs
    
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job["completed_at"],
        result=job["result"],
        error=job["error"]
    )


@app.delete("/jobs/{job_id}", tags=["Processing"])
async def delete_job(job_id: str):
    """Delete a completed job."""
    global _jobs
    
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del _jobs[job_id]
    
    return {"message": "Job deleted"}


@app.get("/models", tags=["System"])
async def list_models():
    """List available models and their status."""
    global _processor
    
    if _processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "layout": {
            "model": _processor.config.models.layout or "DocLayNet YOLO",
            "loaded": _processor._layout_detector is not None
        },
        "text_detection": {
            "model": _processor.config.models.craft or "CRAFT",
            "loaded": _processor._text_detector is not None
        },
        "table": {
            "model": _processor.config.models.tatr,
            "loaded": _processor._table_detector is not None
        },
        "ocr_printed": {
            "model": _processor.config.models.trocr_printed,
            "loaded": _processor._trocr is not None
        },
        "ocr_handwritten": {
            "model": _processor.config.models.trocr_handwritten,
            "loaded": _processor._trocr is not None
        },
        "donut": {
            "model": _processor.config.models.donut,
            "loaded": _processor._donut is not None
        },
        "layoutlmv3": {
            "model": _processor.config.models.layoutlmv3,
            "loaded": _processor._layoutlmv3 is not None
        }
    }


# Run with uvicorn
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "docvision.api.server:app",
        host=host,
        port=port,
        reload=False,
        workers=1
    )
