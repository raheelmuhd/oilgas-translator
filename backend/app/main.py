"""
Oil & Gas Document Translator - FastAPI Backend
Production-grade translation system for technical documents.
"""

import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import select, update

from app.config import TranslationMode, get_providers_for_mode, get_settings
from app.database import async_session, init_db
from app.models import (
    HealthResponse,
    JobStatus,
    JobStatusResponse,
    QuickTranslationRequest,
    QuickTranslationResponse,
    SUPPORTED_LANGUAGES,
    TranslationJob,
    TranslationResponse,
)
from app.services.job_processor import JobProcessor
from app.services.ocr_service import OCRService
from app.services.translation_service import TranslationService

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Oil & Gas Document Translator", mode=settings.translation_mode.value)
    
    # Create directories
    for dir_path in [settings.upload_dir, settings.output_dir, settings.temp_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    await init_db()
    
    yield
    
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade document translation for the oil & gas industry",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instances
ocr_service = OCRService()
translation_service = TranslationService()
job_processor = JobProcessor()


# Job status storage (in-memory for simplicity, use Redis in production)
job_storage = {}


async def update_job_status(
    job_id: str, 
    status: JobStatus, 
    progress: float, 
    message: str,
    extra: dict = None
):
    """Update job status in storage."""
    if job_id not in job_storage:
        job_storage[job_id] = {}
    
    job_storage[job_id].update({
        "status": status,
        "progress": progress,
        "message": message,
        "updated_at": datetime.utcnow().isoformat()
    })
    
    if extra:
        job_storage[job_id].update(extra)
    
    if status == JobStatus.COMPLETED:
        job_storage[job_id]["completed_at"] = datetime.utcnow().isoformat()


async def process_document(
    job_id: str,
    input_path: str,
    source_lang: Optional[str],
    target_lang: str,
    ocr_provider: Optional[str],
    translation_provider: Optional[str]
):
    """Background task to process document."""
    try:
        result = await job_processor.process_job(
            job_id=job_id,
            input_path=input_path,
            source_lang=source_lang,
            target_lang=target_lang,
            ocr_provider=ocr_provider,
            translation_provider=translation_provider,
            status_callback=update_job_status
        )
        
        # Update with output path
        job_storage[job_id]["output_path"] = result["output_path"]
        
    except Exception as e:
        logger.error("Document processing failed", job_id=job_id, error=str(e))
        await update_job_status(job_id, JobStatus.FAILED, 0, f"Error: {str(e)}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "mode": settings.translation_mode.value,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        mode=settings.translation_mode.value,
        ocr_available=len(ocr_service.get_available_providers()) > 0,
        translation_available=len(translation_service.get_available_providers()) > 0
    )


@app.get("/api/v1/providers", tags=["Config"])
async def get_providers():
    """Get available OCR and translation providers."""
    ocr_default, trans_default = get_providers_for_mode(settings.translation_mode)
    
    return {
        "mode": settings.translation_mode.value,
        "ocr": {
            "available": ocr_service.get_available_providers(),
            "default": ocr_default.value
        },
        "translation": {
            "available": translation_service.get_available_providers(),
            "default": trans_default.value
        }
    }


@app.get("/api/v1/languages", tags=["Config"])
async def get_languages():
    """Get supported languages."""
    return {
        "languages": SUPPORTED_LANGUAGES
    }


@app.post("/api/v1/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: Optional[str] = Form(None),
    target_language: str = Form("en"),
    ocr_provider: Optional[str] = Form(None),
    translation_provider: Optional[str] = Form(None)
):
    """
    Upload and translate a document.
    
    - **file**: Document file (PDF, DOCX, images, etc.)
    - **source_language**: Source language code (auto-detect if omitted)
    - **target_language**: Target language code (default: en)
    - **ocr_provider**: OCR provider (paddleocr, azure, gpt4o)
    - **translation_provider**: Translation provider (nllb, deepseek, claude)
    """
    # Validate file size
    max_size = settings.max_file_size_mb * 1024 * 1024
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_dir = Path(settings.upload_dir)
    file_ext = Path(file.filename).suffix
    input_path = upload_dir / f"{job_id}{file_ext}"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Check file size after saving
    if input_path.stat().st_size > max_size:
        input_path.unlink()
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB"
        )
    
    # Initialize job storage
    job_storage[job_id] = {
        "status": JobStatus.PENDING,
        "progress": 0,
        "message": "Document uploaded. Starting processing...",
        "filename": file.filename,
        "source_language": source_language,
        "target_language": target_language,
        "created_at": datetime.utcnow().isoformat(),
        "input_path": str(input_path)
    }
    
    # Start background processing
    background_tasks.add_task(
        process_document,
        job_id,
        str(input_path),
        source_language,
        target_language,
        ocr_provider,
        translation_provider
    )
    
    # Estimate time based on mode
    time_estimates = {
        TranslationMode.SELF_HOSTED: "5-15 minutes",
        TranslationMode.BUDGET: "3-8 minutes",
        TranslationMode.PREMIUM: "2-5 minutes"
    }
    
    return TranslationResponse(
        job_id=job_id,
        message="Document uploaded. Translation started.",
        estimated_time=time_estimates.get(settings.translation_mode, "5-10 minutes"),
        mode=settings.translation_mode.value
    )


@app.get("/api/v1/status/{job_id}", response_model=JobStatusResponse, tags=["Translation"])
async def get_job_status(job_id: str):
    """Get the status of a translation job."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", JobStatus.PENDING),
        progress=job.get("progress", 0),
        message=job.get("message"),
        filename=job.get("filename"),
        source_language=job.get("source_language"),
        target_language=job.get("target_language", "en"),
        total_pages=job.get("total_pages", 0),
        processed_chunks=job.get("processed_chunks", 0),
        total_chunks=job.get("total_chunks", 0),
        created_at=datetime.fromisoformat(job.get("created_at", datetime.utcnow().isoformat())),
        completed_at=datetime.fromisoformat(job["completed_at"]) if job.get("completed_at") else None
    )


@app.get("/api/v1/download/{job_id}", tags=["Translation"])
async def download_translation(job_id: str):
    """Download the translated document."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    if job.get("status") != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Translation not complete")
    
    output_path = job.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    filename = job.get("filename", "translation")
    output_filename = f"{Path(filename).stem}_translated.txt"
    
    return FileResponse(
        path=output_path,
        filename=output_filename,
        media_type="text/plain"
    )


@app.post("/api/v1/translate/quick", response_model=QuickTranslationResponse, tags=["Translation"])
async def quick_translate(request: QuickTranslationRequest):
    """
    Quick translation for small text snippets.
    No file upload, immediate response.
    """
    # Detect language if not specified
    source_lang = request.source_language
    if not source_lang:
        source_lang = await translation_service.detect_language(request.text)
    
    # Translate
    translated, provider = await translation_service.translate(
        request.text,
        source_lang,
        request.target_language
    )
    
    return QuickTranslationResponse(
        original_text=request.text,
        translated_text=translated,
        source_language=source_lang,
        target_language=request.target_language,
        provider=provider
    )


@app.get("/api/v1/jobs", tags=["Translation"])
async def list_jobs(limit: int = 20, status: Optional[str] = None):
    """List recent translation jobs."""
    jobs = []
    
    for job_id, job in list(job_storage.items())[-limit:]:
        if status and job.get("status") != status:
            continue
        
        jobs.append({
            "job_id": job_id,
            "filename": job.get("filename"),
            "status": job.get("status"),
            "progress": job.get("progress"),
            "created_at": job.get("created_at")
        })
    
    return {"jobs": jobs, "total": len(jobs)}


@app.delete("/api/v1/jobs/{job_id}", tags=["Translation"])
async def delete_job(job_id: str):
    """Delete a translation job and its files."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    # Delete files
    for path_key in ["input_path", "output_path"]:
        path = job.get(path_key)
        if path and Path(path).exists():
            Path(path).unlink()
    
    # Remove from storage
    del job_storage[job_id]
    
    return {"message": "Job deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )

