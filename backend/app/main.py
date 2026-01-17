"""
Document Translator - FastAPI Backend
With GPU detection and provider switching support.
"""

import asyncio
import os
import shutil
import uuid
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

# Suppress torch warnings about pin_memory (harmless when no GPU)
warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
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
from app.utils.security import SecurityHeadersMiddleware, sanitize_filename
from app.utils.validators import (
    validate_device,
    validate_file_extension,
    validate_file_size,
    validate_file_content,
    validate_language_code,
)

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

settings = get_settings()


def detect_gpu() -> dict:
    """Detect GPU availability and return system info."""
    gpu_info = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cuda_version": None,
        "using_device": "cpu"
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["gpu_available"] = True
            gpu_info["gpu_name"] = torch.cuda.get_device_name(0)
            gpu_info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["using_device"] = "cuda"
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"GPU detection error: {e}")
    
    return gpu_info


# Detect GPU at startup
GPU_INFO = detect_gpu()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info(
        "Starting Document Translator", 
        mode=settings.translation_mode.value,
        gpu_available=GPU_INFO["gpu_available"],
        device=GPU_INFO["using_device"]
    )
    
    for dir_path in [settings.upload_dir, settings.output_dir, settings.temp_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    await init_db()
    
    yield
    
    logger.info("Shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-grade document translation service",
    lifespan=lifespan
)

# Security headers middleware (add first)
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:8000",
        *settings.cors_origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Service instances
ocr_service = OCRService()
translation_service = TranslationService()
job_processor = JobProcessor()

# Job status storage
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
    translation_provider: Optional[str],
    device: str = "auto"
):
    """Background task to process document."""
    try:
        # Device is validated in the endpoint - actual GPU usage is determined by provider
        result = await job_processor.process_job(
            job_id=job_id,
            input_path=input_path,
            source_lang=source_lang,
            target_lang=target_lang,
            ocr_provider=ocr_provider,
            translation_provider=translation_provider,
            status_callback=update_job_status
        )
        
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
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        return HealthResponse(
            status="healthy",
            version=settings.app_version,
            mode=settings.translation_mode.value,
            ocr_available=len(ocr_service.get_available_providers()) > 0,
            translation_available=len(translation_service.get_available_providers()) > 0
        )
    except Exception as e:
        logger.error("Health check error", error=str(e))
        return HealthResponse(
            status="degraded",
            version=settings.app_version,
            mode=settings.translation_mode.value,
            ocr_available=False,
            translation_available=False
        )


@app.get("/api/v1/system-info", tags=["Config"])
async def get_system_info():
    """
    Get system information including GPU availability and speed estimates.
    Frontend uses this to show warnings and provider options.
    """
    # Check if Ollama is available with configured model
    ollama_available = False
    ollama_model = settings.ollama_model  # Get configured model name
    try:
        from app.config import TranslationProvider
        ollama_translator = translation_service.providers.get(TranslationProvider.OLLAMA)
        if ollama_translator:
            ollama_available = ollama_translator.is_available()
    except Exception:
        pass

    # Check if DeepSeek API is configured
    deepseek_available = bool(settings.deepseek_api_key)
    claude_available = bool(settings.anthropic_api_key)

    # Calculate speed estimates based on hardware
    if GPU_INFO["gpu_available"]:
        nllb_speed = "fast"
        nllb_estimate = "2-5 minutes for 10 pages"
        speed_warning = None
    else:
        nllb_speed = "slow"
        nllb_estimate = "15-30 minutes for 10 pages"
        speed_warning = "No GPU detected. Translation using the free NLLB model will be slow on CPU. Consider using DeepSeek API for faster results (~$0.01 per page)."

    # Determine recommendation based on GPU availability
    if GPU_INFO["gpu_available"]:
        # GPU detected - free local models will be fast
        if ollama_available:
            recommendation = "ollama"
            speed_warning = None  # GPU + Ollama = great combo
            recommendation_reason = f"GPU detected! Using FREE Ollama with {ollama_model} (fast, high accuracy)"
        else:
            recommendation = "nllb"
            speed_warning = None  # GPU + NLLB = good
            recommendation_reason = "GPU detected! Using FREE NLLB with CUDA acceleration"
    else:
        # No GPU - warn about slow performance, recommend API
        if deepseek_available:
            recommendation = "deepseek"
            speed_warning = "No GPU detected. Free local models (Ollama/NLLB) will run on CPU and be SLOW. We recommend using DeepSeek API for faster results (~$0.01 per page)."
            recommendation_reason = "No GPU - DeepSeek API recommended for faster translation"
        elif ollama_available:
            recommendation = "ollama"
            speed_warning = f"No GPU detected. Ollama/{ollama_model} will run on CPU (slower). Consider setting DEEPSEEK_API_KEY for faster results."
            recommendation_reason = "No GPU - Ollama available but will be slow on CPU"
        else:
            recommendation = "nllb"
            speed_warning = "No GPU detected. NLLB will run on CPU (VERY SLOW). Strongly recommend setting DEEPSEEK_API_KEY for faster translation (~$0.01 per page)."
            recommendation_reason = "No GPU - NLLB will be very slow on CPU"

    # Device recommendation: suggest GPU if available, otherwise CPU
    device_recommendation = "gpu" if GPU_INFO["gpu_available"] else "cpu"
    
    return {
        "gpu": GPU_INFO,
        "device_recommendation": device_recommendation,
        "providers": {
            "ollama": {
                "available": ollama_available,
                "speed": "medium",
                "estimate": "5-10 minutes for 10 pages",
                "cost": "Free",
                "model": settings.ollama_model
            },
            "nllb": {
                "available": True,
                "speed": nllb_speed,
                "estimate": nllb_estimate,
                "cost": "Free",
                "device": GPU_INFO["using_device"]
            },
            "deepseek": {
                "available": deepseek_available,
                "speed": "very_fast",
                "estimate": "30 seconds - 2 minutes for 10 pages",
                "cost": "~$0.01 per page (~$0.10 for 10 pages)"
            },
            "claude": {
                "available": claude_available,
                "speed": "very_fast",
                "estimate": "30 seconds - 2 minutes for 10 pages",
                "cost": "~$0.05 per page (~$0.50 for 10 pages)"
            }
        },
        "speed_warning": speed_warning,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason
    }


@app.get("/api/v1/providers", tags=["Config"])
async def get_providers():
    """Get available OCR and translation providers with speed info."""
    ocr_default, trans_default = get_providers_for_mode(settings.translation_mode)
    
    # Add speed indicators
    translation_providers = []
    for provider in translation_service.get_available_providers():
        speed = "slow"
        if provider in ["deepseek", "claude", "gpt4o"]:
            speed = "fast"
        elif provider == "nllb" and GPU_INFO["gpu_available"]:
            speed = "medium"
        
        translation_providers.append({
            "name": provider,
            "speed": speed,
            "cost": "free" if provider in ["nllb", "ollama"] else "paid"
        })
    
    return {
        "mode": settings.translation_mode.value,
        "gpu_available": GPU_INFO["gpu_available"],
        "ocr": {
            "available": ocr_service.get_available_providers(),
            "default": ocr_default.value
        },
        "translation": {
            "available": translation_providers,
            "default": trans_default.value
        }
    }


@app.get("/api/v1/languages", tags=["Config"])
async def get_languages():
    """Get supported languages."""
    return {"languages": SUPPORTED_LANGUAGES}


@app.post("/api/v1/translate", response_model=TranslationResponse, tags=["Translation"])
async def translate_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    source_language: Optional[str] = Form(None),
    target_language: str = Form("en"),
    ocr_provider: Optional[str] = Form(None),
    translation_provider: Optional[str] = Form(None),
    device: Optional[str] = Form("auto")
):
    """
    Upload and translate a document.
    
    Args:
        file: Document file to translate (PDF, DOCX, images)
        source_language: Source language code (auto-detect if not provided)
        target_language: Target language code (default: "en")
        ocr_provider: OCR provider to use (optional)
        translation_provider: Translation provider to use (optional)
        device: Processing device ("cpu", "gpu", or "auto")
        
    Returns:
        TranslationResponse with job_id and estimated time
    """
    try:
        # Validate device selection
        device = validate_device(device or "auto")
        
        # If GPU is requested but not available, reject or suggest alternatives
        if device == "gpu" and not GPU_INFO["gpu_available"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "GPU requested but not available",
                    "gpu_available": False,
                    "suggestion": "Please select 'cpu' device or use DeepSeek API for faster results",
                    "alternatives": ["cpu", "deepseek"]
                }
            )
        
        # Auto-select device: use GPU if available, otherwise CPU
        if device == "auto":
            device = "gpu" if GPU_INFO["gpu_available"] else "cpu"
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Sanitize filename to prevent path traversal attacks
        safe_filename = sanitize_filename(file.filename)
        file_ext = validate_file_extension(safe_filename)
        
        # Validate language codes
        if source_language:
            source_language = validate_language_code(source_language, set(SUPPORTED_LANGUAGES.keys()))
        target_language = validate_language_code(target_language, set(SUPPORTED_LANGUAGES.keys()))
        
        # Save file
        job_id = str(uuid.uuid4())
        upload_dir = Path(settings.upload_dir)
        input_path = upload_dir / f"{job_id}{file_ext}"
        
        try:
            # Save file
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except IOError as e:
            logger.error("File save error", error=str(e), filename=safe_filename)
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
        
        # Validate file size
        file_size = input_path.stat().st_size
        validate_file_size(file_size, settings.max_file_size_mb)
        
        # Validate file content (basic check)
        try:
            validate_file_content(input_path, file_ext)
        except HTTPException:
            # Clean up invalid file
            try:
                input_path.unlink()
            except Exception:
                pass
            raise
        
        # Determine time estimate based on provider
        if translation_provider == "deepseek":
            estimated_time = "1-3 minutes"
        elif translation_provider == "claude":
            estimated_time = "1-3 minutes"
        elif translation_provider == "ollama":
            estimated_time = f"5-15 minutes ({settings.ollama_model} - high accuracy)"
        elif GPU_INFO["gpu_available"]:
            estimated_time = "3-8 minutes"
        else:
            estimated_time = "15-30 minutes (CPU mode - consider using DeepSeek for faster results)"
        
        # Initialize job storage
        job_storage[job_id] = {
            "status": JobStatus.PENDING,
            "progress": 0,
            "message": "Document uploaded. Starting processing...",
            "filename": safe_filename,
            "source_language": source_language,
            "target_language": target_language,
            "translation_provider": translation_provider or "ollama",
            "device": device,
            "created_at": datetime.utcnow().isoformat(),
            "input_path": str(input_path),
            "current_page": 0,
            "total_pages": 0,
            "processed_chunks": 0,
            "total_chunks": 0,
            "gpu_available": GPU_INFO["gpu_available"],
        }
        
        background_tasks.add_task(
            process_document,
            job_id,
            str(input_path),
            source_language,
            target_language,
            ocr_provider,
            translation_provider,
            device
        )
        
        return TranslationResponse(
            job_id=job_id,
            message="Document uploaded. Translation started.",
            estimated_time=estimated_time,
            mode=settings.translation_mode.value
        )
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors, etc.)
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error("Unexpected error in translate_document", error=str(e))
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/v1/status/{job_id}", tags=["Translation"])
async def get_job_status(job_id: str):
    """Get the status of a translation job."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    
    return {
        "job_id": job_id,
        "status": job.get("status", JobStatus.PENDING),
        "progress": job.get("progress", 0),
        "message": job.get("message"),
        "filename": job.get("filename"),
        "source_language": job.get("source_language"),
        "target_language": job.get("target_language", "en"),
        "translation_provider": job.get("translation_provider"),
        "device": job.get("device", "auto"),
        "current_page": job.get("current_page", 0),
        "total_pages": job.get("total_pages", 0),
        "processed_chunks": job.get("processed_chunks", 0),
        "total_chunks": job.get("total_chunks", 0),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
        "pages_translated": job.get("pages_translated", 0),
        "pages_skipped": job.get("pages_skipped", 0),
    }


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
    
    # Determine output filename based on file extension
    if output_path.endswith('.docx'):
        output_filename = f"{Path(filename).stem}_translated.docx"
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        output_filename = f"{Path(filename).stem}_translated.txt"
        media_type = "text/plain"
    
    return FileResponse(
        path=output_path,
        filename=output_filename,
        media_type=media_type
    )


@app.post("/api/v1/translate/quick", response_model=QuickTranslationResponse, tags=["Translation"])
async def quick_translate(request: QuickTranslationRequest):
    """Quick translation for small text snippets."""
    source_lang = request.source_language
    if not source_lang:
        source_lang = await translation_service.detect_language(request.text)
    
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
    
    for path_key in ["input_path", "output_path"]:
        path = job.get(path_key)
        if path and Path(path).exists():
            Path(path).unlink()
    
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
