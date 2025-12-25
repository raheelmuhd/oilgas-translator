"""
Database models and Pydantic schemas for the translator.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class JobStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    TRANSLATING = "translating"
    COMPLETED = "completed"
    FAILED = "failed"


class TranslationJob(Base):
    """Database model for translation jobs."""
    __tablename__ = "translation_jobs"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    filename = Column(String(255), nullable=False)
    source_language = Column(String(10), nullable=True)
    target_language = Column(String(10), nullable=False, default="en")
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING)
    progress = Column(Float, default=0.0)
    message = Column(Text, nullable=True)
    
    # Provider info
    ocr_provider = Column(String(50), nullable=True)
    translation_provider = Column(String(50), nullable=True)
    
    # File paths
    input_path = Column(String(500), nullable=False)
    output_path = Column(String(500), nullable=True)
    
    # Metadata
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Cost tracking
    estimated_cost = Column(Float, default=0.0)
    actual_cost = Column(Float, default=0.0)


# Pydantic schemas for API
class TranslationRequest(BaseModel):
    """Request schema for translation."""
    source_language: Optional[str] = None
    target_language: str = "en"
    ocr_provider: Optional[str] = None
    translation_provider: Optional[str] = None


class TranslationResponse(BaseModel):
    """Response schema for translation initiation."""
    job_id: str
    message: str
    estimated_time: str
    mode: str


class JobStatusResponse(BaseModel):
    """Response schema for job status."""
    job_id: str
    status: JobStatus
    progress: float
    message: Optional[str]
    filename: Optional[str]
    source_language: Optional[str]
    target_language: str
    total_pages: int
    processed_chunks: int
    total_chunks: int
    created_at: datetime
    completed_at: Optional[datetime]


class QuickTranslationRequest(BaseModel):
    """Request for quick text translation."""
    text: str = Field(..., max_length=5000)
    source_language: Optional[str] = None
    target_language: str = "en"


class QuickTranslationResponse(BaseModel):
    """Response for quick text translation."""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    provider: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    mode: str
    ocr_available: bool
    translation_available: bool


# Language mappings for NLLB
NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
    "tr": "tur_Latn",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
}


SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
}

