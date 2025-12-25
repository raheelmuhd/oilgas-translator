"""
Configuration management for the Oil & Gas Document Translator.
Supports three modes: SELF_HOSTED (free), BUDGET, and PREMIUM.
"""

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class TranslationMode(str, Enum):
    SELF_HOSTED = "self_hosted"  # Free - NLLB + PaddleOCR
    BUDGET = "budget"            # ~$3/doc - DeepSeek + PaddleOCR
    PREMIUM = "premium"          # Best quality - Claude + Azure


class OCRProvider(str, Enum):
    PADDLEOCR = "paddleocr"
    AZURE = "azure"
    GPT4O = "gpt4o"


class TranslationProvider(str, Enum):
    NLLB = "nllb"           # Free, CPU-friendly
    OLLAMA = "ollama"       # Free, needs GPU for speed
    DEEPSEEK = "deepseek"   # Budget
    CLAUDE = "claude"       # Premium
    GPT4O = "gpt4o"         # Premium alternative


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App settings
    app_name: str = "Oil & Gas Document Translator"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Mode selection
    translation_mode: TranslationMode = TranslationMode.SELF_HOSTED
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # File handling
    max_file_size_mb: int = 600
    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    temp_dir: str = "./temp"
    
    # Database
    database_url: str = "sqlite+aiosqlite:///./translator.db"
    redis_url: str = "redis://localhost:6379/0"
    
    # Azure Document Intelligence (Premium OCR)
    azure_doc_endpoint: Optional[str] = None
    azure_doc_key: Optional[str] = None
    
    # Anthropic Claude (Premium Translation)
    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-3-5-sonnet-20241022"
    
    # OpenAI (Alternative)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    
    # DeepSeek (Budget Translation)
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    
    # Ollama (Self-hosted alternative)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    
    # NLLB Settings (Free translation model)
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    nllb_device: str = "cpu"  # Use CPU by default
    nllb_batch_size: int = 4
    
    # Processing settings
    chunk_size: int = 1000  # Characters per translation chunk
    max_concurrent_jobs: int = 3
    job_timeout_minutes: int = 60
    
    # Glossary
    glossary_path: str = "./glossary/oilgas_terminology.json"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_providers_for_mode(mode: TranslationMode) -> tuple[OCRProvider, TranslationProvider]:
    """Get the default OCR and translation providers for a given mode."""
    providers = {
        TranslationMode.SELF_HOSTED: (OCRProvider.PADDLEOCR, TranslationProvider.NLLB),
        TranslationMode.BUDGET: (OCRProvider.PADDLEOCR, TranslationProvider.DEEPSEEK),
        TranslationMode.PREMIUM: (OCRProvider.AZURE, TranslationProvider.CLAUDE),
    }
    return providers[mode]

