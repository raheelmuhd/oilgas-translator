"""
Configuration management for the Oil & Gas Document Translator.
UPDATED with accuracy-optimized NLLB settings.
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

    # Debugging / instrumentation
    debug_translation: bool = False
    debug_output_dir: str = "./debug"
    debug_max_chars_per_page: int = 200000
    
    # Mode selection
    translation_mode: TranslationMode = TranslationMode.SELF_HOSTED
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = [
        "http://localhost:3000", 
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    
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
    
    # Ollama (Self-hosted - PRIMARY FREE OPTION)
    # Using qwen3:8b for high-quality translation with excellent structure preservation
    # Simple prompt works best: "Can you translate this to English:"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_max_concurrent: int = 4  # Max concurrent chunk translations
    
    # ==========================================================================
    # NLLB Settings - ACCURACY OPTIMIZED
    # ==========================================================================
    # 
    # Model options (set via NLLB_MODEL in .env):
    #   facebook/nllb-200-distilled-600M  - Fast, 60-70% accuracy, 2GB VRAM
    #   facebook/nllb-200-1.3B            - Medium, 80-85% accuracy, 4GB VRAM (RECOMMENDED)
    #   facebook/nllb-200-3.3B            - Slow, 90%+ accuracy, 10GB VRAM
    #
    nllb_model: str = "facebook/nllb-200-1.3B"  # CHANGED: Default to 1.3B for better quality
    nllb_device: str = "cuda"  # CHANGED: Default to GPU (auto-fallback to CPU)
    
    # Quality vs Speed settings
    nllb_batch_size: int = 2  # Lower = more accurate, higher = faster
    nllb_max_input_tokens: int = 512  # Max tokens per chunk (512 is safe)
    nllb_max_new_tokens: int = 400  # Max output tokens per chunk
    nllb_num_beams: int = 5  # CHANGED: Higher beams = better quality (was 4)
    nllb_no_repeat_ngram_size: int = 3  # Prevents repetition
    
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