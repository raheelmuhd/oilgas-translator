"""
Custom exception classes for the Document Translator application.
"""

from typing import Any, Optional


class DocumentTranslatorError(Exception):
    """Base exception for Document Translator application."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class FileValidationError(DocumentTranslatorError):
    """Raised when file validation fails."""
    pass


class TranslationError(DocumentTranslatorError):
    """Raised when translation fails."""
    pass


class ConfigurationError(DocumentTranslatorError):
    """Raised when configuration is invalid."""
    pass


class ProviderError(DocumentTranslatorError):
    """Raised when a translation/OCR provider fails."""
    pass

