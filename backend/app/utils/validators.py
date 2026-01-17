"""
Input validation utilities for file uploads and API requests.
"""

from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException, UploadFile


# Allowed file extensions
ALLOWED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
}

# Allowed MIME types
ALLOWED_MIME_TYPES = set(ALLOWED_EXTENSIONS.values())

# Device options
VALID_DEVICES = {"cpu", "gpu", "auto"}


def validate_file_extension(filename: str) -> str:
    """
    Validate file extension.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        File extension with leading dot
        
    Raises:
        HTTPException: If file extension is not allowed
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    file_ext = Path(filename).suffix.lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS.keys()))
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_ext}' not allowed. Allowed types: {allowed}"
        )
    
    return file_ext


def validate_file_size(file_size: int, max_size_mb: int) -> None:
    """
    Validate file size.
    
    Args:
        file_size: Size of the file in bytes
        max_size_mb: Maximum allowed size in megabytes
        
    Raises:
        HTTPException: If file size exceeds maximum
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {max_size_mb}MB. "
                   f"File size: {file_size / (1024 * 1024):.2f}MB"
        )


def validate_file_content(file_path: Path, expected_ext: str) -> bool:
    """
    Validate file content matches extension (basic check).
    
    Note: Full MIME type checking requires python-magic library.
    For production, consider adding deeper validation.
    
    Args:
        file_path: Path to the uploaded file
        expected_ext: Expected file extension
        
    Returns:
        True if file appears valid
        
    Raises:
        HTTPException: If file content doesn't match extension
    """
    try:
        # Basic validation - check file signature for common formats
        with open(file_path, "rb") as f:
            header = f.read(4)
        
        # PDF signature: %PDF
        if expected_ext == ".pdf" and not header.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="File does not appear to be a valid PDF"
            )
        
        # DOCX/ZIP signature: PK (ZIP format)
        if expected_ext in (".docx", ".xlsx", ".pptx") and not header.startswith(b"PK\x03\x04"):
            raise HTTPException(
                status_code=400,
                detail=f"File does not appear to be a valid {expected_ext[1:].upper()} file"
            )
        
        return True
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        # If validation fails, log but don't block (fail open for robustness)
        return True


def validate_device(device: str) -> str:
    """
    Validate device selection.
    
    Args:
        device: Device selection string
        
    Returns:
        Validated device string
        
    Raises:
        HTTPException: If device is not valid
    """
    if device not in VALID_DEVICES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device selection. Must be one of: {', '.join(sorted(VALID_DEVICES))}"
        )
    return device


def validate_language_code(lang_code: Optional[str], allowed_languages: set) -> Optional[str]:
    """
    Validate language code.
    
    Args:
        lang_code: Language code to validate
        allowed_languages: Set of allowed language codes
        
    Returns:
        Validated language code or None
        
    Raises:
        HTTPException: If language code is not allowed
    """
    if lang_code is None:
        return None
    
    if lang_code not in allowed_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language code: {lang_code}. "
                   f"Allowed languages: {', '.join(sorted(allowed_languages))}"
        )
    
    return lang_code


def validate_file_upload(
    file: UploadFile,
    max_size_mb: int,
    validate_content: bool = True
) -> Tuple[str, Path]:
    """
    Comprehensive file upload validation.
    
    Args:
        file: Uploaded file object
        max_size_mb: Maximum file size in megabytes
        validate_content: Whether to validate file content
        
    Returns:
        Tuple of (file_extension, validated_file_path)
        
    Raises:
        HTTPException: If validation fails
    """
    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    # Validate extension
    file_ext = validate_file_extension(file.filename)
    
    # Note: File size validation should be done after saving the file
    # as we need to read the actual file size
    
    return file_ext, Path(file.filename)

