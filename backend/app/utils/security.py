"""
Security utilities and middleware for the Document Translator application.
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

settings = get_settings()


# Rate limiting storage (in-memory, for production use Redis)
_rate_limit_store: defaultdict = defaultdict(list)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Remove server header (security through obscurity)
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware (in-memory).
    For production, use Redis-based rate limiting.
    
    Rate limits:
    - Upload endpoint: 10 requests per minute
    - Other endpoints: 100 requests per minute
    """
    
    # Rate limits: (requests, window_seconds)
    RATE_LIMITS = {
        "/api/v1/translate": (10, 60),  # 10 requests per minute
        "/api/v1/translate/quick": (30, 60),  # 30 requests per minute
        "default": (100, 60),  # 100 requests per minute
    }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/api/v1/health"]:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"
        
        # Determine rate limit for this endpoint
        path = request.url.path
        rate_limit = self.RATE_LIMITS.get(path) or self.RATE_LIMITS["default"]
        max_requests, window_seconds = rate_limit
        
        # Check rate limit
        key = f"{client_ip}:{path}"
        now = time.time()
        
        # Clean old entries
        _rate_limit_store[key] = [
            timestamp for timestamp in _rate_limit_store[key]
            if now - timestamp < window_seconds
        ]
        
        # Check if limit exceeded
        if len(_rate_limit_store[key]) >= max_requests:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {max_requests} requests per {window_seconds} seconds."
            )
        
        # Add current request
        _rate_limit_store[key].append(now)
        
        # Add rate limit headers
        response = await call_next(request)
        remaining = max(0, max_requests - len(_rate_limit_store[key]))
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + window_seconds))
        
        return response


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = Path(filename).name
    
    # Remove any remaining dangerous characters
    dangerous_chars = ['..', '/', '\\', '\x00']
    for char in dangerous_chars:
        filename = filename.replace(char, '')
    
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:max_length - len(ext) - 1] + '.' + ext if ext else name[:max_length]
    
    return filename

