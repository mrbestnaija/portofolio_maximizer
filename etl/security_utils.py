"""
Security utilities for error handling and sanitization.

This module provides security-focused utilities to prevent information leakage
and improve error handling in production environments.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def sanitize_error(exc: Exception, is_production: Optional[bool] = None) -> str:
    """
    Sanitize error messages for production.
    
    Prevents information leakage by returning generic error messages in production
    while preserving detailed errors for development/debugging.
    
    Args:
        exc: Exception object to sanitize
        is_production: Whether running in production mode.
                      If None, auto-detects from PORTFOLIO_ENV environment variable
        
    Returns:
        Safe error message (generic in production, detailed in development)
        
    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     safe_msg = sanitize_error(e, is_production=True)
        ...     logger.error(f"Operation failed: {safe_msg}")
        ...     # In production: "Operation failed: An error occurred. Please contact support."
        ...     # In development: "Operation failed: Connection timeout after 30s"
    """
    if is_production is None:
        is_production = os.getenv('PORTFOLIO_ENV', 'development') == 'production'
    
    if is_production:
        # Generic error message for users - prevents information leakage
        # Detailed error is still logged server-side with full traceback
        logger.error(f"Internal error: {exc}", exc_info=True)
        return "An error occurred. Please contact support."
    else:
        # Detailed error for development/debugging
        return str(exc)


def sanitize_log_message(message: str, sensitive_patterns: Optional[list] = None) -> str:
    """
    Remove sensitive information from log messages.
    
    Args:
        message: Log message that may contain sensitive data
        sensitive_patterns: List of regex patterns to match and redact.
                          Defaults to common patterns (API keys, passwords, etc.)
        
    Returns:
        Sanitized log message with sensitive data redacted
    """
    import re
    
    if sensitive_patterns is None:
        # Default patterns for common sensitive data (case-insensitive)
        sensitive_patterns = [
            (r'(api[_-]?key["\s:=]+)([^\s,"\']+)', r'\1***REDACTED***'),
            (r'(password["\s:=]+)([^\s,"\']+)', r'\1***REDACTED***'),
            (r'(token["\s:=]+)([^\s,"\']+)', r'\1***REDACTED***'),
            (r'(secret["\s:=]+)([^\s,"\']+)', r'\1***REDACTED***'),
        ]
    
    sanitized = message
    for pattern, replacement in sensitive_patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    
    return sanitized

