"""
Security middleware for web applications.

Provides security headers and middleware functions to protect against
common web vulnerabilities (OWASP Top 10).
"""

from functools import wraps
from typing import Callable, Any, Dict


def add_security_headers(response: Any) -> Any:
    """
    Add security headers to HTTP response.
    
    Implements security best practices from OWASP:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking
    - X-XSS-Protection: Enables XSS filtering
    - Strict-Transport-Security: Forces HTTPS
    - Content-Security-Policy: Prevents XSS and injection attacks
    
    Args:
        response: HTTP response object (Flask, FastAPI, Django, etc.)
        
    Returns:
        Response object with security headers added
        
    Example:
        # Flask
        @app.after_request
        def after_request(response):
            return add_security_headers(response)
            
        # FastAPI
        @app.middleware("http")
        async def add_security_headers_middleware(request, call_next):
            response = await call_next(request)
            return add_security_headers(response)
    """
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    
    return response


def security_headers_middleware(func: Callable) -> Callable:
    """
    Decorator to add security headers to response.
    
    Usage:
        @app.route('/api/endpoint')
        @security_headers_middleware
        def endpoint():
            return {'status': 'ok'}
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        response = func(*args, **kwargs)
        return add_security_headers(response)
    return wrapper


def get_security_headers_dict() -> Dict[str, str]:
    """
    Get security headers as dictionary (for use with custom frameworks).
    
    Returns:
        Dictionary of security headers
    """
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }

