"""
Secure secret loading utility.

Supports loading secrets from:
1. Docker secrets (production) - mounted at /run/secrets/<secret_name>
2. Environment variables (development) - from .env file
3. Fallback to direct environment variable lookup

This provides secure secret management for both local development and Docker production.
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_secret(env_var_name: str, secret_file_env: Optional[str] = None) -> Optional[str]:
    """
    Load secret from Docker secret file or environment variable.
    
    Priority order:
    1. Docker secret file (if secret_file_env points to a file that exists)
    2. Environment variable (direct lookup)
    3. None (if not found)
    
    Args:
        env_var_name: Environment variable name (e.g., 'ALPHA_VANTAGE_API_KEY')
        secret_file_env: Environment variable containing path to secret file
                        (e.g., 'ALPHA_VANTAGE_API_KEY_FILE'). If None, uses
                        pattern: {env_var_name}_FILE
        
    Returns:
        Secret value or None if not found
        
    Example:
        >>> # In Docker with secrets
        >>> api_key = load_secret('ALPHA_VANTAGE_API_KEY', 'ALPHA_VANTAGE_API_KEY_FILE')
        >>> # Reads from /run/secrets/alpha_vantage_api_key if ALPHA_VANTAGE_API_KEY_FILE is set
        >>> # Falls back to ALPHA_VANTAGE_API_KEY env var if file doesn't exist
        
        >>> # Local development
        >>> api_key = load_secret('ALPHA_VANTAGE_API_KEY')
        >>> # Reads from ALPHA_VANTAGE_API_KEY environment variable
    """
    # Determine secret file path
    if secret_file_env is None:
        secret_file_env = f"{env_var_name}_FILE"
    
    secret_file_path = os.getenv(secret_file_env)
    
    # Try Docker secret file first (production)
    if secret_file_path and Path(secret_file_path).exists():
        try:
            with open(secret_file_path, 'r') as f:
                secret = f.read().strip()
                if secret and not secret.startswith('#'):
                    logger.debug(f"Loaded {env_var_name} from Docker secret: {secret_file_path}")
                    return secret
                else:
                    logger.warning(f"Docker secret file {secret_file_path} is empty or contains only comments")
        except Exception as e:
            logger.warning(f"Failed to read Docker secret file {secret_file_path}: {e}")
    
    # Fallback to environment variable (development/local)
    secret = os.getenv(env_var_name)
    if secret:
        logger.debug(f"Loaded {env_var_name} from environment variable")
        return secret
    
    logger.debug(f"Secret {env_var_name} not found in Docker secrets or environment variables")
    return None


def load_api_key(api_key_env_name: str) -> Optional[str]:
    """
    Convenience function to load API key with standard naming convention.
    
    Args:
        api_key_env_name: Environment variable name (e.g., 'ALPHA_VANTAGE_API_KEY')
        
    Returns:
        API key value or None if not found
    """
    return load_secret(api_key_env_name)


# Convenience functions for common API keys
def load_alpha_vantage_key() -> Optional[str]:
    """Load Alpha Vantage API key."""
    return load_secret('ALPHA_VANTAGE_API_KEY', 'ALPHA_VANTAGE_API_KEY_FILE')


def load_finnhub_key() -> Optional[str]:
    """Load Finnhub API key."""
    return load_secret('FINNHUB_API_KEY', 'FINNHUB_API_KEY_FILE')


def load_xtb_username() -> Optional[str]:
    """Load XTB username."""
    return load_secret('XTB_USERNAME', 'XTB_USERNAME_FILE')


def load_xtb_password() -> Optional[str]:
    """Load XTB password."""
    return load_secret('XTB_PASSWORD', 'XTB_PASSWORD_FILE')

