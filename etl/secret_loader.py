"""
Secure secret loading utility.

Security goals:
- Never hardcode secrets.
- Prefer Docker secrets via *_FILE where available.
- Support local development via `.env` (git-ignored) without printing secret values.
- Treat secrets as local-only data: no logging of secret values, no persistence to git remotes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


_DOTENV_BOOTSTRAPPED = False


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a minimal .env file safely (KEY=VALUE or KEY:VALUE)."""
    mapping: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue

        delimiter = "=" if "=" in line else ":" if ":" in line else None
        if not delimiter:
            continue

        key, value = line.split(delimiter, 1)
        key = key.strip()
        if not key or key in mapping:
            continue

        cleaned = value.strip().strip('"').strip("'")
        if not cleaned:
            continue
        mapping[key] = cleaned

    return mapping


def bootstrap_dotenv() -> None:
    """
    Best-effort loading of `.env` for local development.

    This is intentionally quiet and never logs secret values.

    Disable with:
    - CI=true/1
    - PMX_DISABLE_DOTENV=true/1
    """
    global _DOTENV_BOOTSTRAPPED
    if _DOTENV_BOOTSTRAPPED:
        return
    _DOTENV_BOOTSTRAPPED = True

    if _truthy_env("CI") or _truthy_env("PMX_DISABLE_DOTENV"):
        return

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    try:
        from dotenv import dotenv_values  # type: ignore
    except Exception:
        dotenv_values = None

    if dotenv_values:
        try:
            parsed = dotenv_values(env_path)  # type: ignore[arg-type]
            for key, value in parsed.items():
                if not key or not value:
                    continue
                os.environ.setdefault(key, str(value))
            return
        except Exception:
            pass

    # Fallback: minimal parsing (supports KEY=VALUE and KEY:'value' styles).
    try:
        parsed = _parse_env_file(env_path)
        for key, value in parsed.items():
            os.environ.setdefault(key, value)
    except Exception:
        return


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
    bootstrap_dotenv()

    # Determine secret file path
    if secret_file_env is None:
        secret_file_env = f"{env_var_name}_FILE"

    secret_file_path = (os.getenv(secret_file_env) or "").strip() or None
    
    # Try Docker secret file first (production)
    if secret_file_path and Path(secret_file_path).exists():
        try:
            with open(secret_file_path, 'r', encoding='utf-8') as handle:
                for raw_line in handle:
                    stripped = raw_line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    logger.debug("Loaded %s from *_FILE path: %s", env_var_name, secret_file_path)
                    return stripped
            logger.warning("Secret file %s is empty or contains only comments", secret_file_path)
        except Exception as e:
            logger.warning("Failed to read secret file %s: %s", secret_file_path, e)
    
    # Fallback to environment variable (development/local)
    secret = (os.getenv(env_var_name) or "").strip()
    if secret:
        logger.debug("Loaded %s from environment variable", env_var_name)
        return secret
    
    logger.debug("Secret %s not found in *_FILE path or environment variables", env_var_name)
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


def load_polygon_key() -> Optional[str]:
    """Load Polygon.io API key (stored as MASSIVE_API_KEY in .env)."""
    return load_secret('MASSIVE_API_KEY', 'MASSIVE_API_KEY_FILE')


def load_luno_credentials() -> Dict[str, Optional[str]]:
    """Load LUNO API credentials (key + id).

    Returns:
        Dict with 'api_key' and 'api_id' values (or None if not found).
    """
    return {
        'api_key': load_secret('API_KEY_LUNO', 'API_KEY_LUNO_FILE'),
        'api_id': load_secret('API_ID_LUNO', 'API_ID_LUNO_FILE'),
    }


def load_xtb_username() -> Optional[str]:
    """Load XTB username."""
    return load_secret('XTB_USERNAME', 'XTB_USERNAME_FILE')


def load_xtb_password() -> Optional[str]:
    """Load XTB password."""
    return load_secret('XTB_PASSWORD', 'XTB_PASSWORD_FILE')

