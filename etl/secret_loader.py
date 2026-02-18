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

# Canonical secret names -> accepted legacy aliases.
# Keep this list narrow and explicit to avoid accidental credential mixups.
_SECRET_ALIASES: Dict[str, tuple[str, ...]] = {
    "OPENAI_API_KEY": ("OPENAI_SECRET_KEY",),
    "INTERACTIONS_API_KEY": ("INTERACTIONS_KEY", "INTERACTIONS_TOKEN"),
    "QWEN_API_KEY": ("DASHSCOPE_API_KEY", "QWEN_PASSWORD"),
    "DASHSCOPE_API_KEY": ("QWEN_API_KEY", "QWEN_PASSWORD"),
    "PROJECTS_TOKEN": ("PROJECTS_SECRET",),
    "PMX_EMAIL_USERNAME": ("MAIN_EMAIL_GMAIL", "OPENAI_EMAIL"),
    "PMX_EMAIL_PASSWORD": ("OPENAI_EMAIL_PASSWORD",),
    "PMX_EMAIL_TO": ("MAIN_EMAIL_GMAIL", "ALTERNATIVE_EMAIL_PROTONMAIL"),
    "PMX_EMAIL_FROM": ("MAIN_EMAIL_GMAIL", "OPENAI_EMAIL"),
    "PMX_PROTON_BRIDGE_USERNAME": ("ALTERNATIVE_EMAIL_PROTONMAIL",),
    "DISCORD_BOT_TOKEN": ("DISCORD_TOKEN",),
    "DISCORD_APP_NAME": ("DISCORD_APPLICATION_NAME",),
    "DISCORD_APPLICATION_ID": ("DISCORD_APP_ID", "DISCORD_CLIENT_ID"),
    "DISCORD_PUBLIC_KEY": ("DISCORD_INTERACTIONS_PUBLIC_KEY",),
    "DISCORD_APP_INSTALL_LINK": ("DISCORD_INSTALL_LINK", "DISCORD_APP_INSTALL_URL"),
    "SLACK_BOT_TOKEN": ("SLACK_TOKEN",),
    "SLACK_APP_TOKEN": ("SLACK_SOCKET_MODE_TOKEN",),
    "TELEGRAM_BOT_TOKEN": ("TELEGRAM_HTTP_API_TOKEN",),
}


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


def _csv_env_aliases(name: str) -> list[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    normalized = raw.replace("\r\n", "\n").replace("\n", ",").replace(";", ",")
    out: list[str] = []
    for piece in normalized.split(","):
        key = piece.strip()
        if key:
            out.append(key)
    return out


def _resolve_aliases(env_var_name: str) -> list[str]:
    base = list(_SECRET_ALIASES.get(env_var_name, ()))
    # Optional per-secret extension without code changes:
    # Example: PMX_EMAIL_PASSWORD_ALIASES=GMAIL_APP_PASSWORD
    base.extend(_csv_env_aliases(f"{env_var_name}_ALIASES"))
    seen: set[str] = set()
    out: list[str] = []
    for key in base:
        k = (key or "").strip()
        if not k or k == env_var_name or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _read_secret_from_file_path(secret_file_path: str, *, env_name_for_log: str) -> Optional[str]:
    path = (secret_file_path or "").strip()
    if not path or not Path(path).exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                logger.debug("Loaded %s from *_FILE path: %s", env_name_for_log, path)
                return stripped
        logger.warning("Secret file %s is empty or contains only comments", path)
    except Exception as e:
        logger.warning("Failed to read secret file %s: %s", path, e)
    return None


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

    aliases = _resolve_aliases(env_var_name)

    # Determine secret file env names to probe.
    if secret_file_env is None:
        secret_file_env = f"{env_var_name}_FILE"

    file_env_names = [secret_file_env]
    if secret_file_env == f"{env_var_name}_FILE":
        file_env_names.extend(f"{alias}_FILE" for alias in aliases)

    seen_file_envs: set[str] = set()
    for file_env_name in file_env_names:
        if not file_env_name or file_env_name in seen_file_envs:
            continue
        seen_file_envs.add(file_env_name)
        secret_file_path = (os.getenv(file_env_name) or "").strip() or None
        secret = _read_secret_from_file_path(secret_file_path or "", env_name_for_log=env_var_name)
        if secret:
            return secret

    # Fallback to environment variables (canonical first, then aliases).
    env_names = [env_var_name, *aliases]
    for name in env_names:
        secret = (os.getenv(name) or "").strip()
        if secret:
            source = "environment variable"
            if name != env_var_name:
                source += f" alias ({name})"
            logger.debug("Loaded %s from %s", env_var_name, source)
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
