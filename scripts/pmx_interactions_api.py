#!/usr/bin/env python3
"""
Portfolio Maximizer - Local Interactions API (FastAPI)

This provides a small, security-first HTTP surface for local testing and
external integrations (e.g., via ngrok).

Endpoints:
- POST /interactions
  Secure "interaction" webhook endpoint. Requires auth (API key or JWT).

- GET /verify-roles
  Verifies an Auth0 JWT and returns a minimal RBAC summary.

- GET /terms-of-service
- GET /privacy-policy
  Redirects to URLs configured via .env.

Security defaults:
- Requires `INTERACTIONS_API_KEY` for /interactions unless JWT auth is configured.
- Rate limits all requests (in-memory token bucket).
- Does not log request bodies.
- Adds security headers.

Environment variables (.env, never commit):
- PMX_ENV=local|production (default: local)
- INTERACTIONS_BIND_HOST=127.0.0.1
- INTERACTIONS_PORT=8000
- INTERACTIONS_API_KEY=...  (or INTERACTIONS_API_KEY_FILE=/run/secrets/...)
- INTERACTIONS_RATE_LIMIT_PER_MINUTE=60
- INTERACTIONS_MAX_BODY_BYTES=65536
- INTERACTIONS_REQUIRED_SCOPES=pmx:interactions  (optional; JWT-only)

Auth0 (optional; for JWT validation):
- AUTH0_DOMAIN=your-tenant.us.auth0.com
- AUTH0_AUDIENCE=your-api-identifier
- AUTH0_ISSUER=https://your-tenant.us.auth0.com/
- AUTH0_ALGORITHMS=RS256
- LINKED_ROLES_REQUIRED_SCOPES=pmx:verify_roles (optional)

Public URLs (optional; informational / redirects):
- INTERACTIONS_ENDPOINT_URL=https://<ngrok>.ngrok.io/interactions
- LINKED_ROLES_VERIFICATION_URL=https://<ngrok>.ngrok.io/verify-roles
- TERMS_OF_SERVICE_URL=...
- PRIVACY_POLICY_URL=...
"""

from __future__ import annotations

import argparse
import hmac
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

_PLACEHOLDER_SECRETS = {
    "your_interactions_api_key_here",
    "your_api_key_here",
    "changeme",
    "change_me",
    "replace_me",
    "replace-me",
    "todo",
}


def _looks_like_placeholder_secret(value: str) -> bool:
    v = (value or "").strip().lower()
    if not v:
        return True
    if v in _PLACEHOLDER_SECRETS:
        return True
    if v.startswith("your_") and v.endswith("_here"):
        return True
    return False


def _bootstrap_dotenv() -> None:
    # Best-effort; never logs values.
    try:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except (ImportError, OSError, ValueError):
        return


def _load_secret(name: str) -> Optional[str]:
    try:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))
        from etl.secret_loader import load_secret

        return load_secret(name)
    except (ImportError, OSError, ValueError):
        return (os.getenv(name) or "").strip() or None


def _truthy(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _pmx_env() -> str:
    return (os.getenv("PMX_ENV") or os.getenv("ENV") or "local").strip().lower() or "local"


def _auth_mode() -> str:
    """Return configured auth mode: 'any', 'jwt-only', or 'api-key-only'."""
    mode = (os.getenv("INTERACTIONS_AUTH_MODE") or "any").strip().lower()
    if mode not in {"any", "jwt-only", "api-key-only"}:
        return "any"
    return mode


def _is_production_env() -> bool:
    # Conservative: treat CI as production-like for safety.
    if _truthy("CI"):
        return True
    return _pmx_env() in {"prod", "production", "live"}


def _csv_env(name: str) -> list[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    normalized = raw.replace("\r\n", "\n").replace("\n", ",").replace(";", ",")
    return [p.strip() for p in normalized.split(",") if p and p.strip()]


def _hash_id(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8", errors="ignore")).hexdigest()[:16]


@dataclass
class _TokenBucket:
    capacity: float
    refill_per_sec: float
    tokens: float
    last_ts: float


class RateLimiter:
    """Simple in-memory token bucket rate limiter."""

    def __init__(self, *, per_minute: int) -> None:
        cap = max(1, int(per_minute))
        self._capacity = float(cap)
        self._refill_per_sec = float(cap) / 60.0
        self._buckets: dict[str, _TokenBucket] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        k = (key or "").strip() or "anon"
        b = self._buckets.get(k)
        if b is None:
            self._buckets[k] = _TokenBucket(
                capacity=self._capacity,
                refill_per_sec=self._refill_per_sec,
                tokens=self._capacity - 1.0,
                last_ts=now,
            )
            return True

        # Refill
        elapsed = max(0.0, now - float(b.last_ts))
        b.tokens = min(b.capacity, float(b.tokens) + elapsed * b.refill_per_sec)
        b.last_ts = now

        if b.tokens < 1.0:
            return False
        b.tokens -= 1.0
        return True


def _get_client_key(request: Request, *, api_key: Optional[str], jwt_sub: Optional[str]) -> str:
    if jwt_sub:
        return f"jwt:{_hash_id(jwt_sub)}"
    if api_key:
        return f"key:{_hash_id(api_key)}"
    host = getattr(getattr(request, "client", None), "host", "") or "unknown"
    return f"ip:{host}"


def _audit_log_path() -> Path:
    path = (os.getenv("PMX_AUDIT_LOG_PATH") or "").strip()
    if path:
        return Path(path).expanduser().resolve()
    return (PROJECT_ROOT / "logs" / "security_audit.jsonl").resolve()


def _write_audit_event(event: dict[str, Any]) -> None:
    try:
        path = _audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=True, sort_keys=True)
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
    except (OSError, TypeError, ValueError):
        return


def _extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    text = (auth_header or "").strip()
    if not text:
        return None
    if not text.lower().startswith("bearer "):
        return None
    return text.split(" ", 1)[1].strip() or None


class Auth0JwtVerifier:
    def __init__(self) -> None:
        self.domain = (os.getenv("AUTH0_DOMAIN") or "").strip()
        self.audience = (os.getenv("AUTH0_AUDIENCE") or "").strip()
        issuer = (os.getenv("AUTH0_ISSUER") or "").strip()
        self.issuer = issuer or (f"https://{self.domain}/" if self.domain else "")
        algs = (os.getenv("AUTH0_ALGORITHMS") or "RS256").strip()
        self.algorithms = [a.strip() for a in algs.split(",") if a.strip()]

        self.enabled = bool(self.domain and self.audience and self.issuer)

        self._jwks_client = None

    def verify(self, token: str) -> dict[str, Any]:
        if not self.enabled:
            raise HTTPException(status_code=503, detail="Auth0 JWT validation not configured (set AUTH0_DOMAIN/AUTH0_AUDIENCE).")

        try:
            import jwt  # PyJWT
            from jwt import PyJWKClient
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=503, detail=f"JWT library unavailable: {type(exc).__name__}")

        # RS256 verification requires cryptography (PyJWT[crypto]).
        try:
            import cryptography  # noqa: F401
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="JWT RS256 verification requires cryptography. Install PyJWT[crypto]/cryptography, then retry.",
            )

        if self._jwks_client is None:
            jwks_url = f"https://{self.domain}/.well-known/jwks.json"
            self._jwks_client = PyJWKClient(jwks_url)

        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat"]},
            )
            if not isinstance(claims, dict):
                raise HTTPException(status_code=401, detail="Invalid JWT claims.")
            return claims
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="JWT expired.")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid JWT.")


def _extract_scopes(claims: dict[str, Any]) -> set[str]:
    scopes: set[str] = set()
    perm = claims.get("permissions")
    if isinstance(perm, list):
        for p in perm:
            if isinstance(p, str) and p.strip():
                scopes.add(p.strip())
    scope = claims.get("scope")
    if isinstance(scope, str):
        for p in scope.split():
            if p.strip():
                scopes.add(p.strip())
    return scopes


def _require_scopes(claims: dict[str, Any], required: list[str]) -> None:
    if not required:
        return
    have = _extract_scopes(claims)
    missing = [r for r in required if r not in have]
    if missing:
        raise HTTPException(status_code=403, detail=f"Missing required scope(s): {', '.join(missing)}")


def _get_expected_api_key() -> Optional[str]:
    expected = _load_secret("INTERACTIONS_API_KEY")
    if not expected:
        return None
    expected = expected.strip()
    min_len = int(os.getenv("INTERACTIONS_MIN_KEY_LENGTH") or "16")
    if len(expected) < max(16, min_len):  # floor at 16, never allow weaker
        return None
    if _looks_like_placeholder_secret(expected):
        return None
    return expected


def _api_key_ok(provided: Optional[str]) -> bool:
    expected = _get_expected_api_key()
    if not expected:
        return False
    return hmac.compare_digest((provided or "").strip(), expected)


def _security_headers(response: Response) -> Response:
    # Reuse repo helper if available; otherwise set a minimal set.
    try:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.security_middleware import add_security_headers  # type: ignore

        return add_security_headers(response)
    except (ImportError, AttributeError):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


_bootstrap_dotenv()

_RATE_LIMIT_PER_MIN = int(os.getenv("INTERACTIONS_RATE_LIMIT_PER_MINUTE") or "60")
_MAX_BODY_BYTES = int(os.getenv("INTERACTIONS_MAX_BODY_BYTES") or "65536")
_limiter = RateLimiter(per_minute=_RATE_LIMIT_PER_MIN)
_jwt_verifier = Auth0JwtVerifier()

app = FastAPI(title="PMX Interactions API", version="1.0.0")

_cors_origins = _csv_env("INTERACTIONS_CORS_ORIGINS")
if _cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
        max_age=3600,
    )


@app.middleware("http")
async def _security_middleware(request: Request, call_next):
    # Rate limit first (key selection happens after auth; use IP placeholder here).
    # This catches obvious floods, but the main limiter is keyed after auth in endpoints.
    if not _limiter.allow(_get_client_key(request, api_key=None, jwt_sub=None) + ":pre"):
        return JSONResponse(status_code=429, content={"error": "rate_limited"})

    try:
        response = await call_next(request)
    except HTTPException:
        raise
    except Exception:
        # Avoid leaking stack traces by default.
        if _truthy("PMX_DEBUG") and not _is_production_env():
            raise
        return JSONResponse(status_code=500, content={"error": "internal_error"})

    return _security_headers(response)


@app.get("/healthz")
async def healthz():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat(), "env": _pmx_env()}


async def _read_limited_body(request: Request, *, max_bytes: int) -> bytes:
    if max_bytes <= 0:
        return await request.body()
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="payload_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/interactions")
async def interactions(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    """
    Secure interaction endpoint. For public exposure (ngrok), require:
    - X-API-Key == INTERACTIONS_API_KEY
      OR
    - A valid Auth0 JWT (if configured)
    """

    # Auth
    api_key = (x_api_key or "").strip() or None
    token = _extract_bearer_token(authorization)
    claims: Optional[dict[str, Any]] = None
    jwt_sub: Optional[str] = None
    configured_mode = _auth_mode()

    # Enforce auth mode restrictions before attempting validation.
    if configured_mode == "jwt-only" and api_key and not token:
        raise HTTPException(
            status_code=401,
            detail="JWT-only mode active. API key auth is disabled (set INTERACTIONS_AUTH_MODE=any to allow).",
        )
    if configured_mode == "api-key-only" and token and not api_key:
        raise HTTPException(
            status_code=401,
            detail="API-key-only mode active. JWT auth is disabled (set INTERACTIONS_AUTH_MODE=any to allow).",
        )

    if api_key and _api_key_ok(api_key) and configured_mode != "jwt-only":
        auth_method = "api_key"
    elif token and configured_mode != "api-key-only":
        if not _jwt_verifier.enabled:
            raise HTTPException(status_code=401, detail="JWT auth not configured.")
        claims = _jwt_verifier.verify(token)
        jwt_sub = str(claims.get("sub") or "").strip() or None
        required = _csv_env("INTERACTIONS_REQUIRED_SCOPES")
        _require_scopes(claims, required)
        auth_method = "jwt"
    else:
        # Refuse to serve publicly if no auth mechanism is configured.
        if not _get_expected_api_key() and not _jwt_verifier.enabled:
            raise HTTPException(
                status_code=503,
                detail="No auth configured (set INTERACTIONS_API_KEY>=16 chars, or AUTH0_DOMAIN+AUTH0_AUDIENCE).",
            )
        raise HTTPException(status_code=401, detail="Missing/invalid credentials.")

    client_key = _get_client_key(request, api_key=api_key, jwt_sub=jwt_sub) + ":interactions"
    if not _limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="rate_limited")

    # Body is intentionally NOT logged; read only to validate JSON (best-effort).
    payload: Any = None
    try:
        body = await _read_limited_body(request, max_bytes=_MAX_BODY_BYTES)
        if body:
            payload = json.loads(body)
    except HTTPException:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        payload = None

    # Audit trail (minimal)
    _write_audit_event(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "path": "/interactions",
            "method": "POST",
            "auth": auth_method,
            "principal": _hash_id(jwt_sub or api_key or ""),
            "ok": True,
        }
    )

    # Never return sensitive data by default.
    return {"ok": True, "received": isinstance(payload, (dict, list))}


@app.get("/verify-roles")
async def verify_roles(
    request: Request,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    token = _extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token.")

    claims = _jwt_verifier.verify(token)
    required = _csv_env("LINKED_ROLES_REQUIRED_SCOPES")
    _require_scopes(claims, required)

    jwt_sub = str(claims.get("sub") or "").strip() or None
    client_key = _get_client_key(request, api_key=None, jwt_sub=jwt_sub) + ":verify_roles"
    if not _limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="rate_limited")

    scopes = sorted(_extract_scopes(claims))
    roles = claims.get("roles") if isinstance(claims.get("roles"), list) else None
    namespace_roles = []
    # Support common Auth0 namespaced roles claim (user-defined).
    for k, v in claims.items():
        if isinstance(k, str) and k.endswith("/roles") and isinstance(v, list):
            namespace_roles = [str(x) for x in v if str(x).strip()]
            break

    _write_audit_event(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "path": "/verify-roles",
            "method": "GET",
            "auth": "jwt",
            "principal": _hash_id(jwt_sub or ""),
            "ok": True,
        }
    )

    return {
        "ok": True,
        "sub": jwt_sub,
        "scopes": scopes,
        "roles": roles or namespace_roles,
        "issuer": str(claims.get("iss") or ""),
        "aud": claims.get("aud"),
        "exp": claims.get("exp"),
    }


@app.get("/terms-of-service")
async def terms_of_service():
    url = (os.getenv("TERMS_OF_SERVICE_URL") or "").strip()
    if not url:
        raise HTTPException(status_code=404, detail="TERMS_OF_SERVICE_URL not configured.")
    return RedirectResponse(url=url, status_code=302)


@app.get("/privacy-policy")
async def privacy_policy():
    url = (os.getenv("PRIVACY_POLICY_URL") or "").strip()
    if not url:
        raise HTTPException(status_code=404, detail="PRIVACY_POLICY_URL not configured.")
    return RedirectResponse(url=url, status_code=302)


@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException):
    # Avoid reflecting sensitive details.
    _write_audit_event(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
            "method": str(request.method),
            "auth": "unknown",
            "principal": "",
            "ok": False,
            "status": int(exc.status_code),
        }
    )
    return JSONResponse(status_code=int(exc.status_code), content={"error": "request_failed", "detail": str(exc.detail)})


@app.exception_handler(Exception)
async def _unhandled_exc_handler(request: Request, exc: Exception):
    # Never include stack traces unless explicitly enabled.
    if _truthy("PMX_DEBUG") and not _is_production_env():
        raise exc
    _write_audit_event(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "path": str(request.url.path),
            "method": str(request.method),
            "auth": "unknown",
            "principal": "",
            "ok": False,
            "status": 500,
        }
    )
    return JSONResponse(status_code=500, content={"error": "internal_error"})


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PMX Interactions API service")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate runtime configuration/dependencies and exit.",
    )
    parser.add_argument(
        "--print-openapi",
        action="store_true",
        help="Print OpenAPI schema as JSON and exit.",
    )
    args = parser.parse_args(argv)

    if args.print_openapi:
        print(json.dumps(app.openapi(), ensure_ascii=True))
        return 0

    host = (os.getenv("INTERACTIONS_BIND_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("INTERACTIONS_PORT") or "8000")

    if _is_production_env() and host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing to bind non-loopback host in production (set INTERACTIONS_BIND_HOST=127.0.0.1).")

    if args.check:
        payload = {
            "ok": True,
            "env": _pmx_env(),
            "auth_mode": _auth_mode(),
            "host": host,
            "port": port,
            "jwt_enabled": bool(_jwt_verifier.enabled),
            "api_key_configured": bool(_get_expected_api_key()),
            "rate_limit_per_min": _RATE_LIMIT_PER_MIN,
            "max_body_bytes": _MAX_BODY_BYTES,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # Prefer uvicorn already installed in simpleTrader_env.
    import uvicorn

    uvicorn.run(app, host=host, port=port, reload=not _is_production_env(), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
