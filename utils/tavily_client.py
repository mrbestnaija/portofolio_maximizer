from __future__ import annotations

"""
Tavily API client helpers.

This module provides a small, dependency-light wrapper around Tavily Search for
PMX scripts and automations.
"""

import time
from dataclasses import dataclass
from typing import Any, Optional

import requests


DEFAULT_TAVILY_BASE_URL = "https://api.tavily.com"
ALLOWED_SEARCH_DEPTHS = {"basic", "advanced", "fast", "ultra-fast"}
ALLOWED_TOPICS = {"general", "news", "finance"}


@dataclass(frozen=True)
class TavilySearchResult:
    ok: bool
    query: str
    answer: str
    results: list[dict[str, Any]]
    error: Optional[str] = None
    status_code: Optional[int] = None
    latency_seconds: Optional[float] = None
    raw: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class TavilyUsageResult:
    ok: bool
    error: Optional[str] = None
    status_code: Optional[int] = None
    latency_seconds: Optional[float] = None
    key: Optional[dict[str, Any]] = None
    account: Optional[dict[str, Any]] = None
    raw: Optional[dict[str, Any]] = None


def _normalize_csv(values: Optional[str]) -> list[str]:
    raw = (values or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for piece in raw.replace(";", ",").split(","):
        token = piece.strip()
        if token:
            out.append(token)
    return out


def _resolve_tavily_api_key(explicit_api_key: Optional[str]) -> Optional[str]:
    provided = (explicit_api_key or "").strip()
    if provided:
        return provided
    try:
        from etl.secret_loader import load_secret

        return (load_secret("TAVILY_API_KEY") or "").strip() or None
    except Exception:
        return None


def _extract_error_message(payload: Any, status_code: int) -> str:
    if isinstance(payload, dict):
        for key in ("error", "detail", "message"):
            raw = payload.get(key)
            if isinstance(raw, str) and raw.strip():
                return f"http_{status_code}: {raw.strip()}"
    return f"http_{status_code}"


def _normalize_api_root(base_url: Optional[str]) -> str:
    root = (base_url or DEFAULT_TAVILY_BASE_URL).strip().rstrip("/")
    lowered = root.lower()
    if lowered.endswith("/search") or lowered.endswith("/usage"):
        root = root.rsplit("/", 1)[0]
    return root


def _is_retryable_status(status_code: int) -> bool:
    return int(status_code) in {429, 500, 502, 503, 504}


def _retry_delay_seconds(attempt_number: int) -> float:
    return min(2.0, 0.35 * (2 ** max(0, int(attempt_number))))


def _normalize_search_depth(value: str) -> str:
    depth = (value or "basic").strip().lower() or "basic"
    if depth not in ALLOWED_SEARCH_DEPTHS:
        return "basic"
    return depth


def _normalize_topic(value: str) -> str:
    topic = (value or "general").strip().lower() or "general"
    if topic not in ALLOWED_TOPICS:
        return "general"
    return topic


def search_tavily(
    *,
    query: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
    include_answer: bool = True,
    include_raw_content: bool = False,
    include_images: bool = False,
    include_domains_csv: Optional[str] = None,
    exclude_domains_csv: Optional[str] = None,
    timeout_seconds: float = 20.0,
    request_retries: int = 2,
) -> TavilySearchResult:
    """
    Run a Tavily search request.

    Notes:
    - API key is loaded from `TAVILY_API_KEY` when not supplied directly.
    - Authentication uses `Authorization: Bearer <TAVILY_API_KEY>`.
    """

    q = (query or "").strip()
    if not q:
        return TavilySearchResult(
            ok=False,
            query=q,
            answer="",
            results=[],
            error="missing_query",
        )

    resolved_key = _resolve_tavily_api_key(api_key)
    if not resolved_key:
        return TavilySearchResult(
            ok=False,
            query=q,
            answer="",
            results=[],
            error="missing_tavily_api_key",
        )

    url = _normalize_api_root(base_url) + "/search"
    include_domains = _normalize_csv(include_domains_csv)
    exclude_domains = _normalize_csv(exclude_domains_csv)

    payload: dict[str, Any] = {
        "query": q,
        "max_results": max(1, min(20, int(max_results))),
        "search_depth": _normalize_search_depth(search_depth),
        "topic": _normalize_topic(topic),
        "include_answer": bool(include_answer),
        "include_raw_content": bool(include_raw_content),
        "include_images": bool(include_images),
    }
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "pmx-tavily-client/1.0",
        "Authorization": f"Bearer {resolved_key}",
    }
    started = time.monotonic()
    retries = max(0, min(4, int(request_retries)))
    attempt = 0
    while True:
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=max(1.0, float(timeout_seconds)),
            )
        except Exception as exc:
            if attempt < retries:
                time.sleep(_retry_delay_seconds(attempt))
                attempt += 1
                continue
            return TavilySearchResult(
                ok=False,
                query=q,
                answer="",
                results=[],
                error=f"request_failed: {exc}",
                latency_seconds=round(time.monotonic() - started, 3),
            )

        latency = round(time.monotonic() - started, 3)
        try:
            body = response.json()
        except Exception:
            body = {}

        if int(response.status_code) >= 400:
            if _is_retryable_status(int(response.status_code)) and attempt < retries:
                time.sleep(_retry_delay_seconds(attempt))
                attempt += 1
                continue
            return TavilySearchResult(
                ok=False,
                query=q,
                answer="",
                results=[],
                error=_extract_error_message(body, int(response.status_code)),
                status_code=int(response.status_code),
                latency_seconds=latency,
                raw=body if isinstance(body, dict) else None,
            )
        break

    answer = ""
    if isinstance(body, dict):
        raw_answer = body.get("answer")
        if isinstance(raw_answer, str):
            answer = raw_answer.strip()

    normalized_results: list[dict[str, Any]] = []
    raw_results = body.get("results") if isinstance(body, dict) else None
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            normalized_results.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("url") or "").strip(),
                    "content": str(item.get("content") or "").strip(),
                    "score": item.get("score"),
                    "published_date": item.get("published_date"),
                }
            )

    return TavilySearchResult(
        ok=True,
        query=q,
        answer=answer,
        results=normalized_results,
        status_code=int(response.status_code),
        latency_seconds=latency,
        raw=body if isinstance(body, dict) else None,
    )


def get_tavily_usage(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_seconds: float = 15.0,
    request_retries: int = 1,
) -> TavilyUsageResult:
    """
    Probe Tavily API connectivity via the Usage endpoint.

    Docs reference:
    - GET https://api.tavily.com/usage
    - Authorization header: Bearer <TAVILY_API_KEY>
    """

    resolved_key = _resolve_tavily_api_key(api_key)
    if not resolved_key:
        return TavilyUsageResult(
            ok=False,
            error="missing_tavily_api_key",
        )

    url = _normalize_api_root(base_url) + "/usage"
    headers = {
        "Accept": "application/json",
        "User-Agent": "pmx-tavily-client/1.0",
        "Authorization": f"Bearer {resolved_key}",
    }
    started = time.monotonic()
    retries = max(0, min(4, int(request_retries)))
    attempt = 0

    while True:
        try:
            response = requests.get(
                url,
                headers=headers,
                timeout=max(1.0, float(timeout_seconds)),
            )
        except Exception as exc:
            if attempt < retries:
                time.sleep(_retry_delay_seconds(attempt))
                attempt += 1
                continue
            return TavilyUsageResult(
                ok=False,
                error=f"request_failed: {exc}",
                latency_seconds=round(time.monotonic() - started, 3),
            )

        latency = round(time.monotonic() - started, 3)
        try:
            body = response.json()
        except Exception:
            body = {}

        if int(response.status_code) >= 400:
            if _is_retryable_status(int(response.status_code)) and attempt < retries:
                time.sleep(_retry_delay_seconds(attempt))
                attempt += 1
                continue
            return TavilyUsageResult(
                ok=False,
                error=_extract_error_message(body, int(response.status_code)),
                status_code=int(response.status_code),
                latency_seconds=latency,
                raw=body if isinstance(body, dict) else None,
            )

        key_payload = body.get("key") if isinstance(body, dict) else None
        account_payload = body.get("account") if isinstance(body, dict) else None
        return TavilyUsageResult(
            ok=True,
            status_code=int(response.status_code),
            latency_seconds=latency,
            key=key_payload if isinstance(key_payload, dict) else None,
            account=account_payload if isinstance(account_payload, dict) else None,
            raw=body if isinstance(body, dict) else None,
        )
