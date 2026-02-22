from __future__ import annotations

"""
Robust web search client with provider fallback.

Provider order defaults to: tavily -> duckduckgo -> wikipedia
and can be overridden via:
  - function argument `providers_csv`
  - env `PMX_WEB_SEARCH_PROVIDERS`
"""

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from utils.tavily_client import search_tavily


ALLOWED_PROVIDERS = {"tavily", "duckduckgo", "wikipedia"}
DEFAULT_PROVIDER_ORDER = ("tavily", "duckduckgo", "wikipedia")


@dataclass(frozen=True)
class WebSearchResult:
    ok: bool
    provider: str
    query: str
    answer: str
    results: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    error: Optional[str] = None
    status_code: Optional[int] = None
    latency_seconds: Optional[float] = None


def _provider_order(providers_csv: Optional[str]) -> list[str]:
    raw = (providers_csv or "").strip()
    if not raw:
        raw = (os.getenv("PMX_WEB_SEARCH_PROVIDERS") or "").strip()

    if not raw:
        return list(DEFAULT_PROVIDER_ORDER)

    out: list[str] = []
    for piece in re.split(r"[,;\n|]", raw):
        name = str(piece or "").strip().lower()
        if not name or name not in ALLOWED_PROVIDERS:
            continue
        if name not in out:
            out.append(name)
    return out or list(DEFAULT_PROVIDER_ORDER)


def _strip_html(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _attempt_row(
    *,
    provider: str,
    ok: bool,
    error: Optional[str],
    status_code: Optional[int],
    latency_seconds: Optional[float],
    result_count: int,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "ok": bool(ok),
        "error": str(error or "") or None,
        "status_code": status_code,
        "latency_seconds": latency_seconds,
        "result_count": int(result_count),
    }


def _search_tavily_provider(
    *,
    query: str,
    max_results: int,
    search_depth: str,
    topic: str,
    include_answer: bool,
    include_raw_content: bool,
    include_images: bool,
    include_domains_csv: Optional[str],
    exclude_domains_csv: Optional[str],
    tavily_base_url: Optional[str],
    timeout_seconds: float,
) -> WebSearchResult:
    res = search_tavily(
        query=query,
        base_url=tavily_base_url,
        max_results=max_results,
        search_depth=search_depth,
        topic=topic,
        include_answer=bool(include_answer),
        include_raw_content=bool(include_raw_content),
        include_images=bool(include_images),
        include_domains_csv=include_domains_csv,
        exclude_domains_csv=exclude_domains_csv,
        timeout_seconds=timeout_seconds,
    )
    normalized_results: list[dict[str, Any]] = []
    for item in res.results:
        if not isinstance(item, dict):
            continue
        normalized_results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": str(item.get("url") or "").strip(),
                "content": str(item.get("content") or "").strip(),
                "score": item.get("score"),
                "published_date": item.get("published_date"),
                "source": "tavily",
            }
        )
    return WebSearchResult(
        ok=bool(res.ok and (normalized_results or res.answer)),
        provider="tavily",
        query=query,
        answer=str(res.answer or "").strip(),
        results=normalized_results,
        attempts=[],
        error=res.error,
        status_code=res.status_code,
        latency_seconds=res.latency_seconds,
    )


def _search_duckduckgo_provider(
    *,
    query: str,
    max_results: int,
    timeout_seconds: float,
) -> WebSearchResult:
    started = time.monotonic()
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "no_redirect": 1,
                "skip_disambig": 1,
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "pmx-web-search/1.0 (+https://github.com/mrbestnaija/portofolio_maximizer)",
            },
            timeout=max(1.0, float(timeout_seconds)),
        )
        latency = round(time.monotonic() - started, 3)
    except Exception as exc:
        return WebSearchResult(
            ok=False,
            provider="duckduckgo",
            query=query,
            answer="",
            results=[],
            attempts=[],
            error=f"request_failed: {exc}",
            latency_seconds=round(time.monotonic() - started, 3),
        )

    status_code = int(resp.status_code)
    if status_code >= 400:
        return WebSearchResult(
            ok=False,
            provider="duckduckgo",
            query=query,
            answer="",
            results=[],
            attempts=[],
            error=f"http_{status_code}",
            status_code=status_code,
            latency_seconds=latency,
        )

    try:
        body = resp.json()
    except Exception as exc:
        return WebSearchResult(
            ok=False,
            provider="duckduckgo",
            query=query,
            answer="",
            results=[],
            attempts=[],
            error=f"invalid_json: {exc}",
            status_code=status_code,
            latency_seconds=latency,
        )

    answer = str(body.get("AbstractText") or "").strip() if isinstance(body, dict) else ""
    results: list[dict[str, Any]] = []

    if isinstance(body, dict):
        abstract_url = str(body.get("AbstractURL") or "").strip()
        heading = str(body.get("Heading") or "").strip()
        if answer or abstract_url:
            results.append(
                {
                    "title": heading or "DuckDuckGo Abstract",
                    "url": abstract_url,
                    "content": answer,
                    "score": None,
                    "published_date": None,
                    "source": "duckduckgo",
                }
            )

        def _walk_topics(rows: list[Any]) -> None:
            for row in rows:
                if len(results) >= max_results:
                    return
                if not isinstance(row, dict):
                    continue
                if isinstance(row.get("Topics"), list):
                    _walk_topics(row.get("Topics"))
                    continue
                text = str(row.get("Text") or "").strip()
                url = str(row.get("FirstURL") or "").strip()
                if not text and not url:
                    continue
                results.append(
                    {
                        "title": text[:140] or "DuckDuckGo Result",
                        "url": url,
                        "content": text,
                        "score": None,
                        "published_date": None,
                        "source": "duckduckgo",
                    }
                )

        related = body.get("RelatedTopics")
        if isinstance(related, list):
            _walk_topics(related)

    results = results[:max_results]
    return WebSearchResult(
        ok=bool(results or answer),
        provider="duckduckgo",
        query=query,
        answer=answer,
        results=results,
        attempts=[],
        error=None if (results or answer) else "no_results",
        status_code=status_code,
        latency_seconds=latency,
    )


def _search_wikipedia_provider(
    *,
    query: str,
    max_results: int,
    timeout_seconds: float,
) -> WebSearchResult:
    started = time.monotonic()
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max(1, min(20, int(max_results))),
                "utf8": 1,
                "format": "json",
            },
            headers={
                "Accept": "application/json",
                "User-Agent": "pmx-web-search/1.0 (+https://github.com/mrbestnaija/portofolio_maximizer)",
            },
            timeout=max(1.0, float(timeout_seconds)),
        )
        latency = round(time.monotonic() - started, 3)
    except Exception as exc:
        return WebSearchResult(
            ok=False,
            provider="wikipedia",
            query=query,
            answer="",
            results=[],
            attempts=[],
            error=f"request_failed: {exc}",
            latency_seconds=round(time.monotonic() - started, 3),
        )

    status_code = int(resp.status_code)
    if status_code >= 400:
        return WebSearchResult(
            ok=False,
            provider="wikipedia",
            query=query,
            answer="",
            results=[],
            attempts=[],
            error=f"http_{status_code}",
            status_code=status_code,
            latency_seconds=latency,
        )

    try:
        body = resp.json()
    except Exception as exc:
        return WebSearchResult(
            ok=False,
            provider="wikipedia",
            query=query,
            answer="",
            results=[],
            attempts=[],
            error=f"invalid_json: {exc}",
            status_code=status_code,
            latency_seconds=latency,
        )

    results: list[dict[str, Any]] = []
    search_rows = (
        body.get("query", {}).get("search")
        if isinstance(body, dict) and isinstance(body.get("query"), dict)
        else None
    )
    if isinstance(search_rows, list):
        for row in search_rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            snippet = _strip_html(str(row.get("snippet") or ""))
            page_id = row.get("pageid")
            url = f"https://en.wikipedia.org/?curid={page_id}" if page_id else ""
            results.append(
                {
                    "title": title,
                    "url": url,
                    "content": snippet,
                    "score": None,
                    "published_date": None,
                    "source": "wikipedia",
                }
            )
            if len(results) >= max_results:
                break

    answer = results[0]["content"] if results else ""
    return WebSearchResult(
        ok=bool(results),
        provider="wikipedia",
        query=query,
        answer=answer,
        results=results,
        attempts=[],
        error=None if results else "no_results",
        status_code=status_code,
        latency_seconds=latency,
    )


def search_web_multi(
    *,
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    topic: str = "general",
    include_answer: bool = True,
    include_raw_content: bool = False,
    include_images: bool = False,
    include_domains_csv: Optional[str] = None,
    exclude_domains_csv: Optional[str] = None,
    tavily_base_url: Optional[str] = None,
    timeout_seconds: float = 20.0,
    providers_csv: Optional[str] = None,
) -> WebSearchResult:
    q = str(query or "").strip()
    if not q:
        return WebSearchResult(
            ok=False,
            provider="none",
            query=q,
            answer="",
            results=[],
            attempts=[],
            error="missing_query",
        )

    max_results = max(1, min(10, int(max_results)))
    timeout_seconds = max(1.0, float(timeout_seconds))
    providers = _provider_order(providers_csv)
    attempts: list[dict[str, Any]] = []
    last_error = "no_provider_attempted"

    for provider in providers:
        if provider == "tavily":
            row = _search_tavily_provider(
                query=q,
                max_results=max_results,
                search_depth=search_depth,
                topic=topic,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_images=include_images,
                include_domains_csv=include_domains_csv,
                exclude_domains_csv=exclude_domains_csv,
                tavily_base_url=tavily_base_url,
                timeout_seconds=timeout_seconds,
            )
        elif provider == "duckduckgo":
            row = _search_duckduckgo_provider(
                query=q,
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            )
        elif provider == "wikipedia":
            row = _search_wikipedia_provider(
                query=q,
                max_results=max_results,
                timeout_seconds=timeout_seconds,
            )
        else:
            continue

        attempts.append(
            _attempt_row(
                provider=row.provider,
                ok=row.ok,
                error=row.error,
                status_code=row.status_code,
                latency_seconds=row.latency_seconds,
                result_count=len(row.results),
            )
        )

        if row.ok:
            return WebSearchResult(
                ok=True,
                provider=row.provider,
                query=q,
                answer=row.answer,
                results=row.results,
                attempts=attempts,
                error=None,
                status_code=row.status_code,
                latency_seconds=row.latency_seconds,
            )

        if row.error:
            last_error = row.error

    return WebSearchResult(
        ok=False,
        provider="none",
        query=q,
        answer="",
        results=[],
        attempts=attempts,
        error=last_error,
        status_code=None,
        latency_seconds=None,
    )
