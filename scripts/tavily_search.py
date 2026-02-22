#!/usr/bin/env python3
"""
Search the web using Tavily with free-provider fallback and print concise results.

This is intended as a PMX-safe replacement for Brave-based `web_search`
workflows when Brave quota is constrained.

Default provider order:
  tavily -> duckduckgo -> wikipedia
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.web_search_client import search_web_multi  # noqa: E402


def _read_stdin() -> str:
    try:
        if sys.stdin is None or sys.stdin.closed or sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except Exception:
        return ""


def _clamp(text: str, limit: int) -> str:
    value = (text or "").strip()
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 10)] + "...[trunc]"


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = (os.getenv(name) or "").strip().lower()
    if raw in allowed:
        return raw
    return default


def main(argv: list[str]) -> int:
    # Load repo .env (best effort) without printing secret values.
    try:
        from etl.secret_loader import bootstrap_dotenv

        bootstrap_dotenv()
    except Exception:
        pass

    p = argparse.ArgumentParser(description="Run robust web search (Tavily + free fallbacks).")
    p.add_argument("--query", default="", help="Search query. If omitted, reads from stdin.")
    p.add_argument("--json", action="store_true", help="Output JSON.")
    p.add_argument("--max-results", type=int, default=_env_int("TAVILY_MAX_RESULTS", 5))
    p.add_argument(
        "--search-depth",
        choices=("basic", "advanced", "fast", "ultra-fast"),
        default=_env_choice("TAVILY_SEARCH_DEPTH", "basic", ("basic", "advanced", "fast", "ultra-fast")),
    )
    p.add_argument(
        "--topic",
        choices=("general", "news", "finance"),
        default=_env_choice("TAVILY_TOPIC", "general", ("general", "news", "finance")),
    )
    p.add_argument("--include-raw-content", action="store_true", help="Request raw content in Tavily response.")
    p.add_argument("--include-images", action="store_true", help="Request image metadata in Tavily response.")
    p.add_argument("--include-domains", default="", help="Comma-separated domain allowlist.")
    p.add_argument("--exclude-domains", default="", help="Comma-separated domain blocklist.")
    p.add_argument(
        "--providers",
        default=(os.getenv("PMX_WEB_SEARCH_PROVIDERS") or "").strip(),
        help="Comma-separated provider order (tavily,duckduckgo,wikipedia).",
    )
    p.add_argument("--base-url", default=(os.getenv("TAVILY_BASE_URL") or "").strip())
    p.add_argument("--timeout-seconds", type=float, default=_env_float("TAVILY_TIMEOUT_SECONDS", 20.0))
    p.add_argument("--max-snippet-chars", type=int, default=220)
    args = p.parse_args(argv)

    query = (args.query or "").strip() or _read_stdin().strip()
    if not query:
        print("[tavily_search] Missing --query (and stdin was empty).", file=sys.stderr)
        return 2

    result = search_web_multi(
        query=query,
        max_results=int(args.max_results),
        search_depth=str(args.search_depth or "basic"),
        topic=str(args.topic or "general"),
        include_answer=True,
        include_raw_content=bool(args.include_raw_content),
        include_images=bool(args.include_images),
        include_domains_csv=(args.include_domains or None),
        exclude_domains_csv=(args.exclude_domains or None),
        tavily_base_url=(args.base_url or None),
        timeout_seconds=float(args.timeout_seconds),
        providers_csv=(args.providers or None),
    )

    payload = {
        "ok": bool(result.ok),
        "provider": result.provider,
        "query": result.query,
        "answer": result.answer,
        "results": result.results,
        "attempts": result.attempts,
        "error": result.error,
        "status_code": result.status_code,
        "latency_seconds": result.latency_seconds,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        if not result.ok:
            print(f"[tavily_search] FAILED: {result.error or 'unknown_error'}", file=sys.stderr)
            return 1
        print(f"Provider: {result.provider}")
        if result.answer:
            print(f"Answer: {result.answer}")
            print("")
        if not result.results:
            print("(no results)")
            return 0
        print("Top results:")
        for idx, item in enumerate(result.results, start=1):
            title = _clamp(str(item.get("title") or ""), 160)
            url = str(item.get("url") or "").strip()
            content = _clamp(str(item.get("content") or ""), int(args.max_snippet_chars))
            line = f"{idx}. {title}"
            if url:
                line += f" - {url}"
            print(line)
            if content:
                print(f"   {content}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
