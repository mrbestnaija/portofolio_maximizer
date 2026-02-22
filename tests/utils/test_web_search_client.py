from __future__ import annotations

from typing import Any

from utils import web_search_client as wc
from utils.tavily_client import TavilySearchResult


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def test_search_web_multi_missing_query() -> None:
    out = wc.search_web_multi(query="")
    assert out.ok is False
    assert out.error == "missing_query"


def test_search_web_multi_duckduckgo_success(monkeypatch) -> None:
    monkeypatch.setattr(
        wc.requests,
        "get",
        lambda *args, **kwargs: _FakeResponse(
            200,
            {
                "AbstractText": "",
                "RelatedTopics": [
                    {"Text": "Example result", "FirstURL": "https://example.com/result"}
                ],
            },
        ),
    )

    out = wc.search_web_multi(
        query="example search",
        providers_csv="duckduckgo",
        max_results=3,
    )
    assert out.ok is True
    assert out.provider == "duckduckgo"
    assert len(out.results) >= 1
    assert out.results[0]["source"] == "duckduckgo"


def test_search_web_multi_tavily_fallback_to_wikipedia(monkeypatch) -> None:
    monkeypatch.setattr(
        wc,
        "search_tavily",
        lambda **kwargs: TavilySearchResult(
            ok=False,
            query=str(kwargs.get("query") or ""),
            answer="",
            results=[],
            error="missing_tavily_api_key",
        ),
    )

    def _fake_get(url, params=None, headers=None, timeout=0):  # noqa: ANN001
        assert "wikipedia.org" in url
        assert isinstance(headers, dict)
        assert "User-Agent" in headers
        return _FakeResponse(
            200,
            {
                "query": {
                    "search": [
                        {
                            "title": "Python",
                            "snippet": "Python is a language",
                            "pageid": 23862,
                        }
                    ]
                }
            },
        )

    monkeypatch.setattr(wc.requests, "get", _fake_get)

    out = wc.search_web_multi(
        query="python language",
        providers_csv="tavily,wikipedia",
        max_results=2,
    )
    assert out.ok is True
    assert out.provider == "wikipedia"
    assert len(out.attempts) == 2
    assert out.attempts[0]["provider"] == "tavily"
    assert out.attempts[0]["ok"] is False
    assert out.attempts[1]["provider"] == "wikipedia"
    assert out.attempts[1]["ok"] is True


def test_search_web_multi_all_fail(monkeypatch) -> None:
    monkeypatch.setattr(
        wc,
        "search_tavily",
        lambda **kwargs: TavilySearchResult(
            ok=False,
            query=str(kwargs.get("query") or ""),
            answer="",
            results=[],
            error="missing_tavily_api_key",
        ),
    )

    def _fail_get(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("network down")

    monkeypatch.setattr(wc.requests, "get", _fail_get)

    out = wc.search_web_multi(
        query="market update",
        providers_csv="tavily,duckduckgo,wikipedia",
    )
    assert out.ok is False
    assert out.provider == "none"
    assert len(out.attempts) == 3
