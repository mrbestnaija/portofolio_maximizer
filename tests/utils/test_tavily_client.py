from __future__ import annotations

from typing import Any

from utils import tavily_client as tc


class _FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def test_search_tavily_missing_query() -> None:
    out = tc.search_tavily(query="")
    assert out.ok is False
    assert out.error == "missing_query"


def test_search_tavily_missing_api_key(monkeypatch) -> None:
    monkeypatch.setattr(tc, "_resolve_tavily_api_key", lambda _: None)
    out = tc.search_tavily(query="AAPL outlook")
    assert out.ok is False
    assert out.error == "missing_tavily_api_key"


def test_search_tavily_success(monkeypatch) -> None:
    monkeypatch.setattr(tc, "_resolve_tavily_api_key", lambda _: "tvly-abc123")

    def _fake_post(url, json, headers, timeout):  # noqa: ANN001
        assert url.endswith("/search")
        assert json["query"] == "latest market data"
        assert "api_key" not in json
        assert headers.get("Authorization") == "Bearer tvly-abc123"
        assert headers.get("Content-Type") == "application/json"
        assert timeout == 12.0
        return _FakeResponse(
            200,
            {
                "answer": "Markets are mixed.",
                "results": [
                    {
                        "title": "Sample headline",
                        "url": "https://example.com/news",
                        "content": "Snippet",
                        "score": 0.91,
                    }
                ],
            },
        )

    monkeypatch.setattr(tc.requests, "post", _fake_post)

    out = tc.search_tavily(
        query="latest market data",
        timeout_seconds=12.0,
        max_results=3,
    )
    assert out.ok is True
    assert out.answer == "Markets are mixed."
    assert len(out.results) == 1
    assert out.results[0]["url"] == "https://example.com/news"


def test_search_tavily_invalid_depth_normalizes_to_basic(monkeypatch) -> None:
    monkeypatch.setattr(tc, "_resolve_tavily_api_key", lambda _: "tvly-abc123")

    def _fake_post(url, json, headers, timeout):  # noqa: ANN001
        assert json["search_depth"] == "basic"
        return _FakeResponse(200, {"answer": "", "results": []})

    monkeypatch.setattr(tc.requests, "post", _fake_post)
    out = tc.search_tavily(query="latest market data", search_depth="not-valid")
    assert out.ok is True


def test_search_tavily_http_error(monkeypatch) -> None:
    monkeypatch.setattr(tc, "_resolve_tavily_api_key", lambda _: "tvly-abc123")
    monkeypatch.setattr(
        tc.requests,
        "post",
        lambda *args, **kwargs: _FakeResponse(429, {"error": "quota exceeded"}),
    )

    out = tc.search_tavily(query="fed meeting")
    assert out.ok is False
    assert out.status_code == 429
    assert "http_429" in (out.error or "")


def test_search_tavily_base_url_normalizes_endpoint_suffix(monkeypatch) -> None:
    monkeypatch.setattr(tc, "_resolve_tavily_api_key", lambda _: "tvly-abc123")
    observed: dict[str, str] = {}

    def _fake_post(url, json, headers, timeout):  # noqa: ANN001
        observed["url"] = url
        return _FakeResponse(200, {"answer": "", "results": []})

    monkeypatch.setattr(tc.requests, "post", _fake_post)
    out = tc.search_tavily(query="earnings", base_url="https://api.tavily.com/search")
    assert out.ok is True
    assert observed["url"] == "https://api.tavily.com/search"


def test_get_tavily_usage_missing_api_key(monkeypatch) -> None:
    monkeypatch.setattr(tc, "_resolve_tavily_api_key", lambda _: None)
    out = tc.get_tavily_usage()
    assert out.ok is False
    assert out.error == "missing_tavily_api_key"


def test_get_tavily_usage_success(monkeypatch) -> None:
    monkeypatch.setattr(tc, "_resolve_tavily_api_key", lambda _: "tvly-abc123")

    def _fake_get(url, headers, timeout):  # noqa: ANN001
        assert url == "https://api.tavily.com/usage"
        assert headers.get("Authorization") == "Bearer tvly-abc123"
        assert timeout == 9.0
        return _FakeResponse(
            200,
            {
                "key": {"usage": 3, "limit": 1000},
                "account": {"current_plan": "Bootstrap"},
            },
        )

    monkeypatch.setattr(tc.requests, "get", _fake_get)
    out = tc.get_tavily_usage(timeout_seconds=9.0)
    assert out.ok is True
    assert out.key is not None
    assert out.key.get("usage") == 3
