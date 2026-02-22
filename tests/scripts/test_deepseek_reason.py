from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import scripts.deepseek_reason as mod


def test_extract_thinking_block() -> None:
    raw = "<think>internal reasoning</think>\nfinal answer"
    assert mod._extract_thinking(raw) == "internal reasoning"


def test_reason_handles_connection_error(monkeypatch) -> None:
    class _ConnError(Exception):
        pass

    class _Timeout(Exception):
        pass

    def _post(*args, **kwargs):
        raise _ConnError("offline")

    fake_requests = SimpleNamespace(
        post=_post,
        ConnectionError=_ConnError,
        Timeout=_Timeout,
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    out = mod.reason("ping", model="deepseek-r1:8b")
    assert out.get("ok") is False
    assert "Cannot connect to Ollama" in str(out.get("error"))


def test_main_context_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    rc = mod.main(["--context-file", str(missing), "hello"])
    assert rc == 1


def test_main_json_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        mod,
        "reason",
        lambda *args, **kwargs: {
            "ok": True,
            "model": "deepseek-r1:8b",
            "response": "analysis",
            "thinking": None,
            "total_duration_ms": 10,
            "eval_count": 7,
        },
    )
    rc = mod.main(["--json", "hello"])
    captured = capsys.readouterr()
    assert rc == 0
    assert '"ok": true' in captured.out.lower()
    assert '"response": "analysis"' in captured.out

