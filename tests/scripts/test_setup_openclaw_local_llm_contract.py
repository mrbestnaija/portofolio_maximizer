from __future__ import annotations

from pathlib import Path


def _script_text() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "scripts" / "setup_openclaw_local_llm.ps1"
    return path.read_text(encoding="utf-8")


def test_setup_openclaw_local_llm_uses_native_ollama_base_url() -> None:
    text = _script_text()
    assert '$resolvedOllamaHost = "http://127.0.0.1:11434"' in text
    assert "without `/v1`" in text
    assert 'if ($openclawOllamaBaseUrl.ToLower().EndsWith("/v1")) {' in text
    assert '$env:OPENCLAW_OLLAMA_BASE_URL = $openclawOllamaBaseUrl' in text
    assert '+ "/v1"' not in text
