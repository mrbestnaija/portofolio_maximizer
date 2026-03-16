from __future__ import annotations

from scripts import openclaw_models as mod


def test_ollama_api_base_defaults_to_native_endpoint() -> None:
    assert mod._ollama_api_base_from_configured("") == "http://127.0.0.1:11434"
    assert mod._ollama_api_base_from_configured("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434"


def test_default_model_order_prefers_qwen35_when_available() -> None:
    order = mod._default_ollama_model_order()
    assert order[0] == "qwen3.5:27b"
    assert "qwen3:8b" in order


def test_promote_tool_primary_accepts_qwen35() -> None:
    promoted = mod._promote_tool_primary(["deepseek-r1:8b", "qwen3.5:27b", "qwen3:8b"])
    assert promoted[0] == "qwen3.5:27b"


def test_build_fallbacks_keeps_only_tool_capable_local_models() -> None:
    fallbacks = mod._build_fallbacks(
        local_models_ordered=["qwen3.5:27b", "qwen3:8b", "deepseek-r1:8b", "deepseek-r1:32b"],
        include_remote_qwen=False,
        include_openai=False,
        include_anthropic=False,
        openai_model="gpt-4o-mini",
        anthropic_model="claude-sonnet-4-6",
    )
    assert fallbacks == ["ollama/qwen3.5:27b", "ollama/qwen3:8b"]
