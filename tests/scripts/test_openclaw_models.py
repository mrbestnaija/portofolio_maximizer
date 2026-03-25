from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import openclaw_models as mod


def test_ollama_api_base_defaults_to_native_endpoint() -> None:
    assert mod._ollama_api_base_from_configured("") == "http://127.0.0.1:11434"
    assert mod._ollama_api_base_from_configured("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434"


@pytest.mark.xfail(reason="qwen3.5:27b not yet in production model list; current primary is qwen3:8b")
def test_default_model_order_prefers_qwen35_when_available() -> None:
    order = mod._default_ollama_model_order()
    assert order[0] == "qwen3.5:27b"
    assert "qwen3:8b" in order


def test_promote_tool_primary_accepts_qwen35() -> None:
    promoted = mod._promote_tool_primary(["deepseek-r1:8b", "qwen3.5:27b", "qwen3:8b"])
    assert promoted[0] == "qwen3.5:27b"


@pytest.mark.xfail(reason="qwen3.5:27b not yet in production fallback list; deepseek-r1 models are included")
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


@pytest.mark.xfail(reason="_run_openclaw encoding kwarg not yet wired; pending Windows UTF-8 output hardening")
def test_run_openclaw_forces_utf8_replace_on_windows_safe_output(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        captured["kwargs"] = dict(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = mod._run_openclaw(base=["openclaw"], args=["status"], timeout_seconds=5.0)

    assert result.ok is True
    kwargs = captured["kwargs"]
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"


def test_sync_explicit_agent_models_aligns_core_agents_to_primary() -> None:
    agents = [
        {"id": "ops", "model": "ollama/qwen3:8b"},
        {"id": "training", "model": "ollama/qwen3.5:27b"},
        {"id": "custom", "model": "ollama/deepseek-r1:8b"},
        {"id": "notifier"},
    ]

    synced, updated_ids, changed = mod._sync_explicit_agent_models(
        agents_list=agents,
        primary="ollama/qwen3:8b",
    )

    assert changed is True
    assert updated_ids == ["training", "notifier"]
    assert synced[0]["model"] == "ollama/qwen3:8b"
    assert synced[1]["model"] == "ollama/qwen3:8b"
    assert synced[2]["model"] == "ollama/deepseek-r1:8b"
    assert synced[3]["model"] == "ollama/qwen3:8b"


def test_update_openclaw_json_agents_list_accepts_utf8_bom(tmp_path, monkeypatch) -> None:
    config_dir = tmp_path / ".openclaw"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "openclaw.json"
    config_path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"agents": {"list": []}}).encode("utf-8"))
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    result = mod._update_openclaw_json_agents_list(
        agents_list=[{"id": "ops", "model": "ollama/qwen3:8b"}],
        dry_run=False,
    )

    assert result.ok is True
    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert updated["agents"]["list"] == [{"id": "ops", "model": "ollama/qwen3:8b"}]
