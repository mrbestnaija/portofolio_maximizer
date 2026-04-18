from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import openclaw_models as mod


def test_ollama_api_base_defaults_to_native_endpoint() -> None:
    assert mod._ollama_api_base_from_configured("") == "http://127.0.0.1:11434"
    assert mod._ollama_api_base_from_configured("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434"
    assert mod._ollama_api_base_from_configured("http://127.0.0.1:11434/") == "http://127.0.0.1:11434"


def test_default_model_order_prefers_qwen3_first() -> None:
    order = mod._default_ollama_model_order()
    assert order[0] == "qwen3:8b"
    assert "qwen3:8b" in order


def test_promote_tool_primary_prefers_canonical_qwen3_even_when_qwen35_is_listed_first() -> None:
    promoted = mod._promote_tool_primary(["deepseek-r1:8b", "qwen3.5:27b", "qwen3:8b"])
    assert promoted[0] == "qwen3:8b"
    assert promoted[1] == "deepseek-r1:8b"


def test_promote_tool_primary_drops_qwen35_variants_from_safe_chain() -> None:
    promoted = mod._promote_tool_primary(["qwen3.5:27b", "deepseek-r1:8b"])
    assert promoted == ["deepseek-r1:8b"]


def test_build_fallbacks_preserves_local_order_for_canonical_models() -> None:
    fallbacks = mod._build_fallbacks(
        local_models_ordered=["qwen3:8b", "deepseek-r1:8b", "deepseek-r1:32b"],
        include_remote_qwen=False,
        include_openai=False,
        include_anthropic=False,
        openai_model="gpt-4o-mini",
        anthropic_model="claude-sonnet-4-6",
    )
    assert fallbacks == ["ollama/qwen3:8b", "ollama/deepseek-r1:8b", "ollama/deepseek-r1:32b"]


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


def test_cmd_apply_normalizes_legacy_ollama_base_url_before_writing(monkeypatch) -> None:
    writes: list[tuple[str, object]] = []

    monkeypatch.setattr(mod, "_bootstrap_dotenv", lambda: None)
    monkeypatch.setattr(mod, "_split_command", lambda command: ["openclaw"])
    monkeypatch.setattr(mod, "_detect_default_agent_id", lambda oc_base: "ops")
    monkeypatch.setattr(mod, "_auth_store_path_for_agent", lambda agent_id: Path("auth-store.json"))
    monkeypatch.setattr(mod, "_load_secret", lambda name: None)
    monkeypatch.setattr(
        mod,
        "_sync_auth_store",
        lambda **kwargs: (False, []),
    )
    monkeypatch.setattr(mod, "_discover_ollama_models", lambda base_url, timeout_seconds=2.0: ["qwen3:8b"])

    def fake_set_json(*, oc_base, path, value, timeout_seconds, dry_run):
        writes.append((path, value))
        return mod._CmdResult(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_oc_config_set_json", fake_set_json)
    monkeypatch.setattr(mod, "_restart_gateway", lambda **kwargs: mod._CmdResult(ok=True, returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(mod, "_build_fallbacks", lambda **kwargs: ["ollama/deepseek-r1:8b"])
    monkeypatch.setattr(mod, "_sync_explicit_agent_models", lambda agents_list, primary: (agents_list, [], False))
    monkeypatch.setattr(mod, "_oc_config_get_json", lambda **kwargs: [])
    monkeypatch.setattr(mod, "_update_openclaw_json_agents_list", lambda **kwargs: mod._CmdResult(ok=True, returncode=0, stdout="", stderr=""))

    args = SimpleNamespace(
        command="openclaw",
        agent_id="ops",
        dry_run=False,
        sync_auth=False,
        ollama_base_url="http://127.0.0.1:11434/v1",
        ollama_models="qwen3:8b",
        enable_ollama_provider=True,
        openai_models="",
        enable_openai_provider=False,
        anthropic_models="",
        enable_anthropic_provider=False,
        enable_remote_qwen=False,
        strategy="local-first",
        openai_primary_model="gpt-4o-mini",
        anthropic_primary_model="claude-sonnet-4-6",
        openai_model="gpt-4o-mini",
        anthropic_model="claude-sonnet-4-6",
        remote_qwen_text_model="qwen-portal/coder-model",
        remote_qwen_image_model="qwen-portal/vision-model",
        set_image_defaults=False,
        image_primary="",
        openai_image_model="gpt-4o",
        restart_gateway=False,
    )

    rc = mod._cmd_apply(args)

    assert rc == 0
    provider = next(value for path, value in writes if path == "models.providers.ollama")
    assert provider["baseUrl"] == "http://127.0.0.1:11434"


def test_cmd_apply_prunes_remote_allowlist_when_local_only(monkeypatch) -> None:
    writes: list[tuple[str, object]] = []

    monkeypatch.setattr(mod, "_bootstrap_dotenv", lambda: None)
    monkeypatch.setattr(mod, "_split_command", lambda command: ["openclaw"])
    monkeypatch.setattr(mod, "_detect_default_agent_id", lambda oc_base: "ops")
    monkeypatch.setattr(mod, "_auth_store_path_for_agent", lambda agent_id: Path("auth-store.json"))
    monkeypatch.setattr(mod, "_load_secret", lambda name: None)
    monkeypatch.setattr(
        mod,
        "_sync_auth_store",
        lambda **kwargs: (False, []),
    )
    monkeypatch.setattr(mod, "_discover_ollama_models", lambda base_url, timeout_seconds=2.0: ["qwen3:8b"])

    def fake_set_json(*, oc_base, path, value, timeout_seconds, dry_run):
        writes.append((path, value))
        return mod._CmdResult(ok=True, returncode=0, stdout="", stderr="")

    def fake_get_json(*, oc_base, path, timeout_seconds):
        if path == "agents.defaults.models":
            return {
                "ollama/qwen3:8b": {},
                "qwen-portal/coder-model": {"alias": "qwen"},
                "qwen-portal/vision-model": {},
            }
        return []

    monkeypatch.setattr(mod, "_oc_config_set_json", fake_set_json)
    monkeypatch.setattr(mod, "_restart_gateway", lambda **kwargs: mod._CmdResult(ok=True, returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(mod, "_build_fallbacks", lambda **kwargs: ["ollama/deepseek-r1:8b"])
    monkeypatch.setattr(mod, "_sync_explicit_agent_models", lambda agents_list, primary: (agents_list, [], False))
    monkeypatch.setattr(mod, "_oc_config_get_json", fake_get_json)
    monkeypatch.setattr(mod, "_update_openclaw_json_agents_list", lambda **kwargs: mod._CmdResult(ok=True, returncode=0, stdout="", stderr=""))

    args = SimpleNamespace(
        command="openclaw",
        agent_id="ops",
        dry_run=False,
        sync_auth=False,
        ollama_base_url="http://127.0.0.1:11434",
        ollama_models="qwen3:8b",
        enable_ollama_provider=True,
        openai_models="",
        enable_openai_provider=False,
        anthropic_models="",
        enable_anthropic_provider=False,
        enable_remote_qwen=False,
        strategy="local-first",
        openai_primary_model="gpt-4o-mini",
        anthropic_primary_model="claude-sonnet-4-6",
        openai_model="gpt-4o-mini",
        anthropic_model="claude-sonnet-4-6",
        remote_qwen_text_model="qwen-portal/coder-model",
        remote_qwen_image_model="qwen-portal/vision-model",
        set_image_defaults=False,
        image_primary="",
        openai_image_model="gpt-4o",
        restart_gateway=False,
    )

    rc = mod._cmd_apply(args)

    assert rc == 0
    allowlist = next(value for path, value in writes if path == "agents.defaults.models")
    assert "qwen-portal/coder-model" not in allowlist
    assert "qwen-portal/vision-model" not in allowlist
    assert set(allowlist.keys()) == {"ollama/qwen3:8b", "ollama/deepseek-r1:8b"}


def test_cmd_apply_filters_qwen35_variants_from_local_only_ollama_models(monkeypatch) -> None:
    writes: list[tuple[str, object]] = []

    monkeypatch.setattr(mod, "_bootstrap_dotenv", lambda: None)
    monkeypatch.setattr(mod, "_split_command", lambda command: ["openclaw"])
    monkeypatch.setattr(mod, "_detect_default_agent_id", lambda oc_base: "ops")
    monkeypatch.setattr(mod, "_auth_store_path_for_agent", lambda agent_id: Path("auth-store.json"))
    monkeypatch.setattr(mod, "_load_secret", lambda name: None)
    monkeypatch.setattr(mod, "_sync_auth_store", lambda **kwargs: (False, []))
    monkeypatch.setattr(mod, "_discover_ollama_models", lambda base_url, timeout_seconds=2.0: ["qwen3.5:27b", "qwen3:8b"])

    def fake_set_json(*, oc_base, path, value, timeout_seconds, dry_run):
        writes.append((path, value))
        return mod._CmdResult(ok=True, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "_oc_config_set_json", fake_set_json)
    monkeypatch.setattr(mod, "_restart_gateway", lambda **kwargs: mod._CmdResult(ok=True, returncode=0, stdout="", stderr=""))
    monkeypatch.setattr(mod, "_sync_explicit_agent_models", lambda agents_list, primary: (agents_list, [], False))
    monkeypatch.setattr(mod, "_oc_config_get_json", lambda **kwargs: [])
    monkeypatch.setattr(mod, "_update_openclaw_json_agents_list", lambda **kwargs: mod._CmdResult(ok=True, returncode=0, stdout="", stderr=""))

    args = SimpleNamespace(
        command="openclaw",
        agent_id="ops",
        dry_run=False,
        sync_auth=False,
        ollama_base_url="http://127.0.0.1:11434/v1",
        ollama_models="",
        enable_ollama_provider=True,
        openai_models="",
        enable_openai_provider=False,
        anthropic_models="",
        enable_anthropic_provider=False,
        enable_remote_qwen=False,
        strategy="local-first",
        openai_primary_model="gpt-4o-mini",
        anthropic_primary_model="claude-sonnet-4-6",
        openai_model="gpt-4o-mini",
        anthropic_model="claude-sonnet-4-6",
        remote_qwen_text_model="qwen-portal/coder-model",
        remote_qwen_image_model="qwen-portal/vision-model",
        set_image_defaults=False,
        image_primary="",
        openai_image_model="gpt-4o",
        restart_gateway=False,
    )

    rc = mod._cmd_apply(args)

    assert rc == 0
    provider = next(value for path, value in writes if path == "models.providers.ollama")
    model_ids = [m["id"] for m in provider["models"]]
    assert "qwen3.5:27b" not in model_ids
    assert model_ids[0] == "qwen3:8b"
    model_block = next(value for path, value in writes if path == "agents.defaults.model")
    assert model_block["primary"] == "ollama/qwen3:8b"
    assert all("qwen3.5" not in str(fb) for fb in model_block.get("fallbacks", []))
