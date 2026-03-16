from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import scripts.llm_multi_model_orchestrator as mod


def test_normalize_package_spec_rejects_direct_url_requirements() -> None:
    pkg, err = mod._normalize_package_spec("numpy@https://evil.example/simple", "")
    assert pkg == ""
    assert err is not None
    assert "Direct URL" in err


def test_normalize_index_url_requires_https() -> None:
    url, err = mod._normalize_index_url("http://pypi.org/simple", "index_url")
    assert url == ""
    assert err is not None
    assert "https://" in err


def test_normalize_index_url_rejects_disallowed_host() -> None:
    with patch.dict("scripts.llm_multi_model_orchestrator.os.environ", {}, clear=False):
        url, err = mod._normalize_index_url("https://evil.example/simple", "index_url")
    assert url == ""
    assert err is not None
    assert "not allowed" in err


def test_normalize_index_url_accepts_default_allowed_host() -> None:
    url, err = mod._normalize_index_url("https://pypi.org/simple", "index_url")
    assert err is None
    assert url == "https://pypi.org/simple"


def test_normalize_index_url_accepts_env_allowlisted_host() -> None:
    with patch.dict(
        "scripts.llm_multi_model_orchestrator.os.environ",
        {"PMX_RUNTIME_PIP_ALLOWED_INDEX_HOSTS": "packages.example.com"},
        clear=False,
    ):
        url, err = mod._normalize_index_url("https://packages.example.com/simple", "index_url")
    assert err is None
    assert url == "https://packages.example.com/simple"


def test_install_python_package_tool_disabled_by_default() -> None:
    with patch.dict(
        "scripts.llm_multi_model_orchestrator.os.environ",
        {"PMX_ALLOW_RUNTIME_PIP_INSTALL": "0"},
        clear=False,
    ):
        payload = json.loads(
            mod._install_python_package_tool(
                {
                    "package": "numpy",
                    "verify_import": "numpy",
                    "verify_only": True,
                }
            )
        )
    assert payload["status"] == "FAIL"
    assert payload["action"] == "install_python_package"
    assert "disabled by policy" in payload["error"]


def test_install_torch_runtime_tool_disabled_by_default() -> None:
    with patch.dict(
        "scripts.llm_multi_model_orchestrator.os.environ",
        {"PMX_ALLOW_RUNTIME_PIP_INSTALL": "0"},
        clear=False,
    ):
        payload = json.loads(
            mod._install_torch_runtime_tool(
                {
                    "variant": "cpu",
                    "verify_only": False,
                }
            )
        )
    assert payload["status"] == "FAIL"
    assert payload["action"] == "install_torch_runtime"
    assert "disabled by policy" in payload["error"]


def test_sync_openclaw_config_delegates_to_canonical_model_manager(monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "discover_models", lambda: ["qwen3.5:27b", "qwen3:8b", "deepseek-r1:8b"])
    monkeypatch.setattr(mod, "_repo_python_bin", lambda: r"C:\repo\simpleTrader_env\Scripts\python.exe")

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout="[openclaw_models] set models.providers.ollama\n[openclaw_models] gateway restart: OK\n",
            stderr="",
        )

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    msgs = mod.sync_openclaw_config(dry_run=False)

    assert captured["cmd"] == [
        r"C:\repo\simpleTrader_env\Scripts\python.exe",
        str(mod.PROJECT_ROOT / "scripts" / "openclaw_models.py"),
        "apply",
        "--strategy",
        "local-first",
        "--restart-gateway",
    ]
    assert any("Synced OpenClaw config via scripts/openclaw_models.py" in row for row in msgs)
