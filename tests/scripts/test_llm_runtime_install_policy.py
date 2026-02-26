from __future__ import annotations

import json
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
