from __future__ import annotations

import json

from scripts import verify_openclaw_config as mod


def test_load_cfg_accepts_utf8_bom(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "openclaw.json"
    cfg_path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"tools": {"exec": {"host": "node"}}}).encode("utf-8"))
    monkeypatch.setattr(mod, "OPENCLAW_JSON", cfg_path)

    cfg = mod._load_cfg()
    assert cfg["tools"]["exec"]["host"] == "node"


def test_describe_agent_tools_policy_marks_explicit_policy() -> None:
    policy = mod._describe_agent_tools_policy({"allow": ["exec", "read"], "deny": ["browser"]})
    assert policy == "tools.policy=explicit"
