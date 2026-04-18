from __future__ import annotations

import json
from pathlib import Path

from scripts import verify_openclaw_config as mod


def test_load_cfg_accepts_utf8_bom(tmp_path, monkeypatch) -> None:
    cfg_path = tmp_path / "openclaw.json"
    cfg_path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"tools": {"exec": {"host": "node"}}}).encode("utf-8"))
    monkeypatch.setattr(mod, "OPENCLAW_JSON", cfg_path)

    cfg = mod._load_cfg()
    assert cfg["tools"]["exec"]["host"] == "node"


def test_main_fails_closed_when_openclaw_json_is_malformed(tmp_path, monkeypatch, capsys) -> None:
    cfg_path = tmp_path / "openclaw.json"
    cfg_path.write_text("{\"gateway\": {\"mode\": \"local\",}}\n", encoding="utf-8")
    monkeypatch.setattr(mod, "OPENCLAW_JSON", cfg_path)

    rc = mod.main()
    captured = capsys.readouterr()

    assert rc == 1
    assert "OpenClaw config unreadable" in captured.out
    assert "Traceback" not in captured.err


def test_describe_agent_tools_policy_marks_explicit_policy() -> None:
    policy = mod._describe_agent_tools_policy({"allow": ["exec", "read"], "deny": ["browser"]})
    assert policy == "tools.policy=explicit"


def _valid_cfg(repo_root: Path) -> dict:
    return {
        "models": {
            "providers": {
                "ollama": {
                    "baseUrl": "http://127.0.0.1:11434",
                    "api": "ollama",
                    "models": [
                        {
                            "id": "qwen3:8b",
                            "name": "qwen3:8b",
                            "reasoning": False,
                            "input": "text",
                            "cost": 0.0,
                            "contextWindow": 4096,
                            "maxTokens": 1024,
                        }
                    ],
                }
            }
        },
        "agents": {
            "defaults": {
                "model": {
                    "primary": "ollama/qwen3:8b",
                    "fallbacks": ["ollama/deepseek-r1:8b"],
                },
                "models": ["ollama/qwen3:8b", "ollama/deepseek-r1:8b"],
                "contextTokens": 4096,
                "bootstrapMaxChars": 20000,
                "compaction": {"mode": "safeguard"},
            },
            "list": [
                {
                    "id": "ops",
                    "default": True,
                    "agentDir": "agent-ops",
                    "workspace": str(repo_root),
                    "model": "ollama/qwen3:8b",
                    "tools": {"deny": ["browser"]},
                },
                {
                    "id": "trading",
                    "agentDir": "agent-trading",
                    "workspace": str(repo_root),
                    "model": "ollama/qwen3:8b",
                    "tools": {"deny": ["browser"]},
                },
                {
                    "id": "training",
                    "agentDir": "agent-training",
                    "workspace": str(repo_root),
                    "model": "ollama/qwen3:8b",
                    "tools": {"deny": ["browser"]},
                },
                {
                    "id": "notifier",
                    "agentDir": "agent-notifier",
                    "workspace": str(repo_root),
                    "model": "ollama/qwen3:8b",
                    "tools": {"deny": ["browser"]},
                },
            ],
        },
        "gateway": {
            "mode": "local",
            "port": 18789,
            "bind": "loopback",
            "auth": {"mode": "token", "token": "secret-token"},
        },
        "auth": {"profiles": {}},
        "plugins": {"entries": {"qwen-portal-auth": {"enabled": False}}},
        "channels": {
            "whatsapp": {
                "dmPolicy": "pairing",
                "groupPolicy": "allowlist",
                "allowFrom": [],
                "selfChatMode": False,
                "debounceMs": 1000,
                "sendReadReceipts": True,
                "accounts": {"default": {"enabled": True}},
            }
        },
        "session": {"dmScope": "main"},
        "bindings": [
            {
                "agentId": "ops",
                "match": {"channel": "whatsapp", "accountId": "default"},
            }
        ],
        "tools": {"exec": {"host": "gateway"}, "agentToAgent": {"enabled": False}},
    }


def _valid_env(*, edge_safe: bool = True) -> dict:
    env = {
        "OPENCLAW_LOCAL_ONLY": "1",
        "OPENCLAW_OLLAMA_MODEL_ORDER": "qwen3:8b,deepseek-r1:8b",
        "OPENCLAW_AUTONOMY_GUARD_ENABLED": "1",
        "OPENCLAW_AUTONOMY_BLOCK_INJECTION_PATTERNS": "1",
        "OPENCLAW_APPROVE_HIGH_RISK": "1",
    }
    if edge_safe:
        env["PMX_EDGE_SAFE_RUNTIME"] = "1"
    return env


def test_main_rejects_malformed_agent_turn_cron_jobs(tmp_path: Path, monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cron_path = tmp_path / "jobs.json"
    cron_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "bad-job",
                        "name": "[P2] Gate and Readiness Check",
                        "agentId": "trading",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "*/10 * * * *"},
                        "payload": {"kind": "agentTurn", "message": "do work"},
                        "delivery": {"channel": "whatsapp"},
                        "state": {"consecutiveErrors": 0, "lastStatus": "pending"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "OPENCLAW_CRON_JOBS", cron_path)
    monkeypatch.setattr(mod, "_load_cfg", lambda: _valid_cfg(repo_root))
    monkeypatch.setattr(mod, "_load_env", lambda: _valid_env(edge_safe=True))
    monkeypatch.setattr(
        mod,
        "_load_llm_config",
        lambda: {"llm": {"active_model": "qwen3:8b", "models": {"primary": {"name": "qwen3:8b"}}}},
    )

    rc = mod.main()
    stdout = capsys.readouterr().out

    assert rc == 1
    assert "missing sessionTarget" in stdout
    assert "Cron jobs contain malformed agentTurn records" in stdout


def test_main_rejects_non_string_session_target_cron_jobs(tmp_path: Path, monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cron_path = tmp_path / "jobs.json"
    cron_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "bad-job",
                        "name": "[P2] Gate and Readiness Check",
                        "agentId": "trading",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "*/10 * * * *"},
                        "sessionTarget": 123,
                        "payload": {"kind": "agentTurn", "message": "do work"},
                        "delivery": {"channel": "whatsapp"},
                        "state": {"consecutiveErrors": 0, "lastStatus": "pending"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "OPENCLAW_CRON_JOBS", cron_path)
    monkeypatch.setattr(mod, "_load_cfg", lambda: _valid_cfg(repo_root))
    monkeypatch.setattr(mod, "_load_env", lambda: _valid_env(edge_safe=True))
    monkeypatch.setattr(
        mod,
        "_load_llm_config",
        lambda: {"llm": {"active_model": "qwen3:8b", "models": {"primary": {"name": "qwen3:8b"}}}},
    )

    rc = mod.main()
    stdout = capsys.readouterr().out

    assert rc == 1
    assert "invalid sessionTarget" in stdout
    assert "Cron jobs contain malformed agentTurn records" in stdout


def test_main_rejects_edge_safe_runtime_conflicts(tmp_path: Path, monkeypatch, capsys) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    cron_path = tmp_path / "jobs.json"
    cron_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "good-job",
                        "name": "[P1] Healthy Job",
                        "agentId": "ops",
                        "enabled": True,
                        "schedule": {"kind": "cron", "expr": "0 * * * *"},
                        "sessionTarget": "isolated",
                        "payload": {"kind": "agentTurn", "message": "ok"},
                        "delivery": {"channel": "whatsapp", "fallback": {"channel": "telegram", "to": "+2347"}},
                        "state": {"consecutiveErrors": 0, "lastStatus": "success"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "OPENCLAW_CRON_JOBS", cron_path)
    monkeypatch.setattr(mod, "_load_cfg", lambda: _valid_cfg(repo_root))
    monkeypatch.setattr(
        mod,
        "_load_llm_config",
        lambda: {"llm": {"active_model": "qwen3:8b", "models": {"primary": {"name": "qwen3:8b"}}}},
    )
    monkeypatch.setattr(
        mod,
        "_load_env",
        lambda: {
            **_valid_env(edge_safe=False),
            "PMX_EDGE_SAFE_RUNTIME": "1",
            "OPENCLAW_LOCAL_ONLY": "0",
            "ENABLE_PARALLEL_FORECASTS": "1",
            "ENABLE_PARALLEL_TICKER_PROCESSING": "1",
            "ENABLE_GPU_PARALLEL": "1",
        },
    )

    rc = mod.main()
    stdout = capsys.readouterr().out

    assert rc == 1
    assert "PMX_EDGE_SAFE_RUNTIME requires OPENCLAW_LOCAL_ONLY=1" in stdout
    assert "ENABLE_PARALLEL_FORECASTS" in stdout
    assert "ENABLE_PARALLEL_TICKER_PROCESSING" in stdout
    assert "ENABLE_GPU_PARALLEL" in stdout
