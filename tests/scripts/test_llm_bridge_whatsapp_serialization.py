from __future__ import annotations

import json
from pathlib import Path

import scripts.llm_multi_model_orchestrator as orch


class _DummyActivity:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.tool_calls: list[dict] = []
        self.orchestrations: list[dict] = []

    def log_openclaw_event(self, channel: str, event_type: str, payload=None, model_used: str = "", latency_ms: float = 0, metadata=None) -> None:
        self.events.append(
            {
                "channel": channel,
                "event_type": event_type,
                "payload": payload or {},
                "model_used": model_used,
                "latency_ms": latency_ms,
                "metadata": metadata or {},
            }
        )

    def log_tool_call(self, **kwargs) -> None:
        self.tool_calls.append(dict(kwargs))

    def log_orchestration(self, **kwargs) -> None:
        self.orchestrations.append(dict(kwargs))

    def log_request(self, **kwargs) -> None:
        self.events.append({"event_type": "request", "metadata": dict(kwargs)})


def test_openclaw_bridge_forces_whatsapp_through_orchestrate(monkeypatch) -> None:
    activity = _DummyActivity()
    released: list[str] = []
    bridge_lock = orch._BridgeTurnLock(path=Path("dummy.lock"), token="tok-1", waited_seconds=0.0)

    def _unexpected_fast_path(**_kwargs):
        raise AssertionError("WhatsApp bridge should skip fast-path execution")

    def _unexpected_status_fast_path(_message: str) -> bool:
        raise AssertionError("WhatsApp bridge should skip legacy status fast path")

    called: dict[str, object] = {}

    monkeypatch.setattr(orch, "_get_activity_logger", lambda: activity)
    monkeypatch.setattr(orch, "_run_bridge_fast_path", _unexpected_fast_path)
    monkeypatch.setattr(orch, "_is_status_fast_path_prompt", _unexpected_status_fast_path)
    monkeypatch.setattr(orch, "_acquire_bridge_turn_lock", lambda **_kwargs: bridge_lock)
    monkeypatch.setattr(orch, "_release_bridge_turn_lock", lambda lock: released.append(lock.token if lock else ""))
    monkeypatch.setattr(orch, "_bridge_force_tool_primer", lambda _message: True)
    monkeypatch.setattr(orch, "_bridge_timeout_seconds", lambda _message, _channel: 42)
    monkeypatch.setattr(orch, "_openclaw_prompt_template_for_message", lambda _message: "")
    monkeypatch.setattr(orch, "_base_system_prompt", lambda: "system")
    monkeypatch.setattr(
        orch,
        "orchestrate",
        lambda **kwargs: (called.update(kwargs) or "orchestrated"),
    )

    result = orch.openclaw_bridge("status", channel="whatsapp", session_id="sess-w1")

    assert result == "orchestrated"
    assert called["prompt"] == "status"
    assert called["timeout_seconds"] == 42
    assert released == ["tok-1"]
    assert any(row["event_type"] == "bridge_turn_lock_acquired" for row in activity.events)


def test_openclaw_bridge_non_whatsapp_retains_existing_fast_path_behavior(monkeypatch) -> None:
    activity = _DummyActivity()
    fast_path_calls: list[str] = []

    monkeypatch.setattr(orch, "_get_activity_logger", lambda: activity)
    monkeypatch.setattr(
        orch,
        "_run_bridge_fast_path",
        lambda **kwargs: (fast_path_calls.append(str(kwargs.get("channel") or "")) or "fast-path-response"),
    )
    monkeypatch.setattr(
        orch,
        "_acquire_bridge_turn_lock",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Telegram should not acquire WhatsApp turn lock")),
    )
    monkeypatch.setattr(
        orch,
        "orchestrate",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Telegram fast path should complete before orchestrate")),
    )

    result = orch.openclaw_bridge("status", channel="telegram", session_id="sess-t1")

    assert result == "fast-path-response"
    assert fast_path_calls == ["telegram"]
    assert any(row["event_type"] == "bridge_incoming" for row in activity.events)


def test_bridge_turn_lock_serializes_whatsapp_sessions(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(orch, "_bridge_turn_lock_path", lambda _channel: tmp_path / "whatsapp_turn.lock")

    first = orch._acquire_bridge_turn_lock(
        channel="whatsapp",
        session_id="sess-1",
        wait_seconds=0.05,
        stale_seconds=600.0,
        poll_seconds=0.01,
    )
    assert first is not None
    assert first.path.exists()

    second = orch._acquire_bridge_turn_lock(
        channel="whatsapp",
        session_id="sess-2",
        wait_seconds=0.05,
        stale_seconds=600.0,
        poll_seconds=0.01,
    )
    assert second is None

    orch._release_bridge_turn_lock(first)
    assert not first.path.exists()

    third = orch._acquire_bridge_turn_lock(
        channel="whatsapp",
        session_id="sess-3",
        wait_seconds=0.05,
        stale_seconds=600.0,
        poll_seconds=0.01,
    )
    assert third is not None
    assert third.payload["session_id"] == "sess-3"
    orch._release_bridge_turn_lock(third)


def test_bridge_turn_lock_reclaims_dead_holder(monkeypatch, tmp_path: Path) -> None:
    lock_path = tmp_path / "whatsapp_turn.lock"
    lock_path.write_text(
        json.dumps(
            {
                "pid": 987654,
                "channel": "whatsapp",
                "session_id": "stale-session",
                "token": "stale-token",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(orch, "_bridge_turn_lock_path", lambda _channel: lock_path)
    monkeypatch.setattr(orch, "_process_is_running", lambda _pid: False)

    acquired = orch._acquire_bridge_turn_lock(
        channel="whatsapp",
        session_id="sess-live",
        wait_seconds=0.05,
        stale_seconds=600.0,
        poll_seconds=0.01,
    )

    assert acquired is not None
    assert acquired.payload["session_id"] == "sess-live"
    orch._release_bridge_turn_lock(acquired)


def test_openclaw_bridge_returns_busy_error_when_whatsapp_lock_times_out(monkeypatch) -> None:
    activity = _DummyActivity()
    delivered: list[str] = []

    monkeypatch.setattr(orch, "_get_activity_logger", lambda: activity)
    monkeypatch.setattr(orch, "_bridge_timeout_seconds", lambda _message, _channel: 21)
    monkeypatch.setattr(orch, "_bridge_force_tool_primer", lambda _message: False)
    monkeypatch.setattr(orch, "_openclaw_prompt_template_for_message", lambda _message: "")
    monkeypatch.setattr(orch, "_base_system_prompt", lambda: "system")
    monkeypatch.setattr(orch, "_acquire_bridge_turn_lock", lambda **_kwargs: None)
    monkeypatch.setattr(orch, "_deliver_response", lambda *, channel, to, message: delivered.append(f"{channel}:{to}:{message}"))
    monkeypatch.setattr(
        orch,
        "orchestrate",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Busy WhatsApp bridge should fail before orchestrate")),
    )

    result = orch.openclaw_bridge("check gate", channel="whatsapp", reply_to="+2347000000000", session_id="sess-busy")

    assert "exclusive qwen orchestration slot" in result
    assert delivered and any("exclusive qwen orchestration slot" in row for row in delivered)
    assert any(row["event_type"] == "bridge_turn_lock_timeout" for row in activity.events)


def test_orchestrate_returns_tool_snapshot_when_followup_round_times_out(monkeypatch) -> None:
    activity = _DummyActivity()
    delivered: list[str] = []
    ollama_calls = {"count": 0}

    monkeypatch.setattr(orch, "_ensure_health_fresh", lambda: None)
    monkeypatch.setattr(orch, "_get_activity_logger", lambda: activity)
    monkeypatch.setattr(orch, "PROGRESS_UPDATES_DEFAULT", False)
    monkeypatch.setattr(orch, "MAX_ROUNDS_DEFAULT", 2)
    monkeypatch.setattr(orch, "MAX_TOOL_CALLS_DEFAULT", 8)
    monkeypatch.setattr(orch, "CHAT_ROUND_TIMEOUT_CAP_SECONDS", 25.0)
    monkeypatch.setattr(
        orch,
        "get_best_model_for_role",
        lambda role: "qwen3:8b" if role == "orchestrator" else "deepseek-r1:8b",
    )
    monkeypatch.setattr(
        orch,
        "_runtime_plan",
        lambda **kwargs: {
            "max_rounds": kwargs["max_rounds"],
            "max_tool_calls": kwargs["max_tool_calls"],
            "timeout_seconds": kwargs["timeout_seconds"],
            "force_tool_primer": kwargs["force_tool_primer"],
            "subagent_workflow": kwargs["subagent_workflow"],
            "chat_num_predict": 256,
            "trading_critical": False,
        },
    )
    monkeypatch.setattr(orch, "_record_model_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(orch, "_base_system_prompt", lambda: "system")
    monkeypatch.setattr(orch, "_deliver_response", lambda *, channel, to, message: delivered.append(f"{channel}:{to}:{message}"))

    def fake_ollama_post(_path, _payload, timeout):
        del timeout
        ollama_calls["count"] += 1
        if ollama_calls["count"] == 1:
            return {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "function": {"name": "check_system_health", "arguments": "{}"},
                        }
                    ],
                }
            }
        raise TimeoutError("timed out")

    monkeypatch.setattr(orch, "_ollama_post", fake_ollama_post)
    monkeypatch.setattr(
        orch,
        "execute_tool_call",
        lambda tool_name, args, allow_cache=True, budget_seconds=None: json.dumps(
            {
                "status": "PASS",
                "tool": tool_name,
                "message": "Gateway healthy and WhatsApp linked.",
                "gateway_ok": True,
            }
        ),
    )
    monkeypatch.setattr(
        orch,
        "_run_reasoning_model",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Should not fall back to direct reasoning when tool evidence exists")),
    )

    result = orch.orchestrate(
        "status",
        reply_channel="whatsapp",
        reply_to="+2347000000000",
        timeout_seconds=30,
    )

    assert "evidence-first snapshot" in result
    assert "check_system_health" in result
    assert delivered and "evidence-first snapshot" in delivered[-1]


def test_run_reasoning_model_returns_health_evidence_when_all_models_fail(monkeypatch) -> None:
    activity = _DummyActivity()
    for spec in orch.MODEL_REGISTRY.values():
        monkeypatch.setattr(spec, "available", True)
        monkeypatch.setattr(spec, "error_streak", 0)
        monkeypatch.setattr(spec, "cooldown_until", 0.0)

    monkeypatch.setattr(orch, "_get_activity_logger", lambda: activity)
    monkeypatch.setattr(orch, "_get_system_health_json", lambda: json.dumps({
        "ollama_up": True,
        "gateway_ok": False,
        "profile": {"name": "builtin_default"},
        "models": {
            "qwen3:8b": {"available": True},
            "deepseek-r1:8b": {"available": False},
        },
    }))
    monkeypatch.setattr(orch, "_ollama_post", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("format")))

    result = orch._run_reasoning_model(
        model="deepseek-r1:8b",
        task="System health check",
        allow_fallback=True,
        timeout_seconds=1.0,
        max_predict=32,
    )

    assert "health: DEGRADED" in result
    assert "gateway=down" in result
