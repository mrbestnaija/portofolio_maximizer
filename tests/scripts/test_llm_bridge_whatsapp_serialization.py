from __future__ import annotations

from pathlib import Path

import scripts.llm_multi_model_orchestrator as orch


class _DummyActivity:
    def __init__(self) -> None:
        self.events: list[dict] = []

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
