from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts import pmx_alertmanager_bridge as mod


def test_bridge_shadow_mode_logs_only_without_shadow_targets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("PMX_OBSERVABILITY_OPENCLAW_SHADOW_TARGETS", raising=False)
    monkeypatch.delenv("PMX_EMAIL_TEST_TO", raising=False)
    log_path = tmp_path / "alerts.jsonl"
    bridge = mod.AlertBridge(log_path=log_path, shadow_mode=True)
    result = bridge.handle_alertmanager_payload(
        {
            "status": "firing",
            "alerts": [
                {
                    "labels": {"alertname": "PMXOpenClawGatewayDown", "severity": "warning"},
                    "annotations": {"summary": "Gateway unavailable"},
                }
            ],
        }
    )
    assert result["delivery_mode"] == "shadow_log_only"
    assert log_path.exists()
    logged = log_path.read_text(encoding="utf-8")
    assert "PMXOpenClawGatewayDown" in logged


def test_bridge_live_mode_uses_existing_openclaw_and_email_sinks(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(mod, "_send_openclaw", lambda message, shadow_mode: (calls.append(("openclaw", shadow_mode)) or (True, "ok")))
    monkeypatch.setattr(mod, "_send_email", lambda message, shadow_mode: (calls.append(("email", shadow_mode)) or (True, "sent")))
    bridge = mod.AlertBridge(log_path=tmp_path / "alerts.jsonl", shadow_mode=False)
    result = bridge.handle_alertmanager_payload({"status": "resolved", "alerts": []})
    assert result["delivery_mode"] == "live"
    assert ("openclaw", False) in calls
    assert ("email", False) in calls


def test_bridge_http_endpoints_work(tmp_path: Path) -> None:
    bridge = mod.AlertBridge(log_path=tmp_path / "alerts.jsonl", shadow_mode=True)
    mod._BridgeHandler.bridge = bridge
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod._BridgeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/healthz") as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["status"] == "ok"

        req = urllib.request.Request(
            base + "/alertmanager",
            data=json.dumps({"status": "firing", "alerts": []}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            payload = json.loads(resp.read().decode("utf-8"))
            assert payload["shadow_mode"] is True
    finally:
        server.shutdown()
        server.server_close()
