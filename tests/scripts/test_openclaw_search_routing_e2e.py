from __future__ import annotations

from datetime import datetime, timezone

from scripts import openclaw_search_routing_e2e as e2e


def test_extract_whatsapp_reply_target_prefers_self_e164() -> None:
    payload = {
        "channels": {
            "whatsapp": {
                "self": {
                    "e164": "+15551230000",
                }
            }
        },
        "channelAccounts": {
            "whatsapp": [
                {
                    "allowFrom": ["+19998887777"],
                }
            ]
        },
    }
    assert e2e._extract_whatsapp_reply_target(payload) == "+15551230000"


def test_extract_whatsapp_reply_target_falls_back_to_allow_from() -> None:
    payload = {
        "channels": {"whatsapp": {}},
        "channelAccounts": {
            "whatsapp": [
                {
                    "allowFrom": ["+2347042437712", "+2348061573767"],
                }
            ]
        },
    }
    assert e2e._extract_whatsapp_reply_target(payload) == "+2347042437712"


def test_fast_path_event_counts_tracks_web_and_legacy_status() -> None:
    events = [
        {"event_type": e2e.FAST_PATH_WEB_START},
        {"event_type": e2e.FAST_PATH_WEB_COMPLETE},
        {"event_type": e2e.FAST_PATH_STATUS_START},
        {"event_type": e2e.FAST_PATH_STATUS_COMPLETE},
        {"event_type": "bridge_repo_fast_path_start"},
    ]
    counts = e2e._fast_path_event_counts(events)
    assert counts["web_start"] == 1
    assert counts["web_complete"] == 1
    assert counts["status_start"] == 1
    assert counts["status_complete"] == 1


def test_bridge_output_passed_requires_web_search_pass_marker() -> None:
    raw = "[orchestrator]\n--- Response ---\nweb search: PASS | provider=tavily | attempts=1\n"
    assert e2e._bridge_output_passed(raw) is True
    assert e2e._bridge_output_passed("web search: FAIL | provider=none") is False


def test_channel_ready_rejects_unlinked_whatsapp() -> None:
    payload = {
        "channels": {
            "whatsapp": {
                "configured": True,
                "linked": False,
                "running": True,
                "connected": True,
            }
        }
    }
    ok, reason = e2e._channel_ready(payload, "whatsapp")
    assert ok is False
    assert reason == "whatsapp_not_linked"


def test_parse_iso8601_accepts_z_suffix() -> None:
    parsed = e2e._parse_iso8601("2026-02-20T06:10:39.895890Z")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.astimezone(timezone.utc) <= datetime.now(timezone.utc)
