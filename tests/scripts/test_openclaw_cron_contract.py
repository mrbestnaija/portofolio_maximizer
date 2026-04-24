from __future__ import annotations

from scripts import openclaw_cron_contract as contract


def test_sanitize_cron_jobs_payload_rewrites_legacy_python_paths_and_backfills_session_target() -> None:
    payload = {
        "jobs": [
            {
                "id": "rewrite-job",
                "name": "[P1] Rewrite Legacy Path",
                "agentId": "training",
                "enabled": True,
                "payload": {
                    "kind": "agentTurn",
                    "message": ".\\simpleTrader_env\\Scripts\\python.exe scripts\\check_classifier_readiness.py --json",
                },
                "delivery": {
                    "channel": "whatsapp",
                    "fallback": {"channel": "telegram", "to": "telegram:6515478488"},
                },
            }
        ]
    }

    sanitized, report = contract.sanitize_cron_jobs_payload(
        payload,
        default_session_target="isolated",
        python_executable=r"C:\repo\simpleTrader_env_win\Scripts\python.exe",
    )

    job = sanitized["jobs"][0]
    assert job["sessionTarget"] == "isolated"
    assert "simpleTrader_env_win\\Scripts\\python.exe" in job["payload"]["message"]
    assert report["backfilled_count"] == 1
    assert report["rewritten_count"] == 1
    assert report["changed"] is True


def test_summarize_cron_jobs_counts_stale_python_paths() -> None:
    summary = contract.summarize_cron_jobs(
        {
            "jobs": [
                {
                    "id": "rewrite-job",
                    "name": "[P1] Rewrite Legacy Path",
                    "agentId": "training",
                    "enabled": True,
                    "sessionTarget": "isolated",
                    "payload": {
                        "kind": "agentTurn",
                        "message": ".\\simpleTrader_env\\Scripts\\python.exe scripts\\check_classifier_readiness.py --json",
                    },
                    "delivery": {
                        "channel": "whatsapp",
                        "fallback": {"channel": "telegram", "to": "telegram:6515478488"},
                    },
                }
            ]
        }
    )

    assert summary["status"] == "OK"
    assert summary["invalid_session_target_count"] == 0
    assert summary["delivery_fallback_ready_count"] == 1
    assert summary["delivery_fallback_invalid_count"] == 0
    assert summary["stale_python_path_count"] == 1
