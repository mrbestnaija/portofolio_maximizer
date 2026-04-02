from __future__ import annotations

import json
from pathlib import Path

from scripts import openclaw_production_readiness as mod


def _write_production_gate_artifact(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-14T20:04:49Z",
                "phase3_ready": False,
                "phase3_reason": "GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL",
                "pass_semantics_version": 3,
                "production_profitability_gate": {
                    "gate_semantics_status": "FAIL",
                },
                "inputs": {
                    "audit_dir": "logs/forecast_audits/production",
                    "max_files": 50,
                    "include_research": False,
                },
                "lift_gate": {
                    "pass": False,
                    "violation_rate": 0.625,
                    "max_violation_rate": 0.35,
                    "lift_fraction": 0.25,
                    "min_lift_fraction": 0.25,
                },
                "profitability_proof": {
                    "pass": False,
                    "profit_factor": 0.60,
                    "win_rate": 0.39,
                    "total_pnl": -986.14,
                    "closed_trades": 41,
                    "trading_days": 11,
                },
                "readiness": {
                    "gates_pass": False,
                    "linkage_pass": False,
                    "evidence_hygiene_pass": True,
                    "outcome_matched": 0,
                    "outcome_eligible": 1,
                    "matched_over_eligible": 0.0,
                    "non_trade_context_count": 0,
                    "invalid_context_count": 0,
                    "linkage_waterfall": {
                        "raw_candidates": 28,
                        "production_only": 28,
                        "linked": 1,
                        "hygiene_pass": 28,
                        "matched": 0,
                        "excluded_non_trade_context": 0,
                        "excluded_invalid_context": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_gate_truth_posture_detects_skip_policy_and_phase3_drift(tmp_path: Path) -> None:
    gate_artifact = tmp_path / "gate_status_latest.json"
    gate_artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-14T17:20:45Z",
                "overall_passed": False,
                "phase3_ready": True,
                "phase3_reason": "READY",
                "skipped_optional_gates": 2,
                "max_skipped_optional_gates": 1,
                "skipped_gate_labels": ["check_quant_validation_health", "institutional_unattended_gate"],
            }
        ),
        encoding="utf-8",
    )
    production_artifact = tmp_path / "production_gate_latest.json"
    production_artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-14T17:30:28Z",
                "pass_semantics_version": 3,
                "warmup_expired": False,
                "phase3_ready": False,
                "phase3_reason": "GATES_FAIL,THIN_LINKAGE,EVIDENCE_HYGIENE_FAIL",
                "production_profitability_gate": {
                    "gate_semantics_status": "FAIL",
                },
            }
        ),
        encoding="utf-8",
    )

    truth, blockers, warnings = mod._gate_truth_posture(
        gate_artifact_path=gate_artifact,
        production_gate_artifact_path=production_artifact,
        refresh_production_gate=False,
        timeout_seconds=5.0,
    )

    codes = {row["code"] for row in blockers}
    assert "gate_skip_policy_failed" in codes
    assert "stale_gate_artifact_phase3_drift" in codes
    assert truth["freshest_phase3_source"] == "production_gate_latest"
    assert truth["effective_phase3_ready"] is False
    assert warnings == []


def test_production_gate_snapshot_prefers_strict_phase3_fields(tmp_path: Path) -> None:
    production_artifact = tmp_path / "production_gate_latest.json"
    production_artifact.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-03-14T17:30:28Z",
                "pass_semantics_version": 3,
                "warmup_expired": False,
                "phase3_ready": True,
                "phase3_reason": "READY",
                "phase3_strict_ready": False,
                "phase3_strict_reason": "READY,GATE_SEMANTICS_INCONCLUSIVE_ALLOWED",
                "production_profitability_gate": {
                    "pass": True,
                    "strict_pass": False,
                    "gate_semantics_status": "INCONCLUSIVE_ALLOWED",
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = mod._production_gate_snapshot(production_artifact)

    assert snapshot["phase3_ready"] is False
    assert snapshot["phase3_reason"].endswith("GATE_SEMANTICS_INCONCLUSIVE_ALLOWED")
    assert snapshot["phase3_legacy_ready"] is True
    assert snapshot["gate_semantics_status"] == "INCONCLUSIVE_ALLOWED"


def test_gate_decomposition_snapshot_refreshes_stale_report(tmp_path: Path) -> None:
    production_artifact = tmp_path / "production_gate_latest.json"
    _write_production_gate_artifact(production_artifact)
    summary_cache = tmp_path / "latest_summary.json"
    summary_cache.write_text(
        json.dumps(
            {
                "audit_dir": "logs/forecast_audits/production",
                "max_files": 50,
                "scope": {"include_research": False},
                "dataset_windows": [],
                "window_counts": {
                    "n_outcome_windows_invalid_context": 0,
                    "n_outcome_windows_non_trade_context": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    decomposition_json = tmp_path / "production_gate_decomposition_latest.json"
    decomposition_md = tmp_path / "production_gate_decomposition_latest.md"
    decomposition_json.write_text(
        json.dumps(
            {
                "generated_utc": "2026-03-01T00:00:00+00:00",
                "source_artifact": str(production_artifact.resolve()),
                "source_timestamp_utc": "2026-03-01T00:00:00Z",
                "phase3_ready": True,
                "phase3_reason": "STALE",
            }
        ),
        encoding="utf-8",
    )

    snapshot, blockers, warnings = mod._gate_decomposition_snapshot(
        production_gate_artifact_path=production_artifact,
        decomposition_artifact_path=decomposition_json,
        decomposition_markdown_path=decomposition_md,
        summary_cache_path=summary_cache,
    )

    assert blockers == []
    assert warnings == []
    assert snapshot["refresh_result"]["refreshed"] is True
    assert snapshot["refresh_result"]["reason"] == "source_timestamp_mismatch"
    assert snapshot["component_status"] == [
        {"name": "PERFORMANCE_BLOCKER", "pass": False},
        {"name": "LINKAGE_BLOCKER", "pass": False},
        {"name": "HYGIENE_BLOCKER", "pass": True},
    ]
    assert decomposition_md.exists()


def test_collect_openclaw_production_readiness_suppresses_noisy_helper_output(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(mod, "_gate_truth_posture", lambda **kwargs: ({"gate_artifact": {}, "production_gate_artifact": {}, "freshest_phase3_source": "gate_status_latest", "effective_phase3_ready": None, "effective_phase3_reason": "", "drift_detected": False}, [], []))
    monkeypatch.setattr(mod, "_openclaw_model_posture", lambda path: ({}, [], []))
    monkeypatch.setattr(mod, "_security_posture", lambda: ({}, [], []))
    monkeypatch.setattr(mod, "_openclaw_exec_env_posture", lambda: ({"ok": True}, []))
    monkeypatch.setattr(mod, "_openclaw_regression_posture", lambda timeout_seconds: ({"status": "PASS"}, [], []))

    def _chatty_capital():
        print("NOISY STDOUT FROM HELPER")
        print("NOISY STDERR FROM HELPER", file=__import__("sys").stderr)
        return {
            "ready": False,
            "verdict": "FAIL",
            "reasons": ["R2: gate artifact overall_passed=False"],
            "warnings": [],
            "metrics": {},
        }

    monkeypatch.setattr(mod.capital_mod, "run_capital_readiness", _chatty_capital)

    rc = mod.main(["--json"])
    assert rc == 1

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["action"] == "assess_production_readiness"
    captured = payload["capital_readiness"]["captured_output"]
    assert "NOISY STDOUT FROM HELPER" in "\n".join(captured["stdout_tail"])
    assert "NOISY STDERR FROM HELPER" in "\n".join(captured["stderr_tail"])


def test_action_guide_cli_outputs_requested_human_steps(monkeypatch, capsys) -> None:
    payload = {
        "readiness_status": "FAIL",
        "human_action_guides": [
            {
                "id": "approval_token",
                "title": "Set a non-default approval token",
                "cli_hint": "python scripts/openclaw_production_readiness.py --action-guide approval_token",
                "steps": ["step one"],
                "commands": ["cmd one"],
            },
            {
                "id": "capital_readiness",
                "title": "Triage economics and evidence depth",
                "cli_hint": "python scripts/openclaw_production_readiness.py --action-guide capital_readiness",
                "steps": ["step two"],
                "commands": ["cmd two"],
            },
        ],
    }
    monkeypatch.setattr(mod, "collect_openclaw_production_readiness", lambda **kwargs: payload)

    rc = mod.main(["--action-guide", "approval_token", "--json"])
    assert rc == 0

    out = capsys.readouterr().out
    guide_payload = json.loads(out)
    assert guide_payload["action"] == "production_readiness_action_guide"
    assert guide_payload["requested_guide"] == "approval_token"
    assert len(guide_payload["guides"]) == 1
    assert guide_payload["guides"][0]["id"] == "approval_token"


def test_refresh_production_gate_artifact_uses_repo_python(monkeypatch, tmp_path: Path) -> None:
    artifact_path = tmp_path / "production_gate_latest.json"
    invoked: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        invoked["cmd"] = list(cmd)
        artifact_path.write_text(
            json.dumps(
                {
                    "timestamp_utc": "2026-03-15T10:00:00Z",
                    "phase3_ready": False,
                    "phase3_reason": "GATES_FAIL",
                    "production_profitability_gate": {"gate_semantics_status": "FAIL"},
                }
            ),
            encoding="utf-8",
        )
        return __import__("subprocess").CompletedProcess(cmd, 1, stdout="", stderr="")

    monkeypatch.setattr(mod, "_repo_python_bin", lambda: r"C:\repo\simpleTrader_env\Scripts\python.exe")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    result, blockers = mod._refresh_production_gate_artifact(
        artifact_path=artifact_path,
        timeout_seconds=5.0,
    )

    assert blockers == []
    assert result["ok"] is True
    assert invoked["cmd"][0] == r"C:\repo\simpleTrader_env\Scripts\python.exe"


def test_openclaw_model_posture_flags_legacy_openai_compat_ollama(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "models": {
                    "providers": {
                        "ollama": {
                            "baseUrl": "http://127.0.0.1:11434/v1",
                            "api": "openai-completions",
                            "models": [{"id": "qwen3:8b", "name": "qwen3:8b"}],
                        }
                    }
                },
                "agents": {
                    "defaults": {
                        "model": {"primary": "ollama/qwen3:8b"},
                        "models": {"ollama/qwen3:8b": {}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_discover_ollama_models", lambda base_url, timeout_seconds=2.0: ["qwen3:8b"])

    snapshot, blockers, warnings = mod._openclaw_model_posture(config_path)

    codes = {row["code"] for row in blockers}
    assert snapshot["ollama_provider_api"] == "openai-completions"
    assert "ollama_provider_not_native" in codes
    assert warnings == []
