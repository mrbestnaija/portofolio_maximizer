from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_auto_trader_core_sanitizes_forecast_audits_before_running() -> None:
    text = (_repo_root() / "bash" / "production_cron.sh").read_text(encoding="utf-8")

    assert "sanitize_forecast_audits()" in text
    sanitize_idx = text.index("sanitize_production_forecast_audits.py")
    trader_idx = text.index('scripts/run_auto_trader.py --tickers "${CORE_TICKERS}"')
    assert sanitize_idx < trader_idx
    assert "--apply" in text
    assert "scripts/dashboard_db_bridge.py" in text
    assert "scripts/production_audit_gate.py" in text
    assert "scripts/project_runtime_status.py --pretty" in text
    assert "runtime_status_latest.json" in text


def test_production_cron_preflights_registry_and_routes_family_calibration() -> None:
    text = (_repo_root() / "bash" / "production_cron.sh").read_text(encoding="utf-8")

    assert "canonical_source_registry.yml" in text
    assert "refusing to emit canonical snapshot" in text
    assert "scripts/family_calibration_writer.py" in text
    assert "family_calibration)" in text
    assert "scripts/outcome_linkage_attribution_report.py" in text
    assert text.index("emit_canonical_snapshot: emit_canonical_snapshot.py") < text.index(
        "scripts/outcome_linkage_attribution_report.py"
    )


def test_production_cron_fails_closed_when_core_evidence_artifacts_fail() -> None:
    text = (_repo_root() / "bash" / "production_cron.sh").read_text(encoding="utf-8")

    assert "cycle_rc=0" in text
    assert text.count('exit "${cycle_rc}"') >= 2
    assert "emit_canonical_snapshot failed after auto_trader." in text
    assert "emit_canonical_snapshot failed after auto_trader_core." in text
    assert "project_runtime_status failed after auto_trader_core." in text


def test_cron_runtime_uses_wsl_simpletrader_interpreter() -> None:
    cron_text = (_repo_root() / "bash" / "production_cron.sh").read_text(encoding="utf-8")
    common_text = (_repo_root() / "bash" / "lib" / "common.sh").read_text(encoding="utf-8")

    assert "pmx_resolve_python" in cron_text
    assert "pmx_require_venv_python" in common_text
    assert "simpleTrader_env/bin/python" in common_text
    assert "python.exe" not in cron_text
