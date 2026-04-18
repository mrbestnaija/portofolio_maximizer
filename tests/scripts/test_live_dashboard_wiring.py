from __future__ import annotations

from pathlib import Path


def test_live_dashboard_is_real_time_and_not_demo() -> None:
    html = Path("visualizations/live_dashboard.html").read_text(encoding="utf-8")

    # Must not ship with a fictitious demo payload.
    assert "run_id\": \"demo\"" not in html
    assert "T-3" not in html

    # Must poll the real artifact via cache-busted query param.
    assert (
        "fetch('dashboard_data.json?_=' + Date.now())" in html
        or "fetch(cacheBust(DATA_SOURCE)" in html
    )
    assert "setInterval(refreshDashboard" in html

    # Must not fabricate data when missing.
    assert "return null;" in html

    # Must include the trade panels + required DOM anchors.
    for element_id in (
        "equity-chart",
        "trades-chart",
        "trade-pnl-chart",
        "trade-ticker",
        "signals-body",
        "robustness-status",
        "robustness-eligibility",
        "robustness-coverage",
        "robustness-calibration",
        "live-denominator-status",
        "live-denominator-cohort",
        "live-denominator-linkage",
        "live-denominator-matched",
        "polling-select",
        "payload-schema",
        "payload-quant",
        "audit-open-issues",
        "audit-schema-version",
        "audit-cache-state",
        "audit-diversity",
        "audit-denominator",
        "audit-issues",
        "orchestration-status",
        "orchestration-source",
        "orchestration-mssa",
        "orchestration-garch",
        "orchestration-rank",
        "orchestration-default",
        "orchestration-context",
        "orchestration-issues",
    ):
        assert f'id="{element_id}"' in html

    assert "No robustness data available." in html
    assert "Watcher not connected." in html
    assert "No audit evidence loaded." in html


def test_live_dashboard_robustness_status_precedence_and_tone_wiring() -> None:
    html = Path("visualizations/live_dashboard.html").read_text(encoding="utf-8")

    assert "const status = robustness.overall_status || robustness.status ||" in html
    assert "statusEl.textContent = suff.status ||" not in html
    assert "statusEl.classList.add('status-warn');" in html
    assert ".status-warn { color: var(--accent-2); }" in html


def test_live_dashboard_live_denominator_wiring() -> None:
    html = Path("visualizations/live_dashboard.html").read_text(encoding="utf-8")

    assert "function renderLiveDenominator(data)" in html
    assert "const watcher = data?.live_denominator || {};" in html
    assert "renderLiveDenominator(null);" in html
    assert "renderLiveDenominator(data);" in html


def test_live_dashboard_audit_console_and_polling_wiring() -> None:
    html = Path("visualizations/live_dashboard.html").read_text(encoding="utf-8")

    assert "function renderAuditConsole(data)" in html
    assert "function validatePayloadShape(data)" in html
    assert "dashboardRefreshMs" in html
    assert "const DEFAULT_REFRESH_MS = 15000;" in html
    assert "const REFRESH_MS = 5000;" not in html
    assert "auditConsoleState.focus = btn.dataset.focus || 'all';" in html


def test_live_dashboard_operator_console_wiring() -> None:
    html = Path("visualizations/live_dashboard.html").read_text(encoding="utf-8")

    for element_id in (
        "operator-status",
        "operator-primary",
        "operator-wiring",
        "operator-gate",
        "operator-activity",
        "operator-context",
        "operator-issues",
    ):
        assert f'id="{element_id}"' in html

    assert "function renderOperatorConsole(data)" in html
    assert "renderOperatorConsole(null);" in html
    assert "renderOperatorConsole(data);" in html
    assert "No operator evidence loaded." in html
    assert "maintenance.recovery_events" in html
    assert "reconnects=" in html
    assert "fast_action=" in html


def test_live_dashboard_orchestration_health_wiring() -> None:
    html = Path("visualizations/live_dashboard.html").read_text(encoding="utf-8")

    assert "function renderOrchestrationHealth(data)" in html
    assert "renderOrchestrationHealth(null);" in html
    assert "renderOrchestrationHealth(data);" in html
    assert "orchestration_health" in html
    assert "No orchestration evidence loaded." in html
    assert "white-noise fail" in html
    assert "preprocess_health" in html
    assert "evidence=" in html
