from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_HTML = ROOT / "visualizations" / "live_dashboard.html"


def _read_dashboard() -> str:
    return DASHBOARD_HTML.read_text(encoding="utf-8")


def test_dashboard_has_no_invalid_placeholder_tokens() -> None:
    text = _read_dashboard()
    assert "[ ]" not in text, "Invalid token pattern '[ ]' breaks dashboard JavaScript parsing."


def test_dashboard_unknown_performance_renders_na_not_zero() -> None:
    text = _read_dashboard()
    assert "const performanceUnknown" in text
    assert "No performance data" in text
    assert "N/A trades" in text
