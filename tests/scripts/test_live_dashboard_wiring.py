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
    for element_id in ("equity-chart", "trades-chart", "trade-pnl-chart", "trade-ticker", "signals-body"):
        assert f'id="{element_id}"' in html
